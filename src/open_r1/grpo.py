# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import json
from pathlib import Path
from typing import Any

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.methods import (
    build_amsb_plan,
    build_mgrpo_layer2_prompt,
    compute_mgrpo_transition_metrics,
    compute_semantic_entropy,
    modulation_factor,
)
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config


logger = logging.getLogger(__name__)

CPU_UNSAFE_REWARD_FUNCS = {"code", "binary_code", "ioi_code", "cf_code"}


def is_local_dry_run(script_args: GRPOScriptArguments) -> bool:
    return script_args.dry_run or script_args.runtime_profile == "local-dryrun"


def apply_runtime_profile_overrides(
    script_args: GRPOScriptArguments, training_args: GRPOConfig, model_args: ModelConfig
):
    if not is_local_dry_run(script_args):
        return

    unsafe_reward_funcs = sorted(
        set(script_args.reward_funcs) & CPU_UNSAFE_REWARD_FUNCS
    )
    if unsafe_reward_funcs:
        raise ValueError(
            "Dry-run profile cannot use reward functions that require external code execution: "
            f"{unsafe_reward_funcs}"
        )

    logger.info("Applying local dry-run runtime overrides.")
    if hasattr(training_args, "use_vllm"):
        training_args.use_vllm = False
    training_args.do_eval = False
    training_args.push_to_hub = False
    training_args.report_to = []
    if hasattr(training_args, "bf16"):
        training_args.bf16 = False
    if hasattr(training_args, "fp16"):
        training_args.fp16 = False

    if not script_args.dry_run_skip_train and training_args.max_steps < 1:
        training_args.max_steps = 1

    if hasattr(training_args, "num_generations") and training_args.num_generations < 2:
        training_args.num_generations = 2

    if model_args.attn_implementation == "flash_attention_2":
        model_args.attn_implementation = "eager"

    if model_args.torch_dtype in {"bfloat16", "float16"}:
        model_args.torch_dtype = "float32"


def limit_dataset_for_dry_run(dataset, script_args: GRPOScriptArguments):
    if not is_local_dry_run(script_args):
        return dataset

    max_samples = script_args.dry_run_max_samples
    target_splits = [script_args.dataset_train_split, script_args.dataset_test_split]

    for split_name in target_splits:
        if split_name in dataset:
            sample_count = min(max_samples, len(dataset[split_name]))
            dataset[split_name] = dataset[split_name].select(range(sample_count))
            logger.info(
                f"Dry-run limited split '{split_name}' to {sample_count} samples"
            )

    return dataset


def write_dry_run_summary(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    dataset,
    method_diagnostics: dict[str, Any],
):
    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "runtime_profile": script_args.runtime_profile,
        "method": script_args.method,
        "dry_run": True,
        "dry_run_skip_train": script_args.dry_run_skip_train,
        "train_split": script_args.dataset_train_split,
        "train_samples": len(dataset[script_args.dataset_train_split]),
        "reward_funcs": script_args.reward_funcs,
        "method_diagnostics": method_diagnostics,
        "hub_model_id": getattr(training_args, "hub_model_id", None),
        "output_dir": training_args.output_dir,
    }

    summary_path = output_dir / "dry_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Wrote dry-run summary to {summary_path}")


def apply_method_features(
    dataset,
    script_args: GRPOScriptArguments,
    base_seed: int = 0,
):
    """Attach method-specific helper fields to the dataset for dry-run and logging."""
    method_name = script_args.method
    base_seed = int(base_seed)
    diagnostics: dict[str, Any] = {"method": method_name}

    if method_name == "vanilla":
        return dataset, diagnostics

    for split_name in dataset.keys():
        split_dataset = dataset[split_name]

        if method_name == "mgrpo":
            guiding_phrases = script_args.mgrpo_guiding_phrases or [
                "Re-check the final answer carefully and correct any mistakes.",
            ]

            def _add_mgrpo_fields(example, idx):
                prompt_value = str(example.get(script_args.dataset_prompt_column, ""))
                phrase = guiding_phrases[idx % len(guiding_phrases)]
                return {
                    "mgrpo_guiding_phrase": phrase,
                    "mgrpo_layer2_prompt": build_mgrpo_layer2_prompt(
                        prompt_value,
                        "Initial answer unavailable during preprocessing.",
                        phrase,
                    ),
                }

            split_dataset = split_dataset.map(_add_mgrpo_fields, with_indices=True)
            diagnostics[f"{split_name}_guiding_phrase_count"] = len(guiding_phrases)
            diagnostics[f"{split_name}_layer2_prompt_count"] = len(split_dataset)

            # If correction labels are present in data, compute transition metrics.
            if (
                "mgrpo_turn1_correct" in split_dataset.column_names
                and "mgrpo_turn2_correct" in split_dataset.column_names
            ):
                turn1 = [bool(v) for v in split_dataset["mgrpo_turn1_correct"]]
                turn2 = [bool(v) for v in split_dataset["mgrpo_turn2_correct"]]
                metrics = compute_mgrpo_transition_metrics(turn1, turn2)
                for key, value in metrics.items():
                    diagnostics[f"{split_name}_{key}"] = value

            dataset[split_name] = split_dataset
            continue

        if method_name == "seed":

            def _add_seed_fields(example):
                candidates = example.get("seed_candidate_completions", None)
                if isinstance(candidates, list) and candidates:
                    candidate_values = [str(v) for v in candidates]
                else:
                    candidate_values = [
                        str(example.get(script_args.dataset_prompt_column, ""))
                    ]

                entropy = compute_semantic_entropy(
                    candidate_values,
                    normalize_entropy=script_args.seed_entropy_normalize,
                )
                return {
                    "seed_entropy_hint": entropy,
                    "seed_modulation_hint": modulation_factor(
                        entropy,
                        script_args.seed_alpha,
                        script_args.seed_entropy_modulation,
                    ),
                }

            split_dataset = split_dataset.map(_add_seed_fields)

            entropy_values = (
                split_dataset["seed_entropy_hint"] if len(split_dataset) else []
            )
            modulation_values = (
                split_dataset["seed_modulation_hint"] if len(split_dataset) else []
            )
            if entropy_values:
                diagnostics[f"{split_name}_seed_entropy_mean"] = sum(
                    entropy_values
                ) / len(entropy_values)
                diagnostics[f"{split_name}_seed_entropy_max"] = max(entropy_values)
            if modulation_values:
                diagnostics[f"{split_name}_seed_modulation_mean"] = sum(
                    modulation_values
                ) / len(modulation_values)

            dataset[split_name] = split_dataset

        if method_name == "amsb":

            def _add_amsb_fields(example, idx):
                candidates = example.get("amsb_candidate_completions", None)
                if isinstance(candidates, list) and candidates:
                    candidate_values = [str(value) for value in candidates]
                else:
                    candidate_values = [
                        str(example.get(script_args.dataset_prompt_column, ""))
                    ]

                candidate_labels = example.get("amsb_candidate_labels", None)
                if isinstance(candidate_labels, list):
                    bool_labels = [bool(value) for value in candidate_labels]
                else:
                    bool_labels = None

                plan = build_amsb_plan(
                    candidate_completions=candidate_values,
                    candidate_correctness=bool_labels,
                    balanced_group_size=script_args.amsb_balanced_group_size,
                    max_error_clusters=script_args.amsb_max_error_clusters,
                    entropy_scale_mode=script_args.amsb_entropy_scale_mode,
                    entropy_temperature=script_args.amsb_entropy_temperature,
                    seed=base_seed + idx,
                )

                return {
                    "amsb_reflection_prompt": script_args.amsb_reflection_prompt,
                    "amsb_entropy_hint": plan["entropy"],
                    "amsb_scale_hint": plan["scale"],
                    "amsb_correct_pool_size": plan["correct_pool_size"],
                    "amsb_error_pool_size": plan["error_pool_size"],
                    "amsb_error_cluster_count": plan["error_cluster_count"],
                    "amsb_top_error_cluster_count": plan["top_error_cluster_count"],
                    "amsb_balanced_group_size": plan["balanced_group_size"],
                }

            split_dataset = split_dataset.map(_add_amsb_fields, with_indices=True)

            entropy_values = (
                split_dataset["amsb_entropy_hint"] if len(split_dataset) else []
            )
            scale_values = (
                split_dataset["amsb_scale_hint"] if len(split_dataset) else []
            )
            top_cluster_values = (
                split_dataset["amsb_top_error_cluster_count"]
                if len(split_dataset)
                else []
            )
            if entropy_values:
                diagnostics[f"{split_name}_amsb_entropy_mean"] = sum(
                    entropy_values
                ) / len(entropy_values)
                diagnostics[f"{split_name}_amsb_entropy_max"] = max(entropy_values)
            if scale_values:
                diagnostics[f"{split_name}_amsb_scale_mean"] = sum(scale_values) / len(
                    scale_values
                )
                diagnostics[f"{split_name}_amsb_scale_min"] = min(scale_values)
            if top_cluster_values:
                diagnostics[f"{split_name}_amsb_top_error_clusters_mean"] = sum(
                    top_cluster_values
                ) / len(top_cluster_values)

            diagnostics[f"{split_name}_amsb_balanced_group_size"] = (
                script_args.amsb_balanced_group_size
            )
            dataset[split_name] = split_dataset
            continue

    return dataset, diagnostics


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    apply_runtime_profile_overrides(script_args, training_args, model_args)

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    dataset = get_dataset(script_args)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    ##############
    # Load model #
    ##############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)

    # Format into conversation
    def make_conversation(example, prompt_column: str = script_args.dataset_prompt_column):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")

        prompt.append({"role": "user", "content": example[prompt_column]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    dataset = limit_dataset_for_dry_run(dataset, script_args)
    dataset, method_diagnostics = apply_method_features(
        dataset,
        script_args,
        base_seed=training_args.seed,
    )

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args, script_args),
        processing_class=tokenizer,
    )

    if is_local_dry_run(script_args) and script_args.dry_run_skip_train:
        logger.info(
            "Dry-run validation completed. Skipping trainer.train() by configuration."
        )
        write_dry_run_summary(script_args, training_args, dataset, method_diagnostics)
        return

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["method"] = script_args.method
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    for key, value in method_diagnostics.items():
        if isinstance(value, (int, float)):
            metrics[f"method/{key}"] = value
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
