#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import subprocess
from copy import deepcopy
from typing import List

from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from .evaluation import run_benchmark_jobs
from .hub import push_to_hub_revision


def is_slurm_available() -> bool:
    # returns true if a slurm queueing system is available
    try:
        subprocess.run(["sinfo"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


class DummyConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class PushToHubRevisionCallback(TrainerCallback):
    def __init__(self, model_config) -> None:
        self.model_config = model_config

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            global_step = state.global_step

            # WARNING: if you use dataclasses.replace(args, ...) the accelerator dist state will be broken, so I do this workaround
            # Also if you instantiate a new SFTConfig, the accelerator dist state will be broken
            dummy_config = DummyConfig(
                hub_model_id=args.hub_model_id,
                hub_model_revision=f"{args.hub_model_revision}-step-{global_step:09d}",
                output_dir=f"{args.output_dir}/checkpoint-{global_step}",
                system_prompt=args.system_prompt,
            )

            future = push_to_hub_revision(
                dummy_config, extra_ignore_patterns=["*.pt"]
            )  # don't push the optimizer states

            if is_slurm_available():
                dummy_config.benchmarks = args.benchmarks

                def run_benchmark_callback(_):
                    print(f"Checkpoint {global_step} pushed to hub.")
                    run_benchmark_jobs(dummy_config, self.model_config)

                future.add_done_callback(run_benchmark_callback)


class MethodMetricsCallback(TrainerCallback):
    """Logs static method metadata for MGRPO/SEED runs."""

    def __init__(self, script_args) -> None:
        self.script_args = script_args
        self.has_logged = False

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.has_logged or not state.is_world_process_zero:
            return

        method_name = getattr(self.script_args, "method", "vanilla")
        payload = {
            "method/name": method_name,
            "method/runtime_profile": getattr(
                self.script_args, "runtime_profile", "default"
            ),
        }

        if method_name == "mgrpo":
            payload["method/mgrpo_num_layer2_generations"] = getattr(
                self.script_args,
                "mgrpo_num_layer2_generations",
                1,
            )
            payload["method/mgrpo_num_guiding_phrases"] = len(
                getattr(self.script_args, "mgrpo_guiding_phrases", []) or []
            )
        elif method_name == "seed":
            payload["method/seed_entropy_modulation"] = getattr(
                self.script_args,
                "seed_entropy_modulation",
                "linear",
            )
            payload["method/seed_alpha"] = getattr(self.script_args, "seed_alpha", 1.0)

        if "model" in kwargs and hasattr(kwargs["model"], "config"):
            kwargs["model"].config.method_metadata = deepcopy(payload)

        # We print once to keep a deterministic trace even when W&B is disabled.
        print(f"[method-metadata] {payload}")
        self.has_logged = True


CALLBACKS = {
    "push_to_hub_revision": PushToHubRevisionCallback,
}


def get_callbacks(
    train_config, model_config, script_args=None
) -> List[TrainerCallback]:
    callbacks = []
    for callback_name in train_config.callbacks:
        if callback_name not in CALLBACKS:
            raise ValueError(f"Callback {callback_name} not found in CALLBACKS.")
        callbacks.append(CALLBACKS[callback_name](model_config))

    if (
        script_args is not None
        and getattr(script_args, "method", "vanilla") != "vanilla"
    ):
        callbacks.append(MethodMetricsCallback(script_args))

    return callbacks
