#!/usr/bin/env bash
set -euo pipefail

# benchmark sb-grpo (english checkpoint). merge lora, patch lighteval, run evals
#
# 20GB
# source /network-volume/envs/msb_grpo/bin/activate
# cd /network-volume/M-SB-GRPO-Benchmark/compare/open-r1
#
# 40GB
# # source /network-volume/envs/msb_grpo/bin/activate
# cd /network-volume/Mine-GRPO/compare/open-r1
#
# ./scripts/benchmark_sb_grpo_all.sh > ENGLISH_BENCHMARK.log 2>&1 &
# tail -f ENGLISH_BENCHMARK.log
#
#
# kill:
# pkill -9 -f 'python|lighteval|vllm'

export VLLM_MAX_NUM_SEQS=16
# export VLLM_ENFORCE_EAGER=1

# base config
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
CHECKPOINT_PATH="/network-volume/Mine-GRPO/compare/A-MSB-GRPO-E/checkpoints/latest"
# CHECKPOINT_PATH="/network-volume/M-SB-GRPO-Benchmark/compare/A-MSB-GRPO-E/checkpoints/epoch1_step13699"
MERGED_MODEL_DIR="models/SB-GRPO-English"
RESULTS_DIR="results/SB-GRPO-English"

# conda paths
PY_BIN="/network-volume/envs/msb_grpo/bin/python"
PIP_BIN="/network-volume/envs/msb_grpo/bin/pip"
LIGHTEVAL_BIN="/network-volume/envs/msb_grpo/bin/lighteval"

# fallback to default bins
if [[ ! -f "$PY_BIN" ]]; then
    PY_BIN="python"
    PIP_BIN="pip"
    LIGHTEVAL_BIN="lighteval"
fi

# ds list
DATASETS=("gsm8k" "math_full" "aime24" "aime25" "gpqa")

# vllm config
GPU_MEM="0.85"
MAX_LEN="32768"
SYSTEM_PROMPT="" # Set to empty to match Layer 1 training setup (no system prompt)
MAX_SAMPLES="10" # Set to a number (e.g. 20) for quick testing, or empty "" for full run

echo "=============================================================================="
echo ">>> Initiating SB-GRPO Benchmark Pipeline on the Server"
echo "=============================================================================="

# init dirs
mkdir -p "scripts"
mkdir -p "$RESULTS_DIR"

# auto install deps
echo ">>> [1/6] Bypassing dependency installations (already satisfied)..."
# force torch cu121
"$PIP_BIN" install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
"$PIP_BIN" install vllm==0.6.2 transformers==4.45.2 peft accelerate ray latex2sympy2_extended more-itertools --extra-index-url https://download.pytorch.org/whl/cu124

# lighteval math extension
echo ">>> Installing the standard Lighteval framework from HuggingFace..."
"$PIP_BIN" install -U "lighteval[math] @ git+https://github.com/huggingface/lighteval.git@d3da6b9bbf38104c8b5e1acc86f83541f9a502d1"
# downgrade datasets for gpqa trust_remote_code
"$PIP_BIN" install "datasets<3.0.0"

# gen merge_lora.py
echo ">>> [2/6] Automatically recreating the missing script at scripts/merge_lora.py..."
cat << 'EOF' > scripts/merge_lora.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--lora_model', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    print(f"Loading the base model configuration from: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map='cpu')

    print("Initializing the tokenizer framework...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.lora_model)
    except Exception as e:
        print(f"Failed to load the tokenizer from the LoRA checkpoint: {e}. Reverting to the base model's tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # keep base chat_template
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.chat_template is None and base_tokenizer.chat_template is not None:
        tokenizer.chat_template = base_tokenizer.chat_template

    print(f"Loading and successfully merging the LoRA adapter from: {args.lora_model}")
    model = PeftModel.from_pretrained(base, args.lora_model)
    model = model.merge_and_unload()

    print(f"Persisting the merged model state to: {args.output_dir}")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print("The model merging process has been successfully completed.")

if __name__ == '__main__':
    main()
EOF

# merge lora
FORCE_REMERGE=true # Set to true to ensure latest weights are used

if [[ -d "$CHECKPOINT_PATH" ]]; then
    SHOULD_MERGE=false
    if [[ ! -d "$MERGED_MODEL_DIR" || -z "$(ls -A "$MERGED_MODEL_DIR" 2>/dev/null)" ]]; then
        SHOULD_MERGE=true
    elif [[ "$FORCE_REMERGE" == "true" ]]; then
        echo ">>> [SYSTEM] Force re-merge is enabled. Deleting old merged model..."
        rm -rf "$MERGED_MODEL_DIR"
        SHOULD_MERGE=true
    fi

    if [[ "$SHOULD_MERGE" == "true" ]]; then
        echo ">>> [3/6] Commencing the merging process for $CHECKPOINT_PATH into $MERGED_MODEL_DIR..."
        "$PY_BIN" scripts/merge_lora.py \
            --base_model "$BASE_MODEL" \
            --lora_model "$CHECKPOINT_PATH" \
            --output_dir "$MERGED_MODEL_DIR"
    else
        echo ">>> [3/6] The merged model already exists at $MERGED_MODEL_DIR. Skipping the merge step."
    fi
    TARGET_MODEL="$MERGED_MODEL_DIR"
else
    echo ">>> [CRITICAL ERROR] Checkpoint directory not found at: $CHECKPOINT_PATH"
    echo ">>> [ACTION REQUIRED] Please verify the path or ensure the training process has completed. Terminating..."
    exit 1
fi

# patch stop seq + generation_size via heredoc (avoids shell escape mangling)
echo ">>> [4/6] Automatically patching the Lighteval stop_sequence attribute via $PY_BIN..."
"$PY_BIN" - <<'PYEOF'
import os
import re
import lighteval

lighteval_path = os.path.dirname(lighteval.__file__)

# pass 1: nuke stop_sequence containing \n
stop_patched = 0
for root, _, files in os.walk(lighteval_path):
    for f in files:
        if not (f.endswith(".py") or f.endswith(".jsonl")):
            continue
        path = os.path.join(root, f)
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
        original = content
        replacements = [
            ('stop_sequence=["\\\\n"]', "stop_sequence=[]"),
            ("stop_sequence=['\\\\n']", "stop_sequence=[]"),
            ('stop_sequence=["\\n"]', "stop_sequence=[]"),
            ("stop_sequence=['\\n']", "stop_sequence=[]"),
            ('"stop_sequence": ["\\\\n"]', '"stop_sequence": []'),
            ('"stop_sequence": ["\\n"]', '"stop_sequence": []'),
        ]
        for old, new in replacements:
            content = content.replace(old, new)
        if content != original:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            stop_patched += 1

# pass 2: force generation_size to 16384
gen_patched = 0
for root, _, files in os.walk(lighteval_path):
    for f in files:
        if not f.endswith((".py", ".jsonl", ".json")):
            continue
        path = os.path.join(root, f)
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
        original = content
        content = re.sub(r'generation_size\s*=\s*\d+', 'generation_size=16384', content)
        content = re.sub(r'"generation_size"\s*:\s*\d+', '"generation_size": 16384', content)
        if content != original:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            gen_patched += 1

print(f"Successfully completed the Lighteval patch verification. A total of {stop_patched} stop_sequence files and {gen_patched} generation_size files were updated.")
PYEOF

# check lighteval flags
echo ">>> [5/6] Validating the supported parameters and flag configurations for the Lighteval framework..."
LIGHTEVAL_HELP="$("$LIGHTEVAL_BIN" vllm --help 2>&1 || true)"
CHAT_FLAG=""
PROMPT_FLAG=""

# check chat template
if grep -q -- "--use-chat-template" <<< "$LIGHTEVAL_HELP"; then
    CHAT_FLAG="--use-chat-template"
elif grep -q -- "--use_chat_template" <<< "$LIGHTEVAL_HELP"; then
    CHAT_FLAG="--use_chat_template"
fi

# check sys prompt
if grep -q -- "--system-prompt" <<< "$LIGHTEVAL_HELP"; then
    PROMPT_FLAG="--system-prompt"
elif grep -q -- "--system_prompt" <<< "$LIGHTEVAL_HELP"; then
    PROMPT_FLAG="--system_prompt"
fi

# map ds to task
get_task_string() {
    local ds="$1"
    case "$ds" in
        gsm8k)
            echo "lighteval|gsm8k|0|0"
            ;;
        math_full)
            echo "lighteval|math:algebra|0|0,lighteval|math:counting_and_probability|0|0,lighteval|math:geometry|0|0,lighteval|math:intermediate_algebra|0|0,lighteval|math:number_theory|0|0,lighteval|math:prealgebra|0|0,lighteval|math:precalculus|0|0"
            ;;
        math500)
            echo "lighteval|math_500|0|0"
            ;;
        aime24)
            echo "lighteval|aime24|0|0"
            ;;
        aime25)
            echo "lighteval|aime25|0|0"
            ;;
        gpqa)
            echo "lighteval|gpqa:diamond|0|0"
            ;;
        *)
            echo "$ds"
            ;;
    esac
}

# loop & run evals (each ds in isolated subshell for clean RAM/VRAM release)
echo ">>> [6/6] Commencing the sequential benchmark execution across all defined datasets..."
export VLLM_USE_V1=0

for DS in "${DATASETS[@]}"; do
    TASK_STR="$(get_task_string "$DS")"
    OUT_DIR="$RESULTS_DIR/SB_GRPO_${DS}"

    echo "------------------------------------------------------------------------------"
    echo ">>> Executing the benchmark suite on the dataset: $DS"
    echo ">>> Assigned task sequence: $TASK_STR"
    echo "------------------------------------------------------------------------------"

    # pre-flight cleanup
    echo ">>> [SYSTEM] Cleaning up VRAM and RAM before launching $DS..."
    pkill -9 -f 'vllm' 2>/dev/null || true
    pkill -9 -f 'lighteval' 2>/dev/null || true
    find ~/.cache/huggingface/ -name "*.lock" -delete 2>/dev/null || true
    sleep 8

    # run in isolated subshell so all mem is freed on exit
    (
        CMD=("$LIGHTEVAL_BIN" vllm "model_name=$TARGET_MODEL,dtype=bfloat16,gpu_memory_utilization=$GPU_MEM,max_model_length=$MAX_LEN,max_num_batched_tokens=$MAX_LEN,max_num_seqs=32" "$TASK_STR")

        if [[ -n "$CHAT_FLAG" ]]; then
            CMD+=("$CHAT_FLAG")
        fi

        if [[ -n "$PROMPT_FLAG" ]]; then
            CMD+=("$PROMPT_FLAG" "$SYSTEM_PROMPT")
        fi

        if [[ -n "$MAX_SAMPLES" ]]; then
            CMD+=("--max-samples" "$MAX_SAMPLES")
        fi
        CMD+=("--save-details")

        CMD+=(--output-dir "$OUT_DIR")
        "${CMD[@]}"
    )
    EXIT_CODE=$?

    # check error
    if [[ $EXIT_CODE -ne 0 ]]; then
        echo ">>> [ERROR] The evaluation process for dataset $DS encountered a critical failure with exit code $EXIT_CODE. Proceeding to the next dataset..."
    else
        echo ">>> [SUCCESS] The evaluation for dataset $DS has successfully concluded. Artifacts and results are safely stored at: $OUT_DIR"
    fi
done

echo "=============================================================================="
echo ">>> All designated benchmark workflows have been successfully completed."
echo "=============================================================================="
