#!/usr/bin/env bash
set -euo pipefail

# Run additional benchmark suites for already-trained model variants.
# Intended for Linux/H100 server usage.

MODELS="GRPO,MGRPO,AMSB"
BENCHMARKS="math500,aime24,aime25,gpqa"
MODEL_PREFIX="models/Qwen2.5-7B"
RESULTS_DIR="results/extra-benchmarks"
GPU_MEMORY_UTILIZATION="0.72"
MAX_MODEL_LENGTH="3072"
MAX_NUM_BATCHED_TOKENS="2048"
SYSTEM_PROMPT_MATH="Please reason step by step, and put your final answer within \\boxed{}."
SYSTEM_PROMPT_OTHER=""

SKIP_PATCH=false
SKIP_PREFETCH=false
PREFETCH_ONLY=false
CONTINUE_ON_ERROR=false
FORCE_BOXED_PROMPT=false

usage() {
  cat <<'EOF'
Usage: ./scripts/run_extra_benchmarks_h100.sh [options]

Options:
  --models <csv>                 Model suffix list. Default: GRPO,MGRPO,AMSB
                                 Expected merged path: <model-prefix>-<MODEL>-Final
  --benchmarks <csv>             Benchmarks to run. Default: math500,aime24,aime25,gpqa
                                 Supported: gsm8k,math_full,math500,aime24,aime25,gpqa,lcb,lcb_v4
  --model-prefix <path>          Base model path prefix. Default: models/Qwen2.5-7B
  --results-dir <path>           Output root directory. Default: results/extra-benchmarks
  --gpu-mem <float>              vLLM gpu_memory_utilization. Default: 0.72 (H100 40GB safe)
  --max-model-length <int>       vLLM max_model_length. Default: 3072
  --max-batched-tokens <int>     vLLM max_num_batched_tokens. Default: 2048
  --system-prompt-math <text>    Prompt for math-style sets (gsm8k/math/aime)
  --system-prompt-other <text>   Prompt for non-math sets (default empty)
  --force-boxed-prompt           Use math prompt for all benchmarks
  --skip-patch                   Skip MATH stop-sequence patching
  --skip-prefetch                Skip HF dataset prefetch
  --prefetch-only                Download/warm datasets only, skip benchmark runs
  --continue-on-error            Continue with remaining jobs on failure
  -h, --help                     Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      MODELS="$2"
      shift 2
      ;;
    --benchmarks)
      BENCHMARKS="$2"
      shift 2
      ;;
    --model-prefix)
      MODEL_PREFIX="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --gpu-mem)
      GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --max-model-length)
      MAX_MODEL_LENGTH="$2"
      shift 2
      ;;
    --max-batched-tokens)
      MAX_NUM_BATCHED_TOKENS="$2"
      shift 2
      ;;
    --system-prompt-math)
      SYSTEM_PROMPT_MATH="$2"
      shift 2
      ;;
    --system-prompt-other)
      SYSTEM_PROMPT_OTHER="$2"
      shift 2
      ;;
    --force-boxed-prompt)
      FORCE_BOXED_PROMPT=true
      shift
      ;;
    --skip-patch)
      SKIP_PATCH=true
      shift
      ;;
    --skip-prefetch)
      SKIP_PREFETCH=true
      shift
      ;;
    --prefetch-only)
      PREFETCH_ONLY=true
      shift
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() {
  local level="$1"
  shift
  echo "[$(timestamp)][$level] $*"
}

run_step() {
  local name="$1"
  shift
  log "START" "$name"
  set +e
  "$@"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    log "FAIL" "$name (exit=$rc)"
    if [[ "$CONTINUE_ON_ERROR" != "true" ]]; then
      exit "$rc"
    fi
  else
    log "DONE" "$name"
  fi
}

task_list_candidates_for_benchmark() {
  local bench="$1"
  case "$bench" in
    gsm8k)
      cat <<'EOF'
lighteval|gsm8k|0|0
EOF
      ;;
    math_full)
      cat <<'EOF'
lighteval|math:algebra|0|0,lighteval|math:counting_and_probability|0|0,lighteval|math:geometry|0|0,lighteval|math:intermediate_algebra|0|0,lighteval|math:number_theory|0|0,lighteval|math:prealgebra|0|0,lighteval|math:precalculus|0|0
EOF
      ;;
    math500)
      cat <<'EOF'
lighteval|math_500|0|0
lighteval|math500|0|0
lighteval|math|0|0
lighteval|math:algebra|0|0,lighteval|math:counting_and_probability|0|0,lighteval|math:geometry|0|0,lighteval|math:intermediate_algebra|0|0,lighteval|math:number_theory|0|0,lighteval|math:prealgebra|0|0,lighteval|math:precalculus|0|0
EOF
      ;;
    aime24)
      cat <<'EOF'
lighteval|aime24|0|0
lighteval|aime_2024|0|0
lighteval|aime|0|0
EOF
      ;;
    aime25)
      cat <<'EOF'
lighteval|aime25|0|0
lighteval|aime_2025|0|0
lighteval|aime|0|0
EOF
      ;;
    gpqa)
      cat <<'EOF'
lighteval|gpqa:diamond|0|0
lighteval|gpqa|0|0
EOF
      ;;
    lcb)
      cat <<'EOF'
extended|lcb:codegeneration|0|0
EOF
      ;;
    lcb_v4)
      cat <<'EOF'
extended|lcb:codegeneration_v4|0|0
EOF
      ;;
    *)
      true
      ;;
  esac
}

patch_lighteval_math_stop_sequence() {
  python - <<'PY'
import os
import lighteval

lighteval_path = os.path.dirname(lighteval.__file__)
patched = 0
for root, _, files in os.walk(lighteval_path):
    for filename in files:
        if not (filename.endswith(".py") or filename.endswith(".jsonl")):
            continue
        path = os.path.join(root, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
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
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            patched += 1
print(f"[patch] updated_files={patched}")
PY
}

prefetch_datasets() {
  local benches_csv="$1"
  export BENCHMARKS_TO_PREFETCH="$benches_csv"
  python - <<'PY'
import os
import datasets

benchmarks = [x.strip().lower() for x in os.environ.get("BENCHMARKS_TO_PREFETCH", "").split(",") if x.strip()]
mapping = {
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "math500": ("HuggingFaceH4/MATH-500", None, "test"),
    "aime25": ("TianHongZXY/AIME2025", None, "test"),
}

for bench in benchmarks:
    spec = mapping.get(bench)
    if not spec:
        print(f"[prefetch][skip] no direct HF mapping for benchmark='{bench}', lighteval will fetch as needed")
        continue
    hf_id, cfg, split = spec
    try:
        ds = datasets.load_dataset(hf_id, cfg, split=split)
        _ = ds.select(range(min(1, len(ds))))
        print(f"[prefetch][ok] benchmark='{bench}' hf='{hf_id}' config='{cfg}' split='{split}' rows={len(ds)}")
    except Exception as exc:
        print(f"[prefetch][warn] benchmark='{bench}' hf='{hf_id}' failed: {exc}")
PY
}

mkdir -p "$RESULTS_DIR"

IFS=',' read -r -a MODEL_ARRAY <<< "$MODELS"
IFS=',' read -r -a BENCH_ARRAY <<< "$BENCHMARKS"
for i in "${!BENCH_ARRAY[@]}"; do
  BENCH_ARRAY[$i]="$(echo "${BENCH_ARRAY[$i]}" | xargs | tr '[:upper:]' '[:lower:]')"
done

log "INFO" "Repo root: $REPO_ROOT"
log "INFO" "Models: ${MODEL_ARRAY[*]}"
log "INFO" "Benchmarks: ${BENCH_ARRAY[*]}"
log "INFO" "Results dir: $RESULTS_DIR"

LIGHTEVAL_HELP="$(lighteval vllm --help 2>&1 || true)"
CHAT_TEMPLATE_FLAG=""
SYSTEM_PROMPT_FLAG=""
if grep -q -- "--use-chat-template" <<<"$LIGHTEVAL_HELP"; then
  CHAT_TEMPLATE_FLAG="--use-chat-template"
elif grep -q -- "--use_chat_template" <<<"$LIGHTEVAL_HELP"; then
  CHAT_TEMPLATE_FLAG="--use_chat_template"
fi
if grep -q -- "--system-prompt" <<<"$LIGHTEVAL_HELP"; then
  SYSTEM_PROMPT_FLAG="--system-prompt"
elif grep -q -- "--system_prompt" <<<"$LIGHTEVAL_HELP"; then
  SYSTEM_PROMPT_FLAG="--system_prompt"
fi
if [[ -n "$CHAT_TEMPLATE_FLAG" ]]; then
  log "INFO" "Detected chat-template flag: $CHAT_TEMPLATE_FLAG"
else
  log "WARN" "No chat-template flag found in this lighteval version; running without explicit flag."
fi
if [[ -n "$SYSTEM_PROMPT_FLAG" ]]; then
  log "INFO" "Detected system-prompt flag: $SYSTEM_PROMPT_FLAG"
else
  log "WARN" "No system-prompt flag found in this lighteval version; custom prompts will be ignored."
fi

is_math_like_benchmark() {
  local bench="$1"
  case "$bench" in
    gsm8k|math_full|math500|aime24|aime25)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

resolve_task_for_benchmark() {
  local bench="$1"
  python - "$bench" 2>/dev/null <<'PY'
import sys

bench = sys.argv[1].strip().lower()

try:
    from lighteval.tasks.registry import Registry
except Exception:
    print("")
    raise SystemExit(0)

try:
    all_configs = Registry.load_all_task_configs(custom_tasks=None, load_multilingual=False)
except Exception:
    print("")
    raise SystemExit(0)

keys = list(all_configs.keys())
lower_keys = [k.lower() for k in keys]

search_terms = {
    "math500": ["math_500", "math500", "math-500", "math"],
    "aime24": ["aime24", "aime_24", "aime2024", "aime_2024", "aime"],
    "aime25": ["aime25", "aime_25", "aime2025", "aime_2025", "aime"],
    "gpqa": ["gpqa", "diamond"],
    "gsm8k": ["gsm8k", "gsm"],
    "math_full": ["math:"],
    "lcb": ["lcb", "codegeneration"],
    "lcb_v4": ["lcb", "v4", "codegeneration"],
}

terms = search_terms.get(bench, [bench])

def variants_for_key(key: str):
    out = []
    if "|" in key:
        out.append(key)
        if key.count("|") == 1:
            out.append(f"{key}|0|0")
    else:
        out.append(f"lighteval|{key}|0|0")
        out.append(f"{key}|0|0")
        out.append(key)
    # Keep unique order
    seen = set()
    dedup = []
    for item in out:
        if item not in seen:
            seen.add(item)
            dedup.append(item)
    return dedup

def works(task_string: str) -> bool:
    try:
        Registry(tasks=task_string, load_multilingual=False, custom_tasks=None)
        return True
    except Exception:
        return False

# 1) Exact-ish candidates first.
candidates = []
for term in terms:
    for idx, lk in enumerate(lower_keys):
        if lk == term or lk.endswith(f"|{term}") or lk.endswith(f":{term}"):
            candidates.append(keys[idx])

# 2) Contains candidates next.
for term in terms:
    for idx, lk in enumerate(lower_keys):
        if term in lk:
            candidates.append(keys[idx])

# 3) Unique + bounded list.
seen = set()
ordered = []
for key in candidates:
    if key in seen:
        continue
    seen.add(key)
    ordered.append(key)
ordered = ordered[:50]

for key in ordered:
    for task_string in variants_for_key(key):
        if works(task_string):
            print(task_string)
            raise SystemExit(0)

print("")
PY
}

if [[ "$SKIP_PREFETCH" != "true" ]]; then
  run_step "Prefetch benchmark datasets" prefetch_datasets "$BENCHMARKS"
fi

if [[ "$PREFETCH_ONLY" == "true" ]]; then
  log "INFO" "Prefetch-only mode completed."
  exit 0
fi

if [[ "$SKIP_PATCH" != "true" ]]; then
  # Needed for math_* tasks to avoid early stop at newline in some lighteval versions.
  run_step "Patch lighteval stop_sequence for MATH tasks" patch_lighteval_math_stop_sequence
fi

declare -A RESOLVED_TASKS
unresolved_count=0
for bench in "${BENCH_ARRAY[@]}"; do
  resolved="$(resolve_task_for_benchmark "$bench")"
  RESOLVED_TASKS["$bench"]="$resolved"
  if [[ -n "$resolved" ]]; then
    log "INFO" "Resolved benchmark '$bench' -> '$resolved'"
  else
    log "WARN" "Could not resolve a compatible task for benchmark '$bench' with current lighteval install."
    unresolved_count=$((unresolved_count + 1))
  fi
done

if [[ $unresolved_count -gt 0 ]]; then
  log "WARN" "Some benchmarks are unresolved. Consider upgrading lighteval to the repo-pinned version:"
  log "WARN" "python -m pip install -U \"lighteval @ git+https://github.com/huggingface/lighteval.git@d3da6b9bbf38104c8b5e1acc86f83541f9a502d1\""
fi

if [[ $unresolved_count -eq ${#BENCH_ARRAY[@]} ]]; then
  log "FAIL" "No requested benchmarks could be resolved in current lighteval environment."
  exit 1
fi

for raw_model in "${MODEL_ARRAY[@]}"; do
  model_name="$(echo "$raw_model" | xargs)"
  merged_path="${MODEL_PREFIX}-${model_name}-Final"
  if [[ ! -d "$merged_path" ]]; then
    log "WARN" "Model path not found, skip: $merged_path"
    continue
  fi

  for bench in "${BENCH_ARRAY[@]}"; do
    task_list="${RESOLVED_TASKS[$bench]}"
    if [[ -z "$task_list" ]]; then
      log "WARN" "Skip ${model_name}/${bench}: unresolved benchmark task."
      continue
    fi

    out_dir="${RESULTS_DIR}/${model_name}_${bench}"
    selected_prompt=""
    if [[ "$FORCE_BOXED_PROMPT" == "true" ]] || is_math_like_benchmark "$bench"; then
      selected_prompt="$SYSTEM_PROMPT_MATH"
    elif [[ -n "$SYSTEM_PROMPT_OTHER" ]]; then
      selected_prompt="$SYSTEM_PROMPT_OTHER"
    fi

    model_args="model_name=${merged_path},dtype=bfloat16,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_length=${MAX_MODEL_LENGTH},max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
    cmd=(lighteval vllm "$model_args" "$task_list")
    if [[ -n "$CHAT_TEMPLATE_FLAG" ]]; then
      cmd+=("$CHAT_TEMPLATE_FLAG")
    fi
    if [[ -n "$selected_prompt" && -n "$SYSTEM_PROMPT_FLAG" ]]; then
      cmd+=("$SYSTEM_PROMPT_FLAG" "$selected_prompt")
    fi
    cmd+=(--output-dir "$out_dir")

    run_step "Benchmark ${model_name} on ${bench}" "${cmd[@]}"
  done
done

log "INFO" "Extra benchmark run completed."
