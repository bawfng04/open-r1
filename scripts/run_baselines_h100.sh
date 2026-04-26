#!/usr/bin/env bash
set -euo pipefail

METHODS="vanilla,mgrpo,seed"
DATASETS="math500,gsm8k,aime2025,olympiadbench"
NUM_QUESTIONS=64
NUM_GENERATIONS=8
SEED=42
ACCELERATE_CONFIG="recipes/accelerate_configs/zero3_1gpu.yaml"
VENV_PATH=".venv"
LOG_DIR="logs/pipeline-h100"

SKIP_SETUP=false
SKIP_DATASET_PREPARE=false
SKIP_PREFLIGHT=false
SKIP_TRAIN=false
SKIP_BENCHMARK=false
CONTINUE_ON_ERROR=false
DISABLE_WANDB=false

usage() {
  cat <<'EOF'
Usage: ./scripts/run_baselines_h100.sh [options]

Options:
  --methods <csv>              Methods to run. Default: vanilla,mgrpo,seed
  --datasets <csv>             Bench datasets. Default: math500,gsm8k,aime2025,olympiadbench
  --num-questions <int>        Questions per dataset for benchmark. Default: 64
  --num-generations <int>      Generations per question. Default: 8
  --seed <int>                 Benchmark seed. Default: 42
  --accelerate-config <path>   Accelerate config file. Default: recipes/accelerate_configs/zero3.yaml
  --venv-path <path>           Virtual env path. Default: .venv
  --log-dir <path>             Log output directory. Default: logs/pipeline-h100
  --skip-setup                 Skip driver/env setup
  --skip-dataset-prepare       Skip dataset preparation
  --skip-preflight             Skip preflight checks
  --skip-train                 Skip training stage
  --skip-benchmark             Skip benchmark stage
  --continue-on-error          Continue remaining steps after a failure
  --disable-wandb             Set WANDB_MODE=disabled
  -h, --help                   Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --methods)
      METHODS="$2"
      shift 2
      ;;
    --datasets)
      DATASETS="$2"
      shift 2
      ;;
    --num-questions)
      NUM_QUESTIONS="$2"
      shift 2
      ;;
    --num-generations)
      NUM_GENERATIONS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --accelerate-config)
      ACCELERATE_CONFIG="$2"
      shift 2
      ;;
    --venv-path)
      VENV_PATH="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --skip-setup)
      SKIP_SETUP=true
      shift
      ;;
    --skip-dataset-prepare)
      SKIP_DATASET_PREPARE=true
      shift
      ;;
    --skip-preflight)
      SKIP_PREFLIGHT=true
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=true
      shift
      ;;
    --skip-benchmark)
      SKIP_BENCHMARK=true
      shift
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=true
      shift
      ;;
    --disable-wandb)
      DISABLE_WANDB=true
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

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  local level="$1"
  shift
  echo "[$(timestamp)][$level] $*"
}

mkdir -p "$LOG_DIR"
RUN_ID="$(date '+%Y%m%d-%H%M%S')"
LOG_FILE="$LOG_DIR/h100-baselines-${RUN_ID}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

IFS=',' read -r -a METHOD_ARRAY <<< "$METHODS"
for i in "${!METHOD_ARRAY[@]}"; do
  METHOD_ARRAY[$i]="$(echo "${METHOD_ARRAY[$i]}" | xargs | tr '[:upper:]' '[:lower:]')"
done

if [[ ${#METHOD_ARRAY[@]} -eq 0 ]]; then
  log "FAIL" "No methods selected"
  exit 1
fi

train_config_for_method() {
  local method="$1"
  case "$method" in
    vanilla) echo "recipes/Qwen2.5-Math-7B/grpo/config_h100_prod.yaml" ;;
    mgrpo) echo "recipes/Qwen2.5-Math-7B/grpo/config_mgrpo_h100_prod.yaml" ;;
    seed) echo "recipes/Qwen2.5-Math-7B/grpo/config_seed_h100_prod.yaml" ;;
    amsb) echo "recipes/Qwen2.5-Math-7B/grpo/config_amsb_h100_prod.yaml" ;;
    *)
      log "FAIL" "Unsupported method: $method"
      exit 1
      ;;
  esac
}

benchmark_output_for_method() {
  local method="$1"
  echo "data/benchmark-h100-${method}"
}

declare -a FAILED_STEPS=()
STEP_INDEX=0

run_step() {
  local name="$1"
  shift
  STEP_INDEX=$((STEP_INDEX + 1))
  local started=$SECONDS
  log "START" "Step ${STEP_INDEX}: ${name}"
  set +e
  "$@"
  local rc=$?
  set -e
  local elapsed=$((SECONDS - started))
  if [[ $rc -ne 0 ]]; then
    local msg="Step ${STEP_INDEX} failed after ${elapsed}s: ${name} (exit=${rc})"
    log "FAIL" "$msg"
    FAILED_STEPS+=("$msg")
    if [[ "$CONTINUE_ON_ERROR" != "true" ]]; then
      exit "$rc"
    fi
  else
    log "DONE" "Step ${STEP_INDEX} finished in ${elapsed}s: ${name}"
  fi
}

run_step_shell() {
  local name="$1"
  local cmd="$2"
  run_step "$name" bash -lc "$cmd"
}

log "INFO" "Repo root: $REPO_ROOT"
log "INFO" "Methods: ${METHOD_ARRAY[*]}"
log "INFO" "Log file: $LOG_FILE"

if [[ -f ".env" ]]; then
  run_step_shell "Load .env into current shell" "set -a; source .env; set +a"
fi

if [[ "$DISABLE_WANDB" == "true" ]]; then
  export WANDB_MODE=disabled
  log "INFO" "WANDB_MODE=disabled"
fi

if [[ "$SKIP_SETUP" != "true" ]]; then
  run_step "Driver/runtime precheck" ./scripts/bootstrap/install-drivers.sh
  run_step "Setup Python environment" ./scripts/bootstrap/setup-env.sh --venv-path "$VENV_PATH"
fi

if [[ -f "$VENV_PATH/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
  log "INFO" "Activated venv: $VENV_PATH"
fi

run_step_shell "Environment diagnostics" '
python --version
python -c "import sys,platform; print(\"python_exe=\"+sys.executable); print(\"platform=\"+platform.platform())"
python -c "import torch; print(\"torch=\"+torch.__version__); print(\"cuda_available=\"+str(torch.cuda.is_available())); print(\"cuda_device_count=\"+str(torch.cuda.device_count()))"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi || true; fi
'

if [[ "$SKIP_DATASET_PREPARE" != "true" ]]; then
  run_step "Prepare requested datasets" python scripts/prepare_requested_datasets.py
fi

for method in "${METHOD_ARRAY[@]}"; do
  train_cfg="$(train_config_for_method "$method")"

  if [[ "$SKIP_PREFLIGHT" != "true" ]]; then
    run_step "Preflight checks (${method})" \
      python scripts/validate_server_ready.py \
      --h100-config "$train_cfg" \
      --accelerate-config "$ACCELERATE_CONFIG" \
      --check-deps \
      --check-gpu \
      --strict-dataset \
      --print-json
  fi

  if [[ "$SKIP_TRAIN" != "true" ]]; then
    run_step "Train (${method})" ./scripts/deploy/train-grpo.sh --config "$train_cfg" --accelerate-config "$ACCELERATE_CONFIG"
  fi

  if [[ "$SKIP_BENCHMARK" != "true" ]]; then
    run_step "Benchmark (${method})" \
      python scripts/run_benchmark_dryrun.py \
      --datasets "$DATASETS" \
      --num-questions "$NUM_QUESTIONS" \
      --num-generations "$NUM_GENERATIONS" \
      --seed "$SEED" \
      --runtime-profile h100-prod \
      --method "$method" \
      --output-dir "$(benchmark_output_for_method "$method")"
  fi
done

export OPENR1_METHODS="${METHODS}"
export SUMMARY_OUT="${LOG_DIR}/benchmark_summary_baselines_${RUN_ID}.tsv"
run_step_shell "Benchmark summary table" '
python - <<"PY"
import json
import os
from pathlib import Path

methods = [m.strip().lower() for m in os.environ.get("OPENR1_METHODS", "").split(",") if m.strip()]
summary_file = os.environ.get("SUMMARY_OUT", "benchmark_summary.tsv")
rows = []
for method in methods:
    summary_path = Path(f"data/benchmark-h100-{method}/summary.json")
    if not summary_path.exists():
        print(f"[WARN] Missing summary: {summary_path}")
        continue
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    for dataset_name, payload in data.get("datasets", {}).items():
        metrics = payload.get("metrics", {})
        rows.append(
            {
                "method": method,
                "dataset": dataset_name,
                "pass@1": metrics.get("pass@1"),
                "pass@2": metrics.get("pass@2"),
                "pass@4": metrics.get("pass@4"),
                "pass@8": metrics.get("pass@8"),
            }
        )

if not rows:
    print("[WARN] No benchmark summaries found")
else:
    output_lines = ["method\tdataset\tpass@1\tpass@2\tpass@4\tpass@8"]
    for row in sorted(rows, key=lambda x: (x["method"], x["dataset"])):
        output_lines.append(
            "{method}\t{dataset}\t{p1}\t{p2}\t{p4}\t{p8}".format(
                method=row["method"],
                dataset=row["dataset"],
                p1=row.get("pass@1"),
                p2=row.get("pass@2"),
                p4=row.get("pass@4"),
                p8=row.get("pass@8"),
            )
        )
    table_str = "\n".join(output_lines)
    print(table_str)
    
    # Save to file
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(table_str + "\n")
    print(f"\n[INFO] Benchmark summary saved to: {summary_file}")
PY
'

if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
  log "WARN" "Run finished with failures:"
  for item in "${FAILED_STEPS[@]}"; do
    log "WARN" "$item"
  done
  exit 1
fi

log "INFO" "H100 baselines pipeline completed successfully"
log "INFO" "Detailed log: $LOG_FILE"
