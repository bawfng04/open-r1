#!/usr/bin/env bash
set -euo pipefail

MODEL_ID=""
MODEL_REVISION="main"
BENCHMARKS="math_500,aime24"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-id)
      MODEL_ID="$2"
      shift 2
      ;;
    --model-revision)
      MODEL_REVISION="$2"
      shift 2
      ;;
    --benchmarks)
      BENCHMARKS="$2"
      shift 2
      ;;
    --model-config)
      MODEL_CONFIG="$2"
      shift 2
      if [[ -z "$MODEL_ID" ]]; then
        MODEL_ID=$(python - <<PY
import yaml
with open("$MODEL_CONFIG", "r", encoding="utf-8") as f:
    cfg=yaml.safe_load(f)
print(cfg.get("model_name_or_path", ""))
PY
)
      fi
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL_ID" ]]; then
  echo "Model id is required (--model-id or --model-config)"
  exit 1
fi

python scripts/run_benchmarks.py --model_id "$MODEL_ID" --model_revision "$MODEL_REVISION" --benchmarks "$BENCHMARKS"
