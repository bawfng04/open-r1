#!/usr/bin/env bash
set -euo pipefail

MODE="train"
TASK="grpo"
CONFIG="recipes/Qwen2.5-Math-7B/grpo/config_h100_prod.yaml"
ACCELERATE_CONFIG="recipes/accelerate_configs/zero3.yaml"
VENV_PATH=".venv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
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
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

./scripts/bootstrap/install-drivers.sh
./scripts/bootstrap/setup-env.sh --venv-path "$VENV_PATH"
./scripts/bootstrap/verify-setup.sh --config "$CONFIG"
./scripts/bootstrap/cache-models.sh --config "$CONFIG"

if [[ "$MODE" == "train" && "$TASK" == "grpo" ]]; then
  ./scripts/deploy/train-grpo.sh --config "$CONFIG" --accelerate-config "$ACCELERATE_CONFIG"
  exit 0
fi

if [[ "$MODE" == "eval" ]]; then
  ./scripts/deploy/eval.sh --model-config "$CONFIG"
  exit 0
fi

echo "Unsupported mode/task combination: mode=$MODE task=$TASK"
exit 1
