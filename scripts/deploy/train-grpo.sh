#!/usr/bin/env bash
set -euo pipefail

CONFIG="recipes/Qwen2.5-Math-7B/grpo/config_h100_prod.yaml"
ACCELERATE_CONFIG="recipes/accelerate_configs/zero3.yaml"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --accelerate-config)
      ACCELERATE_CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

python scripts/validate_server_ready.py \
  --h100-config "$CONFIG" \
  --accelerate-config "$ACCELERATE_CONFIG" \
  --check-deps \
  --strict-dataset \
  --check-gpu
export PYTHONPATH="src:${PYTHONPATH:-}"

if command -v accelerate >/dev/null 2>&1; then
  accelerate launch --config_file "$ACCELERATE_CONFIG" src/open_r1/grpo.py --config "$CONFIG"
else
  python -m accelerate.commands.launch --config_file "$ACCELERATE_CONFIG" src/open_r1/grpo.py --config "$CONFIG"
fi
