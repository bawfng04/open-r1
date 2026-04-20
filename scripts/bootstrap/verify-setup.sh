#!/usr/bin/env bash
set -euo pipefail

CONFIG="recipes/Qwen2.5-Math-7B/grpo/config_h100_prod.yaml"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

python scripts/validate_server_ready.py --h100-config "$CONFIG" --check-deps --strict-dataset --check-gpu
python - <<'PY'
import importlib
modules = ["torch", "transformers", "datasets", "trl"]
missing = []
for module_name in modules:
    if importlib.util.find_spec(module_name) is None:
        missing.append(module_name)
if missing:
    raise SystemExit(f"Missing required modules: {missing}")
print("[bootstrap] Python dependency import check passed")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  python - <<'PY'
import torch
print(f"[bootstrap] torch.cuda.is_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[bootstrap] CUDA device count={torch.cuda.device_count()}")
PY
fi

echo "[bootstrap] Setup verification completed"
