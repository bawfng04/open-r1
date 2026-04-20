#!/usr/bin/env bash
set -euo pipefail

CONFIG="recipes/Qwen2.5-Math-7B/grpo/config_h100_prod.yaml"
CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

python - "$CONFIG" "$CACHE_DIR" <<'PY'
import sys
from pathlib import Path

import yaml

cfg_path = Path(sys.argv[1])
cache_dir = Path(sys.argv[2])

with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

model_id = cfg.get("model_name_or_path")
if not model_id:
    raise SystemExit("model_name_or_path missing in config")

try:
    from huggingface_hub import snapshot_download
except Exception:
    print("[bootstrap] huggingface_hub not available; skipping model prefetch")
    raise SystemExit(0)

snapshot_download(repo_id=model_id, cache_dir=str(cache_dir), local_files_only=False)
print(f"[bootstrap] Cached model {model_id} in {cache_dir}")
PY

echo "[bootstrap] Model cache step completed"
