#!/usr/bin/env bash
set -euo pipefail

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[bootstrap] Detected NVIDIA runtime"
  nvidia-smi || true
else
  echo "[bootstrap] nvidia-smi not found. This is acceptable for local dry-run but not for H100 production."
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[bootstrap] python3 not found"
  exit 1
fi

python3 --version

echo "[bootstrap] Driver/runtime precheck completed"
