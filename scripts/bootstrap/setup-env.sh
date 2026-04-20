#!/usr/bin/env bash
set -euo pipefail

VENV_PATH=".venv"
INSTALL_EXTRAS="dev"
INSTALL_H100_EXTRAS="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-path)
      VENV_PATH="$2"
      shift 2
      ;;
    --extras)
      INSTALL_EXTRAS="$2"
      shift 2
      ;;
    --skip-h100-extras)
      INSTALL_H100_EXTRAS="false"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ ! -d "$VENV_PATH" ]]; then
  python3 -m venv "$VENV_PATH"
fi

# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e ".[$INSTALL_EXTRAS]"
python -m pip install accelerate trl transformers datasets pyyaml

if [[ "$INSTALL_H100_EXTRAS" == "true" ]]; then
  python -m pip install vllm==0.8.5.post1
  python -m pip install flash-attn --no-build-isolation
fi

echo "[bootstrap] Environment setup completed: $VENV_PATH"
