#!/usr/bin/env bash
set -euo pipefail

VENV_PATH=".venv"
INSTALL_EXTRAS="dev"

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

echo "[bootstrap] Environment setup completed: $VENV_PATH"
