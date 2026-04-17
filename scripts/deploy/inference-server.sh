#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="Qwen/Qwen2.5-Math-7B"
PORT="8000"
GPU_MEM_UTIL="0.9"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-id)
      MODEL_ID="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --gpu-mem-util)
      GPU_MEM_UTIL="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL"
