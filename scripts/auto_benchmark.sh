#!/bin/bash
set -e
source /network-volume/miniconda3/etc/profile.d/conda.sh && conda activate /network-volume/envs/msb_grpo
export VLLM_USE_V1=0
# MODELS = "GRPO" cho GRPO gốc
# hoặc "MGRPO" cho Multilayer-GRPO
# hoặc "AMSB" cho Attention-based Multilayer-GRPO
MODELS=("AMSB")
mkdir -p results

# Patch lighteval stop_sequence cho tập MATH để không bị ngắt generation giữa chừng khi model xuống dòng
echo ">>> Đang patch lighteval để bỏ stop sequence \n..."
python -c "
import os
import lighteval

lighteval_path = os.path.dirname(lighteval.__file__)
patched = 0

for root, _, files in os.walk(lighteval_path):
    for f in files:
        filepath = os.path.join(root, f)
        if f.endswith('.py') or f.endswith('.jsonl'):
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            orig_content = content
            
            replacements = [
                ('stop_sequence=[\"\\\\n\"]', 'stop_sequence=[]'),
                (\"stop_sequence=['\\\\n']\", \"stop_sequence=[]\"),
                ('stop_sequence=[\"\\n\"]', 'stop_sequence=[]'),
                (\"stop_sequence=['\\n']\", \"stop_sequence=[]\"),
                ('\"stop_sequence\": [\"\\\\n\"]', '\"stop_sequence\": []'),
                ('\"stop_sequence\": [\"\\n\"]', '\"stop_sequence\": []')
            ]
            for old, new in replacements:
                content = content.replace(old, new)
            
            if content != orig_content:
                with open(filepath, 'w', encoding='utf-8') as file:
                    file.write(content)
                print(f'Patched {filepath}')
                patched += 1

if patched == 0:
    print('Đã kiểm tra, không có file nào cần patch (có thể đã patch từ trước).')
"

# Format mới: suite|task|few_shot|truncate_few_shots
MATH_TASKS="lighteval|math:algebra|0|0,lighteval|math:counting_and_probability|0|0,lighteval|math:geometry|0|0,lighteval|math:intermediate_algebra|0|0,lighteval|math:number_theory|0|0,lighteval|math:prealgebra|0|0,lighteval|math:precalculus|0|0"

for M_NAME in "${MODELS[@]}"; do
    MERGED_PATH="models/Qwen2.5-7B-${M_NAME}-Final"
    
    echo ">>> Đang Benchmark $M_NAME trên tập GSM8K..."
    lighteval vllm \
        "model_name=$MERGED_PATH,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_length=4096,max_num_batched_tokens=4096" \
        "lighteval|gsm8k|0|0" \
        --use-chat-template \
        --system-prompt="Please reason step by step, and put your final answer within \boxed{}." \
        --output-dir "results/${M_NAME}_gsm8k"
        
    echo ">>> Đang Benchmark $M_NAME trên tập MATH (FULL)..."
    lighteval vllm \
        "model_name=$MERGED_PATH,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_length=4096,max_num_batched_tokens=4096" \
        "$MATH_TASKS" \
        --use-chat-template \
        --system-prompt="Please reason step by step, and put your final answer within \boxed{}." \
        --output-dir "results/${M_NAME}_math"
done
echo ">>> TẤT CẢ ĐÃ XONG!"