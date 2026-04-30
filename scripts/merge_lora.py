import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--lora_model', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    print(f'Loading base model: {args.base_model}')
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map='auto')
    
    print(f'Loading tokenizer...')
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.lora_model)
    except Exception as e:
        print(f'Could not load tokenizer from lora_model: {e}. Loading from base_model instead.')
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    print(f'Loading and merging LoRA: {args.lora_model}')
    model = PeftModel.from_pretrained(base, args.lora_model)
    model = model.merge_and_unload()
    
    print(f'Saving to {args.output_dir}')
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print('Done!')

if __name__ == '__main__':
    main()
