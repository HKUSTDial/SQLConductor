import argparse
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora(base_model_path: str, lora_path: str, output_path: str, device_map: str = "auto"):
    logger.info(f"Loading base model from {base_model_path} with device_map='{device_map}'")
    
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype, 
        device_map=device_map,
        trust_remote_code=True
    )
    
    logger.info(f"Loading LoRA adapters from {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    logger.info("Merging LoRA into base model...")
    model = model.merge_and_unload()
    
    logger.info(f"Loading tokenizer from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    logger.info(f"Saving merged model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    logger.info("Merge complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save merged model")
    parser.add_argument("--device", type=str, default="auto", help="Device map (auto, cpu, cuda:0, etc.)")
    
    args = parser.parse_args()
    
    merge_lora(args.base_model, args.lora_path, args.output_path, args.device)

