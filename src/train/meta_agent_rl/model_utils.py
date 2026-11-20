
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Dict, Any
from loguru import logger

from src.system.llm_client import LLMClient, LLMResponse
from src.system.config import ModelConfig

class LocalLLMClient(LLMClient):
    """Local Hugging Face client for policy generation."""
    
    def __init__(self, model_config: ModelConfig, model, tokenizer):
        super().__init__(model_config)
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text synchronously through the async client interface."""
        full_prompt = prompt
        if system_prompt:
            
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            try:
                full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                full_prompt = f"{system_prompt}\n\n{prompt}"

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature if temperature > 0 else 1.0,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature <= 1e-5:
            gen_kwargs["do_sample"] = False
            
        with torch.no_grad():

            # Independent rollouts must bypass distributed collective operations.
            unwrapped = self.model
            while hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module
                
            outputs = unwrapped.generate(**inputs, **gen_kwargs)
            
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        content = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        if stop:
            for s in stop:
                if s in content:
                    content = content.split(s)[0]
        
        return LLMResponse(
            content=content,
            model=self.model_config.model_name,
            usage={
                "prompt_tokens": inputs.input_ids.shape[1],
                "completion_tokens": generated_ids.shape[0],
                "total_tokens": outputs.shape[1]
            },
            finish_reason="stop"
        )

def load_model_and_tokenizer(
    model_name_or_path: str, 
    device: Optional[str] = None, 
    allow_zero3_init: bool = True,
    quantization_config: Optional[Dict[str, Any]] = None
):
    """
    Load model and tokenizer for training.
    
    Args:
        model_name_or_path: Path to model or HuggingFace model ID
        device: Target device (None for Accelerate/DeepSpeed auto placement)
        allow_zero3_init: Allow ZeRO-3 initialization hooks
        quantization_config: Dict with quantization settings:
            - enabled: bool - Enable 8-bit quantization
            - load_in_8bit: bool - Use LLM.int8() quantization
    """
    logger.info(f"Loading model from {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = None
    use_quantization = quantization_config and quantization_config.get('enabled', False)
    
    if use_quantization:
        try:
            from transformers import BitsAndBytesConfig
            logger.info("Loading the base model with 8-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=quantization_config.get('load_in_8bit', True),
                # LLM.int8() threshold for outlier detection
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=quantization_config.get('llm_int8_skip_modules', None)
            )
        except ImportError:
            logger.warning("bitsandbytes is unavailable; loading without quantization.")
            bnb_config = None
    else:
        logger.info("Loading model in bfloat16")
    
    from accelerate import Accelerator
    accelerator = Accelerator()
    
    load_context = None
    try:
        import deepspeed
        if not allow_zero3_init:
            load_context = deepspeed.zero.Init(enabled=False)
    except ImportError:
        pass
    
    attn_implementation = "eager"
    try:
        import flash_attn
        attn_implementation = "flash_attention_2"
    except ImportError:
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
             attn_implementation = "sdpa"
        else:
             attn_implementation = "eager"
            
    logger.info(f"Using attention implementation: {attn_implementation}")

    load_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": attn_implementation,
    }
    
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
        
    if load_context:
        with load_context:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, 
                **load_kwargs
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            **load_kwargs
        )
    
    if use_quantization and bnb_config is not None:
        try:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
            logger.info("Model prepared for k-bit training")
        except ImportError:
            logger.warning("peft.prepare_model_for_kbit_training is unavailable.")
    
    if device:
        model = model.to(device)
    
    return model, tokenizer
