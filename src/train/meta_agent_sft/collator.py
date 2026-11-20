import torch
from transformers import PreTrainedTokenizer
from typing import List, Dict, Any
from loguru import logger

class DataCollatorForSupervisedFineTuning:
    """Format chat examples, mask prompt tokens, and pad each batch dynamically."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 4096, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        
        self.ignore_index = -100

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        sample_weights_list = []
        
        for example in examples:
            if example is None:
                continue
            if "messages" not in example:
                logger.warning("Example missing 'messages' key, skipping")
                continue
            messages = example["messages"]
            
            if not isinstance(messages, list) or len(messages) < 3:
                logger.warning(f"Invalid messages format: expected list with at least 3 items, got {type(messages)} with length {len(messages) if isinstance(messages, list) else 'N/A'}")
                continue
            
            weight = example.get("sample_weight", 1.0)
            
            
            # The prompt-only chat template must be a prefix of the full conversation.
            
            try:
                context_messages = messages[:-1]
                context_text = self.tokenizer.apply_chat_template(
                    context_messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                full_text = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                context_tokens = self.tokenizer(
                    context_text, 
                    add_special_tokens=False,
                    padding=False,
                    return_tensors='pt'
                ).input_ids[0]
                
                if len(context_tokens) > self.max_length:
                    continue
                
                full_tokens = self.tokenizer(
                    full_text,
                    add_special_tokens=False,
                    padding=False,
                    return_tensors='pt',
                    max_length=self.max_length,
                    truncation=True
                ).input_ids[0]
                
                labels = full_tokens.clone()
                
                context_len = len(context_tokens)
                if context_len < len(labels):
                    labels[:context_len] = self.ignore_index
                else:
                    labels[:] = self.ignore_index
                
                # Skip fully masked samples to avoid a NaN loss.
                if (labels != self.ignore_index).sum() == 0:
                    continue

                input_ids_list.append(full_tokens)
                labels_list.append(labels)
                attention_mask_list.append(torch.ones_like(full_tokens))
                sample_weights_list.append(weight)
                
            except Exception as e:
                logger.warning(f"Error collating example: {e}")
                continue

        if not input_ids_list:
            logger.warning(f"All {len(examples)} examples in batch were filtered out. Returning minimal batch.")
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            dummy_token = torch.tensor([[pad_token_id]], dtype=torch.long)
            return {
                "input_ids": dummy_token,
                "labels": torch.full_like(dummy_token, self.ignore_index),
                "attention_mask": torch.zeros_like(dummy_token),
                "sample_weights": torch.ones(1, dtype=torch.float)
            }

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=self.ignore_index
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask_list, batch_first=True, padding_value=0
        )
        
        if self.pad_to_multiple_of > 0:
            seq_len = input_ids.size(1)
            pad_len = (self.pad_to_multiple_of - seq_len % self.pad_to_multiple_of) % self.pad_to_multiple_of
            if pad_len > 0:
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                input_ids = torch.nn.functional.pad(
                    input_ids, (0, pad_len), value=pad_token_id
                )
                labels = torch.nn.functional.pad(
                    labels, (0, pad_len), value=self.ignore_index
                )
                attention_mask = torch.nn.functional.pad(
                    attention_mask, (0, pad_len), value=0
                )
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "sample_weights": torch.tensor(sample_weights_list, dtype=torch.float)
        }
