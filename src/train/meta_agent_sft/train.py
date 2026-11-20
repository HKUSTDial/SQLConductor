import argparse
import os
import re
import yaml
from pathlib import Path
from loguru import logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType

from src.train.meta_agent_sft.dataset import MetaAgentSFTDataset
from src.train.meta_agent_sft.collator import DataCollatorForSupervisedFineTuning
from src.train.meta_agent_sft.sampler import BucketSampler


class BucketingTrainer(Trainer):
    """Trainer with weighted loss and optional sequence-length bucket sampling."""
    def __init__(self, train_sampler=None, eval_sampler=None, **kwargs):
        super().__init__(**kwargs)
        self.model_accepts_loss_kwargs = False
        self.train_sampler = train_sampler
        self.eval_sampler = eval_sampler
    
    def get_train_dataloader(self):
        """Use the configured training sampler when provided."""
        if self.train_sampler is None:
            return super().get_train_dataloader()
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=self.train_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory if hasattr(self.args, 'dataloader_pin_memory') else False,
        )
    
    def get_eval_dataloader(self, eval_dataset=None):
        """Use the configured evaluation sampler when provided."""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        if self.eval_sampler is None:
            return super().get_eval_dataloader(eval_dataset)
        
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            sampler=self.eval_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory if hasattr(self.args, 'dataloader_pin_memory') else False,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Average step losses weighted by trajectory stability and token share."""
        sample_weights = inputs.pop("sample_weights", None)  # [B]
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits  # [B, L, V]

        # Causal LM shift: predict token t using tokens 0..t-1
        shift_logits = logits[..., :-1, :]  # [B, L-1, V]
        shift_labels = labels[..., 1:].contiguous()       # [B, L-1]

        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        valid_mask = shift_labels != -100
        valid_losses = loss_fct(shift_logits[valid_mask], shift_labels[valid_mask])
        token_losses = valid_losses.new_zeros(shift_labels.shape)
        token_losses[valid_mask] = valid_losses

        # Normalize each step by its number of valid (non-masked) target tokens.
        valid_counts = valid_mask.float().sum(dim=-1).clamp(min=1)  # [B]
        per_sample_loss = token_losses.sum(dim=-1) / valid_counts               # [B]

        if sample_weights is not None:
            per_sample_loss = per_sample_loss * sample_weights.to(per_sample_loss.device)

        loss = per_sample_loss.mean()
        return (loss, outputs) if return_outputs else loss

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None):
        """Preserve the first, best, and latest N checkpoints.

        Maintain the `checkpoint-first` and `checkpoint-best` symlinks.
        """
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        output_dir = Path(output_dir or self.args.output_dir)

        checkpoints = sorted(
            [d for d in output_dir.iterdir()
             if d.is_dir() and re.fullmatch(r"checkpoint-\d+", d.name)],
            key=lambda d: int(d.name.split("-")[-1])
        )

        if not checkpoints:
            return

        def update_symlink(link_path: Path, target: Path):
            if link_path.is_symlink() or link_path.exists():
                link_path.unlink()
            link_path.symlink_to(target.name)

        update_symlink(output_dir / "checkpoint-first", checkpoints[0])

        best_ckpt = getattr(self.state, "best_model_checkpoint", None)
        if best_ckpt:
            best_path = Path(best_ckpt)
            if best_path.exists():
                update_symlink(output_dir / "checkpoint-best", best_path)

        if len(checkpoints) <= self.args.save_total_limit:
            return

        protected = {checkpoints[0].name}
        if best_ckpt:
            protected.add(Path(best_ckpt).name)

        to_delete = [ckpt for ckpt in checkpoints if ckpt.name not in protected]
        num_to_delete = len(checkpoints) - self.args.save_total_limit

        import shutil
        for ckpt in to_delete[:num_to_delete]:
            logger.info(f"Deleting checkpoint: {ckpt}")
            shutil.rmtree(ckpt)

def train(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    training_args_config = config['training_args']
    model_config = config['model']
    data_config = config['data']
    lora_config_data = config.get('lora', None)
    
    set_seed(training_args_config.get('seed', 42))
    
    logger.info(f"Loading tokenizer from {model_config['name']}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config['name'],
            trust_remote_code=True,
            padding_side="right",
            local_files_only=True
        )
    except OSError as e:
        logger.error(f"Tokenizer not found locally at {model_config['name']}. Please download it first or check the path.")
        raise e

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info(f"Loading dataset from {data_config['train_path']}")
    
    use_bucketing = data_config.get('use_bucketing', False)
    bucket_boundaries = data_config.get('bucket_boundaries', None)
    
    if use_bucketing:
        logger.info("Sequence-length bucketing enabled.")
        if bucket_boundaries:
            logger.info(f"Using custom bucket boundaries: {bucket_boundaries}")
    
    full_dataset = MetaAgentSFTDataset(
        data_path=data_config['train_path'],
        tokenizer=tokenizer,
        max_seq_length=model_config.get('max_length', 4096),
        use_bucketing=use_bucketing,
        bucket_boundaries=bucket_boundaries,
        alpha=data_config.get('alpha', 2.0)
    )
    
    val_size = data_config.get('val_size', 0.05)
    if val_size > 0:
        train_size = int((1 - val_size) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        logger.info(f"Splitting dataset: Train={len(train_dataset)}, Val={len(eval_dataset)}")
    else:
        train_dataset = full_dataset
        eval_dataset = None
        logger.info(f"Using full dataset for training: {len(train_dataset)} examples")
    
    # Accelerate places each process on its local device.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device_map = None if local_rank >= 0 else "auto"
    logger.info(f"Loading model from {model_config['name']} (device_map={device_map})")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_config['name'],
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if training_args_config.get('bf16') else torch.float16,
            device_map=device_map,
            local_files_only=True,
            attn_implementation="flash_attention_2"
        )
    except OSError as e:
        logger.error(f"Model not found locally at {model_config['name']}. Please download it first or check the path.")
        raise e
    
    if lora_config_data:
        logger.info("Applying LoRA...")
        if training_args_config.get('gradient_checkpointing', True):
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_config_data.get('r', 16),
            lora_alpha=lora_config_data.get('lora_alpha', 32),
            lora_dropout=lora_config_data.get('lora_dropout', 0.05),
            target_modules=lora_config_data.get('target_modules', ["q_proj", "v_proj"])
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    collator = DataCollatorForSupervisedFineTuning(
        tokenizer=tokenizer,
        max_length=model_config.get('max_length', 4096)
    )
    
    train_sampler = None
    eval_sampler = None
    if use_bucketing:
        lengths, bucket_indices = full_dataset.get_bucket_info()
        
        per_device_train_batch_size = training_args_config.get('per_device_train_batch_size', 1)
        per_device_eval_batch_size = training_args_config.get('per_device_eval_batch_size', 1)
        
        if isinstance(train_dataset, torch.utils.data.Subset):
            train_subset_indices = train_dataset.indices
            train_subset_lengths = [lengths[orig_idx] for orig_idx in train_subset_indices]
            train_subset_bucket_indices = [bucket_indices[orig_idx] for orig_idx in train_subset_indices]
            
            train_sampler = BucketSampler(
                lengths=train_subset_lengths,
                bucket_indices=train_subset_bucket_indices,
                batch_size=per_device_train_batch_size,
                shuffle=True,
                drop_last=False,
                seed=training_args_config.get('seed', 42)
            )
            logger.info("Using BucketSampler for training")
            
            if eval_dataset is not None and isinstance(eval_dataset, torch.utils.data.Subset):
                eval_subset_indices = eval_dataset.indices
                eval_subset_lengths = [lengths[orig_idx] for orig_idx in eval_subset_indices]
                eval_subset_bucket_indices = [bucket_indices[orig_idx] for orig_idx in eval_subset_indices]
                
                eval_sampler = BucketSampler(
                    lengths=eval_subset_lengths,
                    bucket_indices=eval_subset_bucket_indices,
                    batch_size=per_device_eval_batch_size,
                    shuffle=False,
                    drop_last=False,
                    seed=None
                )
                logger.info("Using BucketSampler for evaluation")
        else:
            train_sampler = BucketSampler(
                lengths=lengths,
                bucket_indices=bucket_indices,
                batch_size=per_device_train_batch_size,
                shuffle=True,
                drop_last=False,
                seed=training_args_config.get('seed', 42)
            )
            logger.info("Using BucketSampler for training")
            
            if eval_dataset is not None:
                eval_sampler = BucketSampler(
                    lengths=lengths,
                    bucket_indices=bucket_indices,
                    batch_size=per_device_eval_batch_size,
                    shuffle=False,
                    drop_last=False,
                    seed=None
                )
                logger.info("Using BucketSampler for evaluation")
    
    training_args = TrainingArguments(
        output_dir=training_args_config['output_dir'],
        num_train_epochs=training_args_config.get('num_train_epochs', 3),
        per_device_train_batch_size=training_args_config.get('per_device_train_batch_size', 1),
        per_device_eval_batch_size=training_args_config.get('per_device_eval_batch_size', 1),
        gradient_accumulation_steps=training_args_config.get('gradient_accumulation_steps', 4),
        learning_rate=float(training_args_config.get('learning_rate', 2e-5)),
        weight_decay=training_args_config.get('weight_decay', 0.01),
        warmup_ratio=training_args_config.get('warmup_ratio', 0.03),
        lr_scheduler_type=training_args_config.get('lr_scheduler_type', 'cosine'),
        logging_steps=training_args_config.get('logging_steps', 10),
        save_strategy=training_args_config.get('save_strategy', 'steps'),
        save_steps=training_args_config.get('save_steps', 500),
        save_total_limit=training_args_config.get('save_total_limit', 10),
        eval_strategy=training_args_config.get('eval_strategy', 'steps') if eval_dataset else "no",
        eval_steps=training_args_config.get('eval_steps', 50),
        fp16=training_args_config.get('fp16', False),
        bf16=training_args_config.get('bf16', False),
        gradient_checkpointing=training_args_config.get('gradient_checkpointing', True),
        ddp_find_unused_parameters=False,
        report_to=training_args_config.get('report_to', "none"),
        remove_unused_columns=False,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="loss" if eval_dataset else None,
        dataloader_drop_last=False,
        dataloader_num_workers=training_args_config.get('dataloader_num_workers', 0)
    )
    
    trainer = BucketingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        train_sampler=train_sampler,
        eval_sampler=eval_sampler,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info(f"Saving model to {training_args_config['output_dir']}")
    trainer.save_model(training_args_config['output_dir'])
    tokenizer.save_pretrained(training_args_config['output_dir'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to training config yaml")
    args = parser.parse_args()
    
    train(args.config)

