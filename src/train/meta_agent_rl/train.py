"""GRPO training with YAML configuration and CLI overrides."""

import argparse
import yaml
import sys
import os
from pathlib import Path
from loguru import logger
from src.train.meta_agent_rl.trainer import GRPOTrainer


def setup_logging(config: dict):
    """Configure log levels and process/module filters from YAML settings."""
    system_config = config.get('system', {})
    logging_config = system_config.get('logging', {})
    log_level = system_config.get('log_level', 'INFO').upper()
    main_process_only = logging_config.get('main_process_only', True)
    suppress_modules = logging_config.get('suppress_modules', [])
    
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', 0)))
    is_main_process = (local_rank == 0)
    
    logger.remove()
    
    def log_filter(record):
        if main_process_only and not is_main_process:
            return False
        module_name = record["name"]
        for suppressed in suppress_modules:
            if module_name and module_name.startswith(suppressed):
                return record["level"].no >= logger.level("WARNING").no
        return True
    
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        filter=log_filter,
        colorize=True
    )
    if is_main_process:
        logger.info(f"Logging configured: level={log_level}, main_process_only={main_process_only}, suppressed={len(suppress_modules)} modules")


def load_yaml_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_nested(d: dict, *keys, default=None):
    """Safely get nested dictionary values."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d if d is not None else default


def main():
    parser = argparse.ArgumentParser(
        description="Train MetaAgent using GRPO/GDPO RL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to YAML config file (e.g., config/train/rl/grpo_default.yaml)"
    )
    
    parser.add_argument("--model_path", type=str, default=None, help="Override: Path to base model")
    parser.add_argument("--train_tasks", type=str, default=None, help="Override: Path to training tasks")
    parser.add_argument("--oracle_summary", type=str, default=None, help="Override: Path to oracle trajectory summary")
    parser.add_argument("--output_dir", type=str, default=None, help="Override: Output directory for checkpoints")
    
    parser.add_argument("--epochs", type=int, default=None, help="Override: Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override: Batch size per GPU")
    parser.add_argument("--group_size", type=int, default=None, help="Override: Group size (G) for GRPO")
    parser.add_argument("--lr", type=float, default=None, help="Override: Learning rate")
    parser.add_argument("--beta", type=float, default=None, help="Override: KL divergence coefficient")
    parser.add_argument("--reward_lambda", type=float, default=None, help="Override: Outcome reward weight (executability weight is 1 - lambda)")
    parser.add_argument("--curriculum_threshold", type=float, default=None, help="Override: Retain questions with MCTS success ratio strictly below this threshold")
    parser.add_argument("--rollout_timeout", type=int, default=None, help="Override: Rollout timeout in seconds")
    
    parser.add_argument("--use_lora", type=lambda x: x.lower() == 'true', default=None, help="Override: Enable/disable LoRA")
    parser.add_argument("--lora_r", type=int, default=None, help="Override: LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=None, help="Override: LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=None, help="Override: LoRA dropout")
    
    parser.add_argument("--use_8bit", type=lambda x: x.lower() == 'true', default=None, help="Override: Enable/disable 8-bit quantization")
    
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    
    setup_logging(config)
    
    # Extract values with CLI override priority
    model_path = args.model_path or get_nested(config, 'rl', 'base_model_path')
    train_tasks = args.train_tasks or get_nested(config, 'system', 'paths', 'train_tasks')
    oracle_summary = args.oracle_summary or get_nested(config, 'system', 'paths', 'oracle_summary')
    output_dir = args.output_dir or get_nested(config, 'system', 'paths', 'output_dir', default='checkpoints/rl_gdpo')
    
    epochs = int(args.epochs if args.epochs is not None else get_nested(config, 'rl', 'training', 'epochs', default=1))
    batch_size = int(args.batch_size if args.batch_size is not None else get_nested(config, 'rl', 'training', 'batch_size', default=1))
    group_size = int(args.group_size if args.group_size is not None else get_nested(config, 'rl', 'training', 'group_size', default=4))
    learning_rate = float(args.lr if args.lr is not None else get_nested(config, 'rl', 'training', 'learning_rate', default=1e-6))
    beta = float(args.beta if args.beta is not None else get_nested(config, 'rl', 'training', 'beta', default=0.1))
    reward_lambda = float(args.reward_lambda if args.reward_lambda is not None else get_nested(config, 'rl', 'training', 'lambda', default=0.8))
    curriculum_threshold = args.curriculum_threshold if args.curriculum_threshold is not None else get_nested(config, 'rl', 'training', 'curriculum_threshold')
    if curriculum_threshold is not None:
        curriculum_threshold = float(curriculum_threshold)
    rollout_timeout = int(args.rollout_timeout if args.rollout_timeout is not None else get_nested(config, 'rl', 'training', 'rollout_timeout', default=600))
    serial_group_processing = bool(get_nested(config, 'rl', 'training', 'serial_group_processing', default=False))
    
    use_lora = bool(args.use_lora if args.use_lora is not None else get_nested(config, 'rl', 'lora', 'enabled', default=True))
    lora_r = int(args.lora_r if args.lora_r is not None else get_nested(config, 'rl', 'lora', 'r', default=16))
    lora_alpha = int(args.lora_alpha if args.lora_alpha is not None else get_nested(config, 'rl', 'lora', 'lora_alpha', default=32))
    lora_dropout = float(args.lora_dropout if args.lora_dropout is not None else get_nested(config, 'rl', 'lora', 'lora_dropout', default=0.05))
    lora_target_modules = get_nested(config, 'rl', 'lora', 'target_modules', default=["q_proj", "k_proj", "v_proj", "o_proj"])
    
    lora_config = {
        'enabled': use_lora,
        'r': lora_r,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
        'target_modules': lora_target_modules
    }
    
    use_8bit = args.use_8bit if args.use_8bit is not None else bool(get_nested(config, 'rl', 'quantization', 'enabled', default=False))
    load_in_8bit = bool(get_nested(config, 'rl', 'quantization', 'load_in_8bit', default=True))
    llm_int8_skip_modules = get_nested(config, 'rl', 'quantization', 'llm_int8_skip_modules', default=None)
    
    quantization_config = {
        'enabled': use_8bit,
        'load_in_8bit': load_in_8bit,
        'llm_int8_skip_modules': llm_int8_skip_modules
    }
    
    checkpoint_config = {
        'save_every_steps': int(get_nested(config, 'rl', 'checkpoint', 'save_every_steps', default=50)),
        'max_checkpoints': int(get_nested(config, 'rl', 'checkpoint', 'max_checkpoints', default=3)),
        'gated_best': {
            'enabled': bool(get_nested(config, 'rl', 'checkpoint', 'gated_best', 'enabled', default=True)),
            'kl_threshold': float(get_nested(config, 'rl', 'checkpoint', 'gated_best', 'kl_threshold', default=0.05)),
            'use_reward_ema': bool(get_nested(config, 'rl', 'checkpoint', 'gated_best', 'use_reward_ema', default=True)),
            'ema_alpha': float(get_nested(config, 'rl', 'checkpoint', 'gated_best', 'ema_alpha', default=0.1)),
        },
        'kl_warning_threshold': float(get_nested(config, 'rl', 'checkpoint', 'kl_warning_threshold', default=0.1)),
    }
    
    if not model_path:
        raise ValueError("Model path is required. Set 'paths.model_path' in config or use --model_path")
    if not train_tasks:
        raise ValueError("Training tasks path is required. Set 'paths.train_tasks' in config or use --train_tasks")
    if not oracle_summary:
        raise ValueError("Oracle summary path is required. Set 'paths.oracle_summary' in config or use --oracle_summary")
    
    from src.system.config import config as system_config
    system_config.load_config(args.config)
    
    trainer = GRPOTrainer(
        model_name_or_path=model_path,
        train_tasks_path=train_tasks,
        trajectory_summary_path=oracle_summary,
        output_dir=output_dir,
        learning_rate=learning_rate,
        batch_size=batch_size,
        group_size=group_size,
        max_epochs=epochs,
        beta=beta,
        reward_lambda=reward_lambda,
        curriculum_threshold=curriculum_threshold,
        rollout_timeout=rollout_timeout,
        lora_config=lora_config,
        quantization_config=quantization_config,
        checkpoint_config=checkpoint_config,
        serial_group_processing=serial_group_processing
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
