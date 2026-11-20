import argparse
import sys
import yaml
from pathlib import Path
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.trajectories.stateful_trajectory_processor import StatefulTrajectoryProcessor
from src.trajectories.trajectory_summary_processor import TrajectorySummaryProcessor
from src.trajectories.trajectory_selection_strategy import get_strategy


def setup_traj_processing_logging(log_level: str = "INFO"):
    """Configure loguru logger with specified log level."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level
    )

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_log_level(value: str) -> str:
    """Parse log level (case-insensitive) and return uppercase."""
    valid_levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
    upper_value = value.upper()
    if upper_value not in valid_levels:
        raise argparse.ArgumentTypeError(
            f"Invalid log level: {value}. Choose from: {', '.join(valid_levels)}"
        )
    return upper_value

def main():
    parser = argparse.ArgumentParser(description="Run stateful trajectory processing.")
    parser.add_argument("action", choices=["summary", "reasoning"], help="Action to perform: summary or reasoning")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing PKL files.")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file path.")
    parser.add_argument("--ref_data", type=str, required=True, help="Path to reference data (gold SQLs) JSON.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--selection_strategy", type=str, default=None, 
                       help="Trajectory selection strategy. Default: 'shortest'")
    
    parser.add_argument("--model", type=str, default=None, help="Model to use for reasoning generation.")
    parser.add_argument("--processes", type=int, default=None, help="Number of parallel processes.")
    parser.add_argument("--base_url", type=str, default=None, help="API Base URL")
    parser.add_argument("--api_key", type=str, default=None, help="API Key")
    parser.add_argument("--log-level", type=parse_log_level, default="INFO",
                       help="Set logging level: trace, debug, info, success, warning, error, critical (default: info)")

    args = parser.parse_args()
    setup_traj_processing_logging(args.log_level)

    model_config = {
        "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "base_url": "http://localhost:9999/v1",
        "api_key": "sk-demo",
        "temperature": 0.7,
        "max_tokens": 1000
    }
    n_processes = 1
    selection_strategy = None

    if args.config:
        try:
            cfg = load_config(args.config)
            reasoning_cfg = cfg.get("stateful_reasoning", {})
            if "model" in reasoning_cfg:
                model_config.update(reasoning_cfg["model"])
            if "execution" in reasoning_cfg:
                n_processes = reasoning_cfg["execution"].get("n_parallel_processes", n_processes)
            if "selection_strategy" in reasoning_cfg:
                selection_strategy = reasoning_cfg["selection_strategy"]
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            sys.exit(1)

    # CLI overrides
    if args.model:
        model_config["model_name"] = args.model
    if args.base_url:
        model_config["base_url"] = args.base_url
    if args.api_key:
        model_config["api_key"] = args.api_key
    if args.processes:
        n_processes = args.processes
    if args.selection_strategy:
        selection_strategy = args.selection_strategy

    logger.info(f"Starting processing from {args.input_dir} to {args.output_file}")
    
    if args.action == "summary":
        if selection_strategy:
            try:
                strategy = get_strategy(selection_strategy)
                processor = TrajectorySummaryProcessor(selection_strategy=strategy)
                logger.info(f"Using selection strategy: {selection_strategy}")
            except ValueError as e:
                logger.error(f"Invalid selection strategy: {e}")
                sys.exit(1)
        else:
            processor = TrajectorySummaryProcessor()
            logger.info("Using default selection strategy: shortest")
        
        logger.info(f"Running Summary Phase with {n_processes} processes")
        processor.process_dataset(
            input_dir=Path(args.input_dir),
            output_file=Path(args.output_file),
            ref_data_path=Path(args.ref_data),
            n_processes=n_processes
        )
    
    elif args.action == "reasoning":
        processor = StatefulTrajectoryProcessor(
            model_path=model_config["model_name"],
            base_url=model_config.get("base_url"),
            api_key=model_config.get("api_key"),
            temperature=model_config.get("temperature", 0.7),
            top_p=model_config.get("top_p", 0.9),
            max_tokens=model_config.get("max_tokens", 1000)
        )
        logger.info(f"Running Reasoning Phase using model: {model_config['model_name']} with {n_processes} processes")
        processor.process_dataset(
            input_dir=Path(args.input_dir),
            output_file=Path(args.output_file),
            ref_data_path=Path(args.ref_data),
            n_processes=n_processes
        )
    
    logger.info("Processing completed.")

if __name__ == "__main__":
    main()

