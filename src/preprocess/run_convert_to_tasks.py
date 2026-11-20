"""
Step 4: Convert to Task PKL
Convert the value retrieval result into SQLConductor Task pkl for inference.

Usage:
    python -m src.preprocess.run_convert_to_tasks --config config/preprocess/bird_dev.toml
"""

import argparse

from loguru import logger

from src.preprocess.config import PreprocessConfigManager
from src.preprocess.convert_to_tasks import convert_to_tasks


def main():
    parser = argparse.ArgumentParser(description="Step 4: Convert to Task PKL")
    parser.add_argument("--config", type=str, required=True, help="Path to preprocessing TOML config")
    args = parser.parse_args()

    config_mgr = PreprocessConfigManager()
    config_mgr.load(args.config)

    logger.info("Converting to Task format...")
    tasks = convert_to_tasks(config_mgr)
    logger.info(f"Conversion completed. {len(tasks)} tasks saved.")


if __name__ == "__main__":
    main()
