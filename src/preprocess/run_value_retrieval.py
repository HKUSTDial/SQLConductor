"""
Step 3: Value Retrieval
Extract keywords via LLM, query vector DB, update schema with retrieved values.

Usage:
    python -m src.preprocess.run_value_retrieval --config config/preprocess/bird_dev.toml
"""

import argparse

from loguru import logger

from src.preprocess.config import PreprocessConfigManager
from src.preprocess.value_retrieval import ValueRetrievalRunner


def main():
    parser = argparse.ArgumentParser(description="Step 3: Value Retrieval")
    parser.add_argument("--config", type=str, required=True, help="Path to preprocessing TOML config")
    args = parser.parse_args()

    config_mgr = PreprocessConfigManager()
    config_mgr.load(args.config)

    logger.info("Starting value retrieval...")
    runner = ValueRetrievalRunner(config_mgr)
    runner.run()
    logger.info("Value retrieval completed.")


if __name__ == "__main__":
    main()
