"""
Step 1: Dataset Preprocessing
Load raw dataset JSON, build database schemas, save as pkl.

Usage:
    python -m src.preprocess.run_preprocess_dataset --config config/preprocess/bird_dev.toml
"""

import sys
import argparse

from loguru import logger

from src.preprocess.config import PreprocessConfigManager
from src.preprocess.dataset import DatasetFactory, save_dataset


def main():
    parser = argparse.ArgumentParser(description="Step 1: Preprocess dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to preprocessing TOML config")
    args = parser.parse_args()

    config_mgr = PreprocessConfigManager()
    config_mgr.load(args.config)
    cfg = config_mgr.dataset_config

    logger.info(f"Preprocessing dataset: {cfg.type} / {cfg.split}")
    dataset = DatasetFactory.get_dataset(cfg)
    logger.info(f"Loaded {len(dataset)} items")

    save_dataset(dataset, cfg.save_path)
    logger.info(f"Dataset saved to {cfg.save_path}")


if __name__ == "__main__":
    main()
