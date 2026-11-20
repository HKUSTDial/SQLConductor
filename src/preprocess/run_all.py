"""
Run the full inference preprocessing pipeline (all 4 steps).

Usage:
    python -m src.preprocess.run_all --config config/preprocess/bird_dev.toml
    python -m src.preprocess.run_all --config config/preprocess/bird_dev.toml --n_parallel_vector_db 4
    python -m src.preprocess.run_all --config config/preprocess/bird_dev.toml --skip_vector_db  # skip if already created
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from src.preprocess.config import PreprocessConfigManager
from src.preprocess.dataset import DatasetFactory, save_dataset, load_dataset
from src.preprocess.vector_db import make_vector_db_for_db_path
from src.preprocess.value_retrieval import ValueRetrievalRunner
from src.preprocess.convert_to_tasks import convert_to_tasks


def main():
    parser = argparse.ArgumentParser(description="Run full inference preprocessing pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to preprocessing TOML config")
    parser.add_argument("--n_parallel_vector_db", type=int, default=None, help="Parallelism for vector DB creation")
    parser.add_argument("--skip_dataset", action="store_true", help="Skip Step 1 (dataset preprocessing)")
    parser.add_argument("--skip_vector_db", action="store_true", help="Skip Step 2 (vector DB creation)")
    parser.add_argument("--skip_value_retrieval", action="store_true", help="Skip Step 3 (value retrieval)")
    parser.add_argument("--skip_convert", action="store_true", help="Skip Step 4 (convert to tasks)")
    args = parser.parse_args()

    config_mgr = PreprocessConfigManager()
    config_mgr.load(args.config)

    if not args.skip_dataset:
        logger.info("Step 1: Dataset Preprocessing")
        cfg = config_mgr.dataset_config
        dataset = DatasetFactory.get_dataset(cfg)
        logger.info(f"Loaded {len(dataset)} items from {cfg.type}/{cfg.split}")
        save_dataset(dataset, cfg.save_path)
    else:
        logger.info("Step 1 skipped (--skip_dataset)")

    if not args.skip_vector_db:
        logger.info("Step 2: Create Vector Database")
        vdb_config = config_mgr.vector_database_config
        n_par = args.n_parallel_vector_db or vdb_config.n_parallel
        dataset = load_dataset(config_mgr.dataset_config.save_path)
        db_paths = dataset.get_all_database_paths()
        logger.info(f"Found {len(db_paths)} unique databases")

        table_filter = None
        table_filter_path = config_mgr.dataset_config.table_filter
        if table_filter_path and Path(table_filter_path).exists():
            with open(table_filter_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            table_filter = {db_id: set(tables) for db_id, tables in raw.items()}
            logger.info(f"Loaded table_filter for vector DB: {len(table_filter)} databases")

        with ThreadPoolExecutor(max_workers=n_par) as executor:
            futures = {}
            for p in db_paths:
                db_id = Path(p).stem
                allowed = table_filter.get(db_id) if table_filter else None
                future = executor.submit(make_vector_db_for_db_path, p, vdb_config, allowed_tables=allowed)
                futures[future] = p
            for future in tqdm(as_completed(futures), total=len(futures), desc="Creating vector DBs"):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error: {e}")
        logger.info("Vector database creation completed.")
    else:
        logger.info("Step 2 skipped (--skip_vector_db)")

    if not args.skip_value_retrieval:
        logger.info("Step 3: Value Retrieval")
        runner = ValueRetrievalRunner(config_mgr)
        runner.run()
        logger.info("Value retrieval completed.")
    else:
        logger.info("Step 3 skipped (--skip_value_retrieval)")

    if not args.skip_convert:
        logger.info("Step 4: Convert to Task PKL")
        tasks = convert_to_tasks(config_mgr)
        logger.info(f"Conversion completed. {len(tasks)} tasks ready for inference.")
    else:
        logger.info("Step 4 skipped (--skip_convert)")

    logger.info("Preprocessing pipeline finished!")


if __name__ == "__main__":
    main()
