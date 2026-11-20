"""
Step 2: Create Vector Database
Build ChromaDB vector stores for each database in the dataset.

Usage:
    # Process all databases in parallel
    python -m src.preprocess.run_create_vector_db --config config/preprocess/bird_dev.toml --n_parallel 4

    # Process a single database
    python -m src.preprocess.run_create_vector_db --config config/preprocess/bird_dev.toml --db_path data/bird/dev/dev_databases/california_schools/california_schools.sqlite
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from src.preprocess.config import PreprocessConfigManager
from src.preprocess.dataset import load_dataset
from src.preprocess.vector_db import make_vector_db_for_db_path


def _load_table_filter(config_mgr: PreprocessConfigManager):
    """Load the optional table_filter from the dataset config.

    Returns a dict mapping db_id -> set of allowed table names, or None if
    no table_filter is configured.
    """
    table_filter_path = config_mgr.dataset_config.table_filter
    if not table_filter_path:
        return None
    path = Path(table_filter_path)
    if not path.exists():
        logger.warning(f"table_filter file not found: {path}, ignoring for vector DB creation.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    table_filter = {db_id: set(tables) for db_id, tables in raw.items()}
    logger.info(
        f"Loaded table_filter for vector DB creation: "
        f"{len(table_filter)} databases, "
        f"{sum(len(v) for v in table_filter.values())} tables total"
    )
    return table_filter


def main():
    parser = argparse.ArgumentParser(description="Step 2: Create vector databases")
    parser.add_argument("--config", type=str, required=True, help="Path to preprocessing TOML config")
    parser.add_argument("--db_path", type=str, default=None, help="Process a single database path instead of all")
    parser.add_argument("--n_parallel", type=int, default=None, help="Number of databases to process in parallel")
    args = parser.parse_args()

    config_mgr = PreprocessConfigManager()
    config_mgr.load(args.config)
    vdb_config = config_mgr.vector_database_config
    n_parallel = args.n_parallel or vdb_config.n_parallel

    table_filter = _load_table_filter(config_mgr)

    if args.db_path:
        db_id = Path(args.db_path).stem
        allowed = table_filter.get(db_id) if table_filter else None
        logger.info(f"Creating vector DB for single database: {args.db_path}")
        make_vector_db_for_db_path(args.db_path, vdb_config, allowed_tables=allowed)
    else:
        dataset_save_path = config_mgr.dataset_config.save_path
        logger.info(f"Loading dataset from {dataset_save_path}")
        dataset = load_dataset(dataset_save_path)
        db_paths = dataset.get_all_database_paths()
        logger.info(f"Found {len(db_paths)} unique databases")

        with ThreadPoolExecutor(max_workers=n_parallel) as executor:
            futures = {}
            for db_path in db_paths:
                db_id = Path(db_path).stem
                allowed = table_filter.get(db_id) if table_filter else None
                future = executor.submit(
                    make_vector_db_for_db_path, db_path, vdb_config, allowed_tables=allowed,
                )
                futures[future] = db_path
            for future in tqdm(as_completed(futures), total=len(futures), desc="Creating vector DBs"):
                db_path = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Unhandled exception for {db_path}: {e}")

    logger.info("Vector database creation completed.")


if __name__ == "__main__":
    main()
