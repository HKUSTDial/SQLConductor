"""Convert value retrieval results from DataItem pickles to Task pickles."""

import pickle
from copy import deepcopy
from pathlib import Path
from typing import List

from loguru import logger
from tqdm import tqdm

from src.preprocess.config import PreprocessConfigManager
from src.preprocess.dataset import load_dataset, DataItem
from src.runner.task import Task
from src.database.database_manager import DatabaseManager
from src.database.utils import build_table_profile_statement


def convert_to_tasks(config_manager: PreprocessConfigManager) -> List[Task]:
    """
    Load value retrieval results and convert to SQLConductor Task objects.
    
    The Task objects include:
    - schema_context: DDL-like text with injected retrieved value examples
    - table_schema_dict: Dict[str, TableSchema] with injected value examples
    - retrieved_values: The raw retrieved values dict
    """
    vr_save_path = config_manager.value_retrieval_config.save_path
    output_save_path = config_manager.output_config.save_path
    max_example_length = config_manager.output_config.max_example_length

    logger.info(f"Loading value retrieval results from {vr_save_path}")
    dataset = load_dataset(vr_save_path)

    first_item: DataItem = dataset[0]
    database_root_dir = str(Path(first_item.database_path).parent.parent)
    logger.info(f"Inferred database_root_dir: {database_root_dir}")

    tasks = []
    for item in tqdm(dataset, desc="Converting to Task format"):
        item: DataItem

        database_schema = DatabaseManager.get_database_schema(item.database_id, database_root_dir)
        table_schema_dict = deepcopy(database_schema.tables)

        # Restrict the schema to the configured table selection.
        if item.allowed_tables is not None:
            allowed_set = {t.lower() for t in item.allowed_tables}
            table_schema_dict = {
                tbl: schema
                for tbl, schema in table_schema_dict.items()
                if tbl.lower() in allowed_set
            }
            if not table_schema_dict:
                logger.warning(
                    f"[Q{item.question_id}] per-question table filter left 0 tables "
                    f"(allowed={item.allowed_tables}). Retaining the full schema."
                )
                table_schema_dict = deepcopy(database_schema.tables)

        schema_after = item.database_schema_after_value_retrieval
        if schema_after:
            for t_name, t_info in schema_after.get("tables", {}).items():
                if t_name in table_schema_dict:
                    for c_name, c_info in t_info.get("columns", {}).items():
                        if c_name in table_schema_dict[t_name].columns:
                            table_schema_dict[t_name].columns[c_name].value_examples = c_info.get("value_examples", [])

        ddl_statements = []
        for table_name, table_schema_obj in table_schema_dict.items():
            ddl_stmt = build_table_profile_statement(
                table_schema_obj.to_dict(),
                add_value_description=True,
                add_column_description=True,
                add_value_examples=True,
                add_expanded_column_name=True,
                max_example_length=max_example_length,
            )
            ddl_statements.append(ddl_stmt)
        schema_context = "\n".join(ddl_statements)

        task = Task(
            question_id=item.question_id,
            db_id=item.database_id,
            question=item.question,
            evidence=item.evidence,
            sql=item.gold_sql,
            difficulty=item.difficulty,
            schema_context=schema_context,
            table_schema_dict=table_schema_dict,
            retrieved_values=item.retrieved_values,
        )
        tasks.append(task)

    output_path = Path(output_save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(tasks, f)
    logger.info(f"Saved {len(tasks)} tasks to {output_path}")

    return tasks
