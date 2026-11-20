"""
Dataset loading and DataItem definition for the preprocessing pipeline.
"""

import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Any, Dict

from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.preprocess.config import DatasetConfig
from src.database.utils import load_database_schema_dict


class DataItem(BaseModel):
    """
    A single data item in the preprocessing pipeline.
    """
    question_id: int = Field(..., description="The question id")
    question: str = Field(..., description="The question text")
    evidence: str = Field(default="", description="The evidence / hint")
    gold_sql: str = Field(default="", description="The gold SQL query")
    difficulty: str = Field(default="", description="The difficulty level")
    database_id: str = Field(..., description="The database id")
    database_path: str = Field(..., description="Path to the .sqlite file")
    database_schema: Dict[str, Any] = Field(..., description="The full database schema dict")

    # Optional table selection applied when converting this item to a Task.
    allowed_tables: Optional[List[str]] = Field(default=None, description="Tables allowed for this question (gold + FK neighbours)")

    question_keywords: Optional[List[str]] = Field(default=None, description="Extracted keywords")
    retrieved_values: Optional[Dict[str, Dict[str, Any]]] = Field(default=None, description="Retrieved values per table/column")
    database_schema_after_value_retrieval: Optional[Dict[str, Any]] = Field(default=None, description="Schema with injected retrieved values")

    value_retrieval_time: Optional[float] = Field(default=None)
    value_retrieval_llm_cost: Optional[Dict[str, Any]] = Field(default=None)
    total_time: Optional[float] = Field(default=None)
    total_llm_cost: Optional[Dict[str, Any]] = Field(default=None)


class BaseDataset(ABC):
    """Abstract base dataset class."""
    _config: DatasetConfig = None
    _data: List[DataItem] = None
    _database_schema_cache: Dict[str, Any] = {}
    _table_filter: Optional[Dict[str, List[str]]] = None
    _per_question_table_filter: Optional[Dict[str, List[str]]] = None

    def __init__(self, dataset_config: DatasetConfig):
        self._config = dataset_config
        self._database_schema_cache = {}
        self._table_filter = self._load_table_filter()
        self._per_question_table_filter = self._load_per_question_table_filter()
        self._data = self._load_data()

    def _load_table_filter(self) -> Optional[Dict[str, List[str]]]:
        """Load the optional table filter file (db_id -> list of table names to keep)."""
        if not self._config.table_filter:
            return None
        filter_path = Path(self._config.table_filter)
        if not filter_path.exists():
            logger.warning(f"table_filter file not found: {filter_path}, ignoring.")
            return None
        with open(filter_path, "r", encoding="utf-8") as f:
            table_filter = json.load(f)
        total_dbs = len(table_filter)
        total_tables = sum(len(v) for v in table_filter.values())
        logger.info(
            f"Loaded table_filter from {filter_path}: "
            f"{total_dbs} databases, {total_tables} tables total"
        )
        return table_filter

    def _load_per_question_table_filter(self) -> Optional[Dict[str, List[str]]]:
        """Load the optional per-question table filter (question_id -> list of table names)."""
        if not self._config.per_question_table_filter:
            return None
        filter_path = Path(self._config.per_question_table_filter)
        if not filter_path.exists():
            logger.warning(f"per_question_table_filter file not found: {filter_path}, ignoring.")
            return None
        with open(filter_path, "r", encoding="utf-8") as f:
            pq_filter = json.load(f)
        logger.info(
            f"Loaded per_question_table_filter from {filter_path}: "
            f"{len(pq_filter)} questions"
        )
        return pq_filter

    def _load_database_schema(self, database_id: str) -> Dict[str, Any]:
        if database_id in self._database_schema_cache:
            return self._database_schema_cache[database_id]
        database_path = self._get_database_path(database_id)
        database_dir = str(Path(database_path).parent)
        db_root_dir = str(Path(database_path).parent.parent)
        allowed_tables = None
        if self._table_filter and database_id in self._table_filter:
            allowed_tables = set(self._table_filter[database_id])
        schema = _load_database_schema_dict_for_preprocess(database_path, allowed_tables=allowed_tables)
        self._database_schema_cache[database_id] = schema
        return schema

    @abstractmethod
    def _load_data(self) -> List[DataItem]:
        pass

    @abstractmethod
    def _get_database_path(self, database_id: str) -> str:
        pass

    def get_all_database_paths(self) -> List[str]:
        return list(set([item.database_path for item in self._data]))

    def get_all_database_ids(self) -> List[str]:
        return list(set([item.database_id for item in self._data]))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int):
        return self._data[index]

    def __iter__(self):
        return iter(self._data)


class BirdDataset(BaseDataset):
    """BIRD dataset loader."""

    def _load_data(self) -> List[DataItem]:
        data_path = Path(self._config.root_path) / self._config.split / f"{self._config.split}.json"
        with open(data_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        if self._config.max_samples is not None:
            data_list = data_list[:self._config.max_samples]

        data = []
        for item in tqdm(data_list, desc="Loading BIRD data"):
            question_id = item.get("question_id")
            database_id = item.get("db_id")
            database_path = self._get_database_path(database_id)
            database_schema = self._load_database_schema(database_id)
            allowed_tables = (
                self._per_question_table_filter.get(str(question_id))
                if self._per_question_table_filter is not None
                else None
            )
            data.append(DataItem(
                question_id=question_id,
                question=item.get("question", ""),
                evidence=item.get("evidence", ""),
                gold_sql=item.get("SQL", ""),
                difficulty=item.get("difficulty", ""),
                database_id=database_id,
                database_path=database_path,
                database_schema=database_schema,
                allowed_tables=allowed_tables,
            ))
        return data

    def _get_database_path(self, database_id: str) -> str:
        return str(Path(self._config.root_path) / self._config.split / f"{self._config.split}_databases" / database_id / f"{database_id}.sqlite")


class SpiderDataset(BaseDataset):
    """Spider dataset loader."""

    def _load_data(self) -> List[DataItem]:
        if self._config.split == "dev":
            data_path = Path(self._config.root_path) / "dev.json"
        elif self._config.split == "test":
            data_path = Path(self._config.root_path) / "test.json"
        else:
            raise ValueError(f"Invalid split: {self._config.split}")

        with open(data_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        if self._config.max_samples is not None:
            data_list = data_list[:self._config.max_samples]

        data = []
        for question_id, item in tqdm(enumerate(data_list), desc="Loading Spider data"):
            database_id = item.get("db_id")
            database_path = self._get_database_path(database_id)
            database_schema = self._load_database_schema(database_id)
            data.append(DataItem(
                question_id=question_id,
                question=item.get("question", ""),
                evidence="",
                gold_sql=item.get("query", ""),
                difficulty="",
                database_id=database_id,
                database_path=database_path,
                database_schema=database_schema,
            ))
        return data

    def _get_database_path(self, database_id: str) -> str:
        if self._config.split == "dev":
            return str(Path(self._config.root_path) / "database" / database_id / f"{database_id}.sqlite")
        elif self._config.split == "test":
            return str(Path(self._config.root_path) / "test_database" / database_id / f"{database_id}.sqlite")
        else:
            raise ValueError(f"Invalid split: {self._config.split}")


class DatasetFactory:
    """Factory to create dataset instances."""

    @staticmethod
    def get_dataset(dataset_config: DatasetConfig) -> BaseDataset:
        if dataset_config.type == "bird":
            return BirdDataset(dataset_config)
        elif dataset_config.type == "spider":
            return SpiderDataset(dataset_config)
        else:
            raise ValueError(f"Invalid dataset type: {dataset_config.type}")


def save_dataset(dataset: BaseDataset, save_path: str) -> None:
    """Save dataset to a pkl file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)
    logger.info(f"Dataset saved to {save_path}")


def load_dataset(load_path: str) -> BaseDataset:
    """Load dataset from a pkl file."""
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {load_path}")
    with open(load_path, "rb") as f:
        dataset = pickle.load(f)
    logger.info(f"Dataset loaded from {load_path} ({len(dataset)} items)")
    return dataset



import sqlite3
import chardet
import pandas as pd
import re


def _execute_sql_readonly(db_path: str, sql: str):
    """Execute SQL in read-only mode, return rows or empty list on error."""
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            conn.text_factory = lambda x: str(x, "utf-8", errors="replace")
            cursor = conn.cursor()
            cursor.execute(sql)
            return cursor.fetchall()
    except Exception:
        return []


def _load_table_names(db_path: str) -> List[str]:
    rows = _execute_sql_readonly(db_path, "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence';")
    return [r[0] for r in rows]


def _load_column_names_and_types(db_path: str, table_name: str) -> List[tuple]:
    rows = _execute_sql_readonly(db_path, f"PRAGMA table_info(`{table_name}`);")
    return [(r[1], r[2]) for r in rows]


def _load_primary_keys(db_path: str, table_name: str) -> List[str]:
    rows = _execute_sql_readonly(db_path, f"PRAGMA table_info(`{table_name}`);")
    return [r[1] for r in rows if r[5] != 0]


def _load_foreign_keys(db_path: str, table_name: str) -> List[tuple]:
    rows = _execute_sql_readonly(db_path, f"PRAGMA foreign_key_list(`{table_name}`);")
    dedup = set()
    for fk in rows:
        source_col = fk[3]
        target_table = fk[2]
        target_col = fk[4]
        dedup.add((source_col, target_table, target_col))

    result = []
    for source_col, target_table, target_col in dedup:
        if target_col is None:
            pks = _load_primary_keys(db_path, target_table)
            if pks:
                target_col = pks[0]
            else:
                continue
        result.append((table_name, source_col, target_table, target_col))
    return result


def _load_value_examples(db_path: str, table_name: str, column_name: str, max_num: int = 3, max_len: int = 100) -> List[str]:
    sql = f"SELECT DISTINCT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL AND `{column_name}` != '' AND length(cast(`{column_name}` as text)) <= {max_len} LIMIT {max_num};"
    rows = _execute_sql_readonly(db_path, sql)
    return [str(r[0]) for r in rows]


def _load_value_statistics(db_path: str, table_name: str, column_name: str) -> Dict[str, Any]:
    sql = f"""
        SELECT COUNT(`{column_name}`) AS total_count, COUNT(DISTINCT `{column_name}`) AS distinct_count, SUM(CASE WHEN `{column_name}` IS NULL THEN 1 ELSE 0 END) AS null_count
        FROM (SELECT `{column_name}` FROM `{table_name}` LIMIT 100000) AS limited_dataset;
    """
    rows = _execute_sql_readonly(db_path, sql)
    if rows:
        return {"total_count": rows[0][0], "distinct_count": rows[0][1], "null_count": rows[0][2]}
    return {"total_count": 0, "distinct_count": 0, "null_count": 0}


def _normalize_description_string(description: str) -> str:
    description = description.replace("\r", " ").replace("\n", " ").replace("commonsense evidence:", "").strip()
    while "  " in description:
        description = description.replace("  ", " ")
    return description.strip()


def _load_database_description(db_path: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load database description CSVs (BIRD format)."""
    db_dir = Path(db_path).parent
    desc_dir = db_dir / "database_description"
    if not desc_dir.exists():
        return {}
    database_description = {}
    for csv_file in desc_dir.glob("*.csv"):
        table_name_lower = csv_file.stem.lower().strip()
        encoding_type = chardet.detect(csv_file.read_bytes())["encoding"]
        table_desc = {}
        try:
            df = pd.read_csv(csv_file, encoding=encoding_type, index_col=False)
        except Exception:
            continue
        for _, row in df.iterrows():
            if pd.isna(row.get("original_column_name")):
                continue
            col_lower = row["original_column_name"].strip().lower()
            expanded = row["column_name"].strip() if pd.notna(row.get("column_name")) else ""
            col_desc = _normalize_description_string(row["column_description"]) if pd.notna(row.get("column_description")) else ""
            data_format = row["data_format"].strip() if pd.notna(row.get("data_format")) else ""
            val_desc = _normalize_description_string(row["value_description"]) if pd.notna(row.get("value_description")) else ""
            if val_desc.lower().startswith("not useful"):
                val_desc = val_desc[len("not useful"):].strip()
            table_desc[col_lower] = {
                "original_column_name_lower": col_lower,
                "expanded_column_name": expanded,
                "column_description": col_desc,
                "data_format": data_format,
                "value_description": val_desc,
            }
        database_description[table_name_lower] = table_desc
    return database_description


def _load_database_schema_dict_for_preprocess(db_path: str, allowed_tables: Optional[set] = None) -> Dict[str, Any]:
    """
    Load database schema as a plain dict.

    Args:
        db_path: Path to the SQLite database file.
        allowed_tables: If provided, only load schema for these tables.
    """
    db_id = Path(db_path).stem
    db_description = _load_database_description(db_path)
    schema = {"db_id": db_id, "db_path": str(db_path), "tables": {}}

    all_table_names = _load_table_names(db_path)
    if allowed_tables is not None:
        skipped = [t for t in all_table_names if t not in allowed_tables]
        all_table_names = [t for t in all_table_names if t in allowed_tables]
        if skipped:
            logger.info(
                f"[{db_id}] table_filter active: keeping {len(all_table_names)} tables, "
                f"skipping {len(skipped)}"
            )

    for table_name in all_table_names:
        table_dict = {"table_name": table_name, "columns": {}}
        primary_keys = _load_primary_keys(db_path, table_name)
        foreign_keys = _load_foreign_keys(db_path, table_name)

        for col_name, col_type in _load_column_names_and_types(db_path, table_name):
            col = {
                "column_name": col_name,
                "column_type": col_type,
                "primary_key": col_name.lower() in [pk.lower() for pk in primary_keys],
                "foreign_keys": [],
                "description": "",
                "value_examples": [],
                "value_statistics": {"total_count": 0, "distinct_count": 0, "null_count": 0},
            }

            for src_table, src_col, tgt_table, tgt_col in foreign_keys:
                if src_col.lower() == col_name.lower():
                    col["foreign_keys"].append((tgt_table, tgt_col))

            desc_parts = []
            tbl_desc = db_description.get(table_name.lower(), {})
            col_desc_info = tbl_desc.get(col_name.lower(), {})
            if col_desc_info.get("expanded_column_name", ""):
                desc_parts.append(f"Expanded Column Name: {col_desc_info['expanded_column_name']}")
            if col_desc_info.get("column_description", ""):
                desc_parts.append(f"Column Description: {col_desc_info['column_description']}")
            if col_desc_info.get("value_description", ""):
                desc_parts.append(f"Value Description: {col_desc_info['value_description']}")
            col["description"] = " | ".join(desc_parts) if desc_parts else ""

            if col_type.upper() != "BLOB":
                col["value_examples"] = _load_value_examples(db_path, table_name, col_name)
            
            col["value_statistics"] = _load_value_statistics(db_path, table_name, col_name)

            table_dict["columns"][col_name] = col
        schema["tables"][table_name] = table_dict

    for tbl_name, tbl_dict in schema["tables"].items():
        for col_name, col_dict in tbl_dict["columns"].items():
            valid_fks = []
            for tgt_table, tgt_col in col_dict["foreign_keys"]:
                if tgt_table in schema["tables"] and tgt_col in schema["tables"][tgt_table]["columns"]:
                    valid_fks.append((tgt_table, tgt_col))
            col_dict["foreign_keys"] = valid_fks

    return schema
