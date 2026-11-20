"""
Vector database creation using ChromaDB.

Indexes all TEXT/VARCHAR/CHAR column values from each database into a
ChromaDB persistent collection for later value retrieval.
"""

import re
import uuid
import shutil
from pathlib import Path
from typing import List, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction, OpenAIEmbeddingFunction
from loguru import logger
from tqdm import tqdm

from src.preprocess.config import VectorDatabaseConfig
from src.preprocess.dataset import _execute_sql_readonly, _load_table_names, _load_column_names_and_types



import numpy as np
from chromadb.api.types import Documents, Embeddings


class QwenEmbeddingFunction(SentenceTransformerEmbeddingFunction):
    """
    Wrapper of SentenceTransformerEmbeddingFunction for Qwen embedding models.
    Uses the 'query' prompt for encoding.
    """

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self._model.encode(
            list(input),
            prompt_name="query",
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return [np.array(embedding, dtype=np.float32) for embedding in embeddings]



UUID_PATTERN = re.compile(
    r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
)
NUMBER_PATTERN = re.compile(r'^[0-9]+\.?[0-9]*$')


def _is_uuid_column(values: List[str]) -> bool:
    return all(UUID_PATTERN.match(v) for v in values)


def _is_number_column(values: List[str]) -> bool:
    return all(NUMBER_PATTERN.match(v) for v in values)

import threading

_embedding_lock = threading.Lock()

def get_embedding_function(
    model_name_or_path: str,
    api_type: str = "local",
    use_qwen3_embedding: bool = False,
    local_files_only: bool = False,
    normalize_embeddings: bool = False,
    base_url: str = None,
    api_key: str = None,
):
    """Create the appropriate ChromaDB embedding function."""
    if api_type == "local":
        with _embedding_lock:
            if use_qwen3_embedding:
                logger.info(f"Using Qwen3 embedding: {model_name_or_path}")
                return QwenEmbeddingFunction(
                    model_name=model_name_or_path,
                    device="cuda",
                    trust_remote_code=True,
                    local_files_only=local_files_only,
                    normalize_embeddings=normalize_embeddings,
                )
            else:
                logger.info(f"Using SentenceTransformer embedding: {model_name_or_path}")
                return SentenceTransformerEmbeddingFunction(
                    model_name=model_name_or_path,
                    device="cuda",
                    trust_remote_code=True,
                    local_files_only=local_files_only,
                    normalize_embeddings=normalize_embeddings,
                )
    elif api_type == "openai":
        logger.info(f"Using OpenAI embedding: {model_name_or_path}")
        return OpenAIEmbeddingFunction(
            model_name=model_name_or_path,
            api_base=base_url,
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unsupported embedding api_type: {api_type}")



def _process_one_column(
    db_path: str,
    table_name: str,
    column_name: str,
    column_type: str,
    max_value_length: int,
    batch_size: int,
    lower_meta_data: bool,
    collection: Any,
    db_id: str,
):
    """Index all distinct values of a single TEXT column into the ChromaDB collection."""
    upper_type = column_type.upper()
    if upper_type != "TEXT" and not upper_type.startswith("VARCHAR") and not upper_type.startswith("CHAR"):
        return

    sql = f"""
    SELECT DISTINCT `{column_name}` FROM `{table_name}`
    WHERE `{column_name}` IS NOT NULL
    AND LENGTH(CAST(`{column_name}` AS TEXT)) <= {max_value_length};
    """
    rows = _execute_sql_readonly(db_path, sql)
    value_examples = [str(row[0]) for row in rows]

    if not value_examples:
        return
    if _is_uuid_column(value_examples) or _is_number_column(value_examples):
        return

    for i in range(0, len(value_examples), batch_size):
        batch = value_examples[i:i + batch_size]
        meta = (
            {"db_id": db_id.lower(), "table_name": table_name.lower(), "column_name": column_name.lower()}
            if lower_meta_data
            else {"db_id": db_id, "table_name": table_name, "column_name": column_name}
        )
        collection.add(
            ids=[str(uuid.uuid4()) for _ in range(len(batch))],
            documents=batch,
            metadatas=[meta for _ in range(len(batch))],
        )



def make_vector_db(
    db_path: str,
    vector_db_path: str,
    max_value_length: int = 100,
    batch_size: int = 1024,
    n_parallel: int = 1,
    lower_meta_data: bool = True,
    embedding_function=None,
    allowed_tables: "Optional[set]" = None,
) -> bool:
    """
    Create a ChromaDB vector database for one SQLite database.
    Returns True on success, False on failure.

    Args:
        allowed_tables: If provided, only index columns from these tables.
    """
    vector_db_path = str(vector_db_path)
    if Path(vector_db_path).exists():
        shutil.rmtree(vector_db_path)
        logger.info(f"Cleaned existing vector DB at {vector_db_path}")

    db_id = Path(db_path).stem
    logger.info(f"Making vector DB for {db_id} → {vector_db_path}")

    client = PersistentClient(path=vector_db_path)
    collection = client.create_collection(
        name=db_id,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    all_table_names = _load_table_names(db_path)
    if allowed_tables is not None:
        skipped = [t for t in all_table_names if t not in allowed_tables]
        all_table_names = [t for t in all_table_names if t in allowed_tables]
        if skipped:
            logger.info(
                f"[{db_id}] table_filter active: indexing {len(all_table_names)} tables, "
                f"skipping {len(skipped)}"
            )

    all_tasks = []
    for table_name in all_table_names:
        for col_name, col_type in _load_column_names_and_types(db_path, table_name):
            all_tasks.append((table_name, col_name, col_type))

    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = [
            executor.submit(
                _process_one_column,
                db_path, tbl, col, ctype,
                max_value_length, batch_size, lower_meta_data, collection, db_id,
            )
            for tbl, col, ctype in all_tasks
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Vector DB: {db_id}"):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Failed to process column: {e}")
                if Path(vector_db_path).exists():
                    shutil.rmtree(vector_db_path)
                return False

    return True



def make_vector_db_for_db_path(
    db_path: str,
    vector_database_config: VectorDatabaseConfig,
    allowed_tables: Optional[Set[str]] = None,
) -> bool:
    """Create vector DB for a single database, skipping if already done.

    Args:
        allowed_tables: If provided, only index columns belonging to these
            tables.  Passed through to :func:`make_vector_db`.
    """
    db_id = Path(db_path).stem
    success_flag = Path(vector_database_config.store_root_path) / db_id / "success_flag"

    if success_flag.exists():
        logger.info(f"Vector DB for {db_id} already exists (success_flag found), skipping.")
        return True

    try:
        embedding_function = get_embedding_function(
            model_name_or_path=vector_database_config.embedding_model_name_or_path,
            api_type=vector_database_config.api_type,
            use_qwen3_embedding=vector_database_config.use_qwen3_embedding,
            local_files_only=vector_database_config.local_files_only,
            normalize_embeddings=vector_database_config.normalize_embeddings,
            base_url=vector_database_config.base_url,
            api_key=vector_database_config.api_key,
        )
        success = make_vector_db(
            db_path=db_path,
            vector_db_path=str(Path(vector_database_config.store_root_path) / db_id),
            max_value_length=vector_database_config.max_value_length,
            batch_size=vector_database_config.batch_size,
            n_parallel=vector_database_config.n_parallel,
            lower_meta_data=vector_database_config.lower_meta_data,
            embedding_function=embedding_function,
            allowed_tables=allowed_tables,
        )
        if success:
            success_flag.parent.mkdir(parents=True, exist_ok=True)
            success_flag.touch()
            logger.info(f"Successfully made vector DB for {db_id}")
        else:
            logger.error(f"Failed to make vector DB for {db_id}")
        return success
    except Exception as e:
        logger.error(f"Failed to make vector DB for {db_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        vdb_path = Path(vector_database_config.store_root_path) / db_id
        if vdb_path.exists():
            shutil.rmtree(vdb_path)
        return False
