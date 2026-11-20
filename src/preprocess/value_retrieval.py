"""Retrieve database values using question keywords and vector search."""

import copy
import json
import re
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, TypeVar, Tuple

from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection
from loguru import logger
from tqdm import tqdm
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
from openai import OpenAI, RateLimitError, APITimeoutError

from src.preprocess.config import PreprocessConfigManager, LLMConfig, VectorDatabaseConfig, ValueRetrievalConfig
from src.preprocess.dataset import BaseDataset, DataItem, load_dataset, save_dataset
from src.preprocess.vector_db import get_embedding_function



KEYWORDS_EXTRACTION_PROMPT = """
Objective: Analyze the given question and hint to identify and extract keywords, keyphrases, and named entities. These elements are crucial for understanding the core components of the inquiry and the guidance provided. This process involves recognizing and isolating significant terms and phrases that could be instrumental in formulating searches or queries related to the posed question.

Instructions:
Read the Question Carefully: Understand the primary focus and specific details of the question. Look for any named entities (such as organizations, locations, etc.), technical terms, and other phrases that encapsulate important aspects of the inquiry.

Analyze the Hint: The hint is designed to direct attention toward certain elements relevant to answering the question. Extract any keywords, phrases, or named entities that could provide further clarity or direction in formulating an answer.

List Keyphrases and Entities: Combine your findings from both the question and the hint into a single Python list. This list should contain:
- Keywords: Single words that capture essential aspects of the question or hint.
- Keyphrases: Short phrases or named entities that represent specific concepts, locations, organizations, or other significant details.

Ensure to maintain the original phrasing or terminology used in the question and hint.

Example 1:
Question: What is the annual revenue of Acme Corp in the United States for 2022?
Hint: Focus on financial reports and U.S. market performance for the fiscal year 2022.

<result>
["annual revenue", "Acme Corp", "United States", "2022", "financial reports", "U.S. market performance", "fiscal year"]
</result>

Example 2:
Question: In the Winter and Summer Olympics of 1988, which game has the most number of competitors? Find the difference of the number of competitors between the two games.
Hint: the most number of competitors refer to MAX(COUNT(person_id)); SUBTRACT(COUNT(person_id where games_name = '1988 Summer'), COUNT(person_id where games_name = '1988 Winter'));

<result>
["Winter Olympics", "Summer Olympics", "1988", "1988 Summer", "Summer", "1988 Winter", "Winter"]
</result>

Example 3:
Question: How many Men's 200 Metres Freestyle events did Ian James Thorpe compete in?
Hint: Men's 200 Metres Freestyle events refer to event_name = 'Swimming Men''s 200 metres Freestyle'; events compete in refers to event_id;

<result>
["Swimming Men's 200 metres Freestyle", "Ian James Thorpe", "Ian", "James", "Thorpe"]
</result>

Task:
Given the following question and hint, identify and list all relevant keywords, keyphrases, and named entities.

Question: {QUESTION}
Hint: {HINT}

Output only the following XML format (no explanations), ensure the list is a valid Python list:

<result>
[your_keywords_python_list]
</result>
"""



class _EmptyResponseError(Exception):
    """Custom exception for empty LLM responses, used to trigger retry."""
    pass


class _LLM:
    """Lightweight LLM wrapper for keyword extraction."""

    def __init__(self, llm_config: LLMConfig):
        self._config = llm_config
        if llm_config.api_type == "openai":
            self._client = OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)
        else:
            raise ValueError(f"Unsupported api_type for preprocessing LLM: {llm_config.api_type}")

    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(15),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, _EmptyResponseError)),
    )
    def ask(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[List[Any], Dict[str, int]]:
        request_params = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
            "timeout": 300,
        }
        if self._config.reasoning_effort is not None:
            request_params["reasoning_effort"] = self._config.reasoning_effort
        request_params.update(kwargs)

        response = self._client.chat.completions.create(**request_params)
        if not response.choices:
            raise _EmptyResponseError("No response from the model")
        
        for choice in response.choices:
            if choice.message.content is None or choice.message.content.strip() == "":
                raise _EmptyResponseError("Model returned empty content")

        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        return [choice.message for choice in response.choices], token_usage



def _parse_keywords_response(response: str) -> Optional[List[str]]:
    """Parse keywords from LLM response XML."""
    try:
        raw_list = re.search(r"<result>(.*?)</result>", response, re.DOTALL).group(1)
        keywords_list = json.loads(raw_list)
        if isinstance(keywords_list, list):
            return keywords_list
        return None
    except Exception:
        return None


def extract_keywords(question: str, evidence: str, llm: _LLM, max_retry: int = 10, fix_end_token: bool = False) -> Tuple[List[str], Dict[str, int]]:
    """Extract keywords from question + evidence using the LLM."""
    prompt = KEYWORDS_EXTRACTION_PROMPT.format(QUESTION=question, HINT=evidence)
    total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    keywords_list = None

    for _ in range(max_retry):
        try:
            responses, token_usage = llm.ask([{"role": "user", "content": prompt}], n=1)
            for k in total_token_usage:
                total_token_usage[k] += token_usage.get(k, 0)

            content = responses[0].content.strip()
            if fix_end_token and not content.endswith("</result>"):
                content += "</result>"

            parsed = _parse_keywords_response(content)
            if parsed is not None:
                keywords_list = parsed
                break
        except Exception as e:
            logger.warning(f"Keyword extraction attempt failed: {e}")

    if keywords_list is None:
        logger.warning("Keyword extraction failed; using words from the input.")
        keywords_list = question.split(" ") + evidence.split(" ")

    # Include individual words from multi-word keywords.
    processed = set()
    for kw in keywords_list:
        kw = kw.strip()
        if kw:
            processed.add(kw)
            processed.update(kw.split(" "))
    keywords_list = [k for k in processed if k]

    return keywords_list, total_token_usage



@retry(
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
)
def embed_keywords(keywords: List[str], embedding_function: Any, batch_size: int = 1024) -> List[List[float]]:
    """Embed a list of keywords using the embedding function."""
    if not keywords:
        return []
    all_embeddings = []
    for i in range(0, len(keywords), batch_size):
        batch = keywords[i:i + batch_size]
        batch_embeddings = embedding_function(batch)
        all_embeddings.extend(batch_embeddings)
    return all_embeddings



def retrieve_values_for_one_column(
    query_embeddings: List[List[float]],
    collection: Collection,
    table_name: str,
    column_name: str,
    n_results: int,
    lower_meta_data: bool,
) -> Dict[str, Any]:
    """Query ChromaDB for a single column and return top-k values."""
    t_name = table_name.lower() if lower_meta_data else table_name
    c_name = column_name.lower() if lower_meta_data else column_name

    query_results = collection.query(
        query_embeddings=query_embeddings,
        where={"$and": [{"table_name": {"$eq": t_name}}, {"column_name": {"$eq": c_name}}]},
        n_results=n_results,
    )

    values = []
    for documents, distances in zip(query_results["documents"], query_results["distances"]):
        for doc, dist in zip(documents, distances):
            values.append((doc, dist))

    seen = set()
    top_k = []
    for value, distance in sorted(values, key=lambda x: x[1]):
        if value not in seen:
            seen.add(value)
            top_k.append({"value": value, "distance": distance})
            if len(top_k) >= n_results:
                break

    return {"table_name": t_name, "column_name": c_name, "values": top_k}



def _map_lower_table_name(table_name: str, schema: Dict[str, Any]) -> Optional[str]:
    for tbl in schema["tables"].values():
        if tbl["table_name"].lower() == table_name.lower():
            return tbl["table_name"]
    return None


def _map_lower_column_name(table_name: str, column_name: str, schema: Dict[str, Any]) -> Optional[str]:
    for tbl in schema["tables"].values():
        if tbl["table_name"].lower() == table_name.lower():
            for col in tbl["columns"].values():
                if col["column_name"].lower() == column_name.lower():
                    return col["column_name"]
    return None



class ValueRetrievalRunner:
    """
    Runs the value retrieval pipeline:
    - Loads the preprocessed dataset
    - For each item: extract keywords → embed → query vector DB → update schema
    - Saves the result incrementally
    """

    def __init__(self, config_manager: PreprocessConfigManager):
        self._config = config_manager
        self._vr_config = config_manager.value_retrieval_config
        self._vdb_config = config_manager.vector_database_config

        self._llm = _LLM(self._vr_config.llm)

        vr_save_path = self._vr_config.save_path
        dataset_save_path = config_manager.dataset_config.save_path
        if Path(vr_save_path).exists():
            logger.info(f"Resuming value retrieval from checkpoint: {vr_save_path}")
            self._dataset = load_dataset(vr_save_path)
        else:
            logger.info(f"Loading dataset from {dataset_save_path}")
            self._dataset = load_dataset(dataset_save_path)

        self._embedding_function = get_embedding_function(
            model_name_or_path=self._vdb_config.embedding_model_name_or_path,
            api_type=self._vdb_config.api_type,
            use_qwen3_embedding=self._vdb_config.use_qwen3_embedding,
            local_files_only=self._vdb_config.local_files_only,
            normalize_embeddings=self._vdb_config.normalize_embeddings,
            base_url=self._vdb_config.base_url,
            api_key=self._vdb_config.api_key,
        )

        self._collection_cache: Dict[str, Collection] = {}
        self._client_cache: Dict[str, PersistentClient] = {}
        self._db_lock = threading.Lock()

        self._thread_pool = ThreadPoolExecutor(max_workers=self._vr_config.n_parallel)

    def _get_collection(self, db_id: str) -> Collection:
        with self._db_lock:
            if db_id not in self._collection_cache:
                vdb_path = str(Path(self._vdb_config.store_root_path) / db_id)
                client = PersistentClient(path=vdb_path)
                self._client_cache[db_id] = client
                self._collection_cache[db_id] = client.get_collection(
                    name=db_id,
                    embedding_function=self._embedding_function,
                )
            return self._collection_cache[db_id]

    def _retrieve_for_item(self, data_item: DataItem):
        """Process one data item end-to-end."""
        start = time.time()

        keywords, token_usage = extract_keywords(
            data_item.question, data_item.evidence, self._llm,
            max_retry=self._vr_config.max_retry,
            fix_end_token=self._vr_config.llm.fix_end_token,
        )
        data_item.question_keywords = keywords
        data_item.value_retrieval_llm_cost = token_usage

        query_embeddings = embed_keywords(keywords, self._embedding_function, self._vdb_config.batch_size)

        collection = self._get_collection(data_item.database_id)
        data_item.retrieved_values = defaultdict(dict)

        column_tasks = []
        for tbl_name, tbl_dict in data_item.database_schema["tables"].items():
            for col_name, col_dict in tbl_dict["columns"].items():
                ct = col_dict["column_type"].upper()
                if ct == "TEXT" or ct.startswith("VARCHAR") or ct.startswith("CHAR"):
                    column_tasks.append((tbl_name, col_name))

        if column_tasks:
            with ThreadPoolExecutor(max_workers=min(len(column_tasks), 8)) as col_executor:
                future_map = {
                    col_executor.submit(
                        retrieve_values_for_one_column,
                        query_embeddings, collection, t, c,
                        self._vr_config.n_results, self._vdb_config.lower_meta_data,
                    ): (t, c)
                    for t, c in column_tasks
                }
                for future in as_completed(future_map):
                    result = future.result()
                    orig_tbl = _map_lower_table_name(result["table_name"], data_item.database_schema)
                    orig_col = _map_lower_column_name(result["table_name"], result["column_name"], data_item.database_schema)
                    if orig_tbl and orig_col:
                        data_item.retrieved_values[orig_tbl][orig_col] = result["values"]

        data_item.retrieved_values = dict(data_item.retrieved_values)

        schema_after = copy.deepcopy(data_item.database_schema)
        for tbl_name, col_dict in data_item.retrieved_values.items():
            for col_name, values in col_dict.items():
                original_examples = data_item.database_schema["tables"][tbl_name]["columns"][col_name]["value_examples"]
                new_examples = [v["value"] for v in values] + original_examples
                new_examples = new_examples[:self._vr_config.n_results]
                schema_after["tables"][tbl_name]["columns"][col_name]["value_examples"] = new_examples
        data_item.database_schema_after_value_retrieval = schema_after

        data_item.value_retrieval_time = time.time() - start
        data_item.total_time = (data_item.total_time or 0) + data_item.value_retrieval_time
        if data_item.total_llm_cost is None:
            data_item.total_llm_cost = dict(data_item.value_retrieval_llm_cost)
        else:
            for k, v in data_item.value_retrieval_llm_cost.items():
                data_item.total_llm_cost[k] = data_item.total_llm_cost.get(k, 0) + v

    def save_result(self):
        save_dataset(self._dataset, self._vr_config.save_path)

    def run(self):
        """Run value retrieval for all items, saving incrementally."""
        futures = []
        for data_item in self._dataset:
            if data_item.database_schema_after_value_retrieval is not None:
                continue
            future = self._thread_pool.submit(self._retrieve_for_item, data_item)
            futures.append(future)

        for idx, future in tqdm(enumerate(as_completed(futures), start=1), total=len(futures), desc="Value Retrieval"):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing data item: {e}")
                import traceback
                logger.error(traceback.format_exc())

            if idx % 5 == 0:
                self.save_result()

        self.save_result()
        self._cleanup()

    def _cleanup(self):
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None
        self._client_cache.clear()
        self._collection_cache.clear()
