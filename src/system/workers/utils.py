import re
import random
import copy
import time
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter
from loguru import logger
from src.system.config import config
from src.system.prompt_factory import get_br_pair_selection_prompt
from src.system.llm_client import LLMResponse
from src.system.types import WorkerType, ReasoningStep, SQLCandidate, ReasoningSession, WorkerResult, WorkerInput
from src.database.schema import TableSchema
from src.database.utils import build_table_profile_statement
from src.database.sql_execution import (
    cached_execute_sql_with_timeout,
    execute_sql_with_timeout,
    normalize_sql,
    format_execution_result,
    is_valid_execution_result,
    has_non_empty_result,
    SQLExecutionResultType
)

SQL_GENERATION_LLM_KWARGS_TEMPERATURE = 0.7
SQL_REVISION_LLM_KWARGS_TEMPERATURE = 0.7

SCHEMA_PRUNER_LLM_KWARGS_N = 8
SQL_GENERATION_LLM_KWARGS_N = 36
SQL_REVISION_LLM_KWARGS_N = 5

SQL_VALIDATION_MAX_TRIES = 36

VALID_SQL_THRESHOLD = 0.8      # Valid SQL proportion
CONSISTENCY_THRESHOLD = 0.8    # Consistency confidence

CONFIDENCE_THRESHOLD_UNRESTRICTED = 0.5


def get_icl_examples_path() -> Optional[str]:
    """Derive the ICL examples file path from config's dataset and split fields.

    If ``system.data.icl_examples_path`` is set in the config, that value is used directly.  
    Otherwise the path is derived by convention: data/{dataset}/{split}/{dataset}_{split}_few_shots.json

    Returns:
        The path string if the file exists, otherwise None.
    """
    from pathlib import Path

    data_cfg = config.config_data.get("system", {}).get("data", {})
    dataset = data_cfg.get("dataset")
    split = data_cfg.get("split")

    if not dataset or not split:
        return None

    path = Path(f"data/{dataset}/{split}/{dataset}_{split}_few_shots.json")
    if path.exists():
        return str(path)

    return None


def _extract_sql_query_answer(response: str) -> Optional[str]:
    try:
        # Match either <result> or <sql> tags
        match = re.search(r"<(?:result|sql)>(.*?)</(?:result|sql)>", response, flags=re.DOTALL)
        if match:
            sql_query = match.group(1).strip()
            return normalize_sql(sql_query)
        return None
    except Exception as e:
        logger.error(f"Error parsing sql response: {e}")
        return None


def update_session_state(session: ReasoningSession, worker_type: WorkerType, result: WorkerResult):
    """Update the session state using the worker result."""
    if worker_type == WorkerType.SQL_CONSTRUCTOR and (
        result.success or result.sql_candidates or _extract_sql_query_answer(result.result)
    ):
        # Regeneration clears cached SQL candidates and refinements.
        session.constructed_sql = session.refined_sql = None
        session.sql_candidates = session.refined_candidates = None
        session.sql_consistency_score = None
    if worker_type == WorkerType.SQL_CONSTRUCTOR and result.sql_candidates:
        session.sql_candidates = result.sql_candidates
    elif worker_type == WorkerType.SQL_REFINER and result.sql_candidates:
        session.refined_candidates = result.sql_candidates

    if not result.success:
        if worker_type == WorkerType.SQL_CONSTRUCTOR and result.result:
            extracted_sql = _extract_sql_query_answer(result.result)
            if extracted_sql:
                session.constructed_sql = extracted_sql
        elif worker_type == WorkerType.SQL_REFINER and result.result:
            extracted_sql = _extract_sql_query_answer(result.result)
            if extracted_sql:
                session.refined_sql = extracted_sql
        return

    if worker_type == WorkerType.QUESTION_DECOMPOSER:
        session.rephrased_question = result.result
    elif worker_type == WorkerType.SCHEMA_PRUNER:
        session.selected_schema_context = result.result
    elif worker_type == WorkerType.VALUE_EXTRACTOR:
        session.identified_column_values = result.result
    elif worker_type == WorkerType.OPERATION_IDENTIFIER:
        session.identified_column_functions = result.result
    elif worker_type == WorkerType.SQL_CONSTRUCTOR:
        session.constructed_sql = _extract_sql_query_answer(result.result)
        if result.metadata and "confidence" in result.metadata:
            session.sql_consistency_score = result.metadata["confidence"]
    elif worker_type == WorkerType.SQL_REFINER:
        session.refined_sql = _extract_sql_query_answer(result.result)
        if result.metadata and "confidence" in result.metadata:
            session.sql_consistency_score = result.metadata["confidence"]


def build_worker_input(session: ReasoningSession, table_schema_dict: Dict[str, TableSchema], steps: List[ReasoningStep]) -> WorkerInput:
    """Build worker input from the current session state."""
    return WorkerInput(
        question=session.current_question,
        schema=session.current_schema,
        context={
            "table_schema_dict": table_schema_dict,
            "db_path": session.db_path,
            "session": session,
            "steps": steps,
            "question_id": session.question_id,
            "retrieved_values": session.retrieved_values
        }
    )


def _build_hint_from_steps(steps: List[ReasoningStep], context: Dict[str, Any]) -> str:
    previous_thoughts = []
    
    session = context.get("session")
    evidence = session.evidence if session else None

    hint_parts = []
    if evidence:
        hint_parts.append(f"{evidence}")

    for step in steps:
        if not (step.worker_result and step.worker_result.success):
            continue

        worker_type = step.worker_result.worker_type
        result = step.worker_result.result

        if worker_type == WorkerType.VALUE_EXTRACTOR:
            previous_thoughts.append(f"Identified column values: {result}\n")
        elif worker_type == WorkerType.OPERATION_IDENTIFIER:
            previous_thoughts.append(f"Identified column functions: {result}\n")
        elif worker_type in (WorkerType.SQL_CONSTRUCTOR, WorkerType.SQL_REFINER):
            db_path = context.get("db_path")
            sql_query = _extract_sql_query_answer(result)
            
            candidates_info = ""
            if hasattr(step.worker_result, "sql_candidates") and step.worker_result.sql_candidates and db_path:
                valid_cands = [c for c in step.worker_result.sql_candidates if c.is_valid]
                if valid_cands:
                    cand_groups = defaultdict(list)
                    for c in valid_cands:
                        res = cached_execute_sql_with_timeout(db_path, c.sql)
                        if is_valid_execution_result(res):
                            cand_groups[frozenset(tuple(row) for row in res.result)].append((c.sql, res))
                    
                    if cand_groups:
                        sorted_groups = sorted(cand_groups.values(), key=len, reverse=True)
                        top_groups = []
                        for grp in sorted_groups:
                            if any(c_sql == sql_query for c_sql, _ in grp):
                                continue
                            top_groups.append(grp)
                            
                        if top_groups:
                            candidates_info += "\nAlternative SQL Candidates (grouped by execution result):\n"
                            for i, group in enumerate(top_groups[:2], 1):
                                rep_sql, rep_res = group[0]
                                res_str = format_execution_result(rep_res, row_limit=2)
                                candidates_info += f"Alternative {i} ({len(group)} votes):\nSQL: {rep_sql}\nResult:\n{res_str}\n"

            worker_label = "Generated SQL" if worker_type == WorkerType.SQL_CONSTRUCTOR else "Refined SQL"
            if sql_query and db_path:
                sql_execution_result = cached_execute_sql_with_timeout(db_path, sql_query)
                sql_execution_result_str = format_execution_result(sql_execution_result)
                previous_thoughts.append(f"Selected Best {worker_label}: {sql_query}\n\nSelected SQL execution result:\n{sql_execution_result_str}\n{candidates_info}")
            else:
                previous_thoughts.append(f"Selected Best {worker_label} raw response: {result}\n{candidates_info}")

    if not previous_thoughts and not hint_parts:
        return ""
    
    if previous_thoughts:
        hint_parts.append("Here are my previous thoughts:\n" + "\n".join(previous_thoughts))
    
    return "\n\n".join(hint_parts)


def _build_and_condense_schema_context(
    table_schema_dict: Dict[str, TableSchema],
    max_examples_per_column: int = 3
) -> str:
    """Build schema context, reducing detail to fit the configured length limit."""
    system_config = config.get_system_config()
    max_length = system_config.get("schema_max_length", 20000)
    
    def build_context(
        examples_per_column: int,
        add_value_description: bool = True,
        add_column_description: bool = True,
        add_expanded_column_name: bool = True,
    ) -> str:
        schema_context_ddl_list = []
        for table_name, table_schema in table_schema_dict.items():
            temp_schema_dict = table_schema.to_dict()
            if examples_per_column < max_examples_per_column:
                for col_name, col_data in temp_schema_dict["columns"].items():
                    if "value_examples" in col_data:
                        col_data["value_examples"] = col_data["value_examples"][:examples_per_column]
            
            schema_context_ddl_list.append(build_table_profile_statement(
                temp_schema_dict,
                add_value_description=add_value_description,
                add_column_description=add_column_description,
                add_value_examples=examples_per_column > 0,
                add_expanded_column_name=add_expanded_column_name
            ))
        return "\n".join(schema_context_ddl_list)

    condensation_levels = [
        ("full (examples=3)", dict(examples_per_column=max_examples_per_column)),
        ("reduce examples to 2", dict(examples_per_column=2)),
        ("reduce examples to 1", dict(examples_per_column=1)),
        ("remove value examples", dict(examples_per_column=0)),
        ("remove value descriptions", dict(examples_per_column=0, add_value_description=False)),
        ("remove column descriptions", dict(examples_per_column=0, add_value_description=False, add_column_description=False)),
        ("bare DDL (remove expanded names)", dict(examples_per_column=0, add_value_description=False, add_column_description=False, add_expanded_column_name=False)),
    ]

    schema_context = None
    for level_idx, (label, kwargs) in enumerate(condensation_levels):
        schema_context = build_context(**kwargs)
        if len(schema_context) <= max_length:
            if level_idx > 0:
                logger.info(
                    f"Schema context length: {len(schema_context)} (limit: {max_length})."
                )
            break

    if len(schema_context) > max_length:
        logger.warning(
            f"Schema context remains over the length limit ({len(schema_context)}/{max_length})."
        )

    return schema_context


def merge_schema_linking_results(results: List[Dict[str, List[str]]]) -> Dict[str, List[str]]:
    """
    Merges multiple schema linking results (tables and columns selections).
    """
    merged_result = {}
    for result in results:
        for table_name, columns in result.items():
            if table_name not in merged_result:
                merged_result[table_name] = set()
            merged_result[table_name].update(columns)
    return {table_name: list(columns) for table_name, columns in merged_result.items()}


def select_fastest_sql_in_sc_candidates(
    sql_candidates: List[Tuple[str, str, Optional[str]]],
    db_path: str,
    question_id: Optional[str] = None,
    worker_type: Optional[str] = None,
    strategy_stats: Optional[Dict[str, Dict[str, int]]] = None,
    log_prefix: str = ""
) -> Tuple[Optional[str], Optional[str], float, bool, Dict[str, Any]]:
    """
    Execute a list of SQL candidates and select the most consistent group and the fastest SQL within the group.
    
    Args:
        sql_candidates: List of tuples (sql, raw_content, metadata/strategy_name)
        db_path: Path to the database file
        question_id: Optional question ID for logging
        worker_type: Optional worker type for logging
        strategy_stats: Optional strategy statistics dict for detailed logging
        log_prefix: Optional prefix for log messages
    
    Returns:
        Tuple of (best_sql, best_raw_content, confidence_score, success, stats_dict)
        - best_sql: The selected SQL query string (None if no valid SQL found)
        - best_raw_content: The raw LLM response that generated best_sql (None if not found)
        - confidence_score: Self-consistency confidence score (0.0 to 1.0)
        - success: Whether a valid SQL was found and selected
        - stats_dict: Dictionary containing execution statistics
    """
    if not sql_candidates:
        return None, None, 0.0, False, {
            "executed": 0,
            "valid": 0,
            "fail": 0,
            "timeout": 0,
            "empty": 0,
            "parse_fail": 0
        }
    
    result_groups = defaultdict(list)
    response_map = {}
    execution_time_map = {}
    valid_sqls = []
    non_empty_keys = set()
    
    executed_count = 0
    fail_count = 0
    timeout_count = 0
    empty_result_count = 0
    
    strategy_execution_stats = {}
    if strategy_stats:
        for strategy_name in strategy_stats.keys():
            strategy_execution_stats[strategy_name] = {
                "executed": 0,
                "valid": 0,
                "fail": 0,
                "timeout": 0,
                "empty": 0
            }
    
    from src.database.sql_execution import normalize_sql

    normalized_sql_map = {}
    
    for sql, raw_content, metadata in sql_candidates:
        response_map[sql] = raw_content
        norm_sql = normalize_sql(sql)
        
        if norm_sql not in normalized_sql_map:
            normalized_sql_map[norm_sql] = {
                "first_sql": sql,
                "exec_result": None,
                "exec_time": None,
                "occurrences": []
            }
        
        normalized_sql_map[norm_sql]["occurrences"].append((sql, metadata))
        
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max(len(normalized_sql_map), 1), 32)) as executor:
        future_to_sql = {}
        for norm_sql, data in normalized_sql_map.items():
            def exec_sql(s):
                start_t = time.time()
                exc_res = execute_sql_with_timeout(db_path, s)
                return exc_res, time.time() - start_t
            
            future = executor.submit(exec_sql, data["first_sql"])
            future_to_sql[future] = norm_sql
            
        for future in concurrent.futures.as_completed(future_to_sql):
            norm_sql = future_to_sql[future]
            exc_res, exec_time = future.result()
            normalized_sql_map[norm_sql]["exec_result"] = exc_res
            normalized_sql_map[norm_sql]["exec_time"] = exec_time
        
    for norm_sql, data in normalized_sql_map.items():
        exec_result = data["exec_result"]
        execution_time = data["exec_time"]
        
        for sql, metadata in data["occurrences"]:
            executed_count += 1
            
            if is_valid_execution_result(exec_result):
                execution_time_map[sql] = execution_time
            
            if metadata and metadata in strategy_execution_stats:
                strategy_execution_stats[metadata]["executed"] += 1
            
            if exec_result.result_type == SQLExecutionResultType.ERROR:
                fail_count += 1
                if metadata and metadata in strategy_execution_stats:
                    strategy_execution_stats[metadata]["fail"] += 1
            elif exec_result.result_type == SQLExecutionResultType.TIMEOUT:
                timeout_count += 1
                if metadata and metadata in strategy_execution_stats:
                    strategy_execution_stats[metadata]["timeout"] += 1
            elif is_valid_execution_result(exec_result):
                result_key = frozenset(tuple(row) for row in exec_result.result)
                result_groups[result_key].append(sql)
                
                if not has_non_empty_result(exec_result):
                    empty_result_count += 1
                    if metadata and metadata in strategy_execution_stats:
                        strategy_execution_stats[metadata]["empty"] += 1
                else:
                    non_empty_keys.add(result_key)
                    valid_sqls.append(sql)
                    if metadata and metadata in strategy_execution_stats:
                        strategy_execution_stats[metadata]["valid"] += 1

    parse_fail_count = 0
    if strategy_stats:
        for strategy_name, stats in strategy_stats.items():
            parse_fail_count += stats.get("generated", 0) - stats.get("parsed", 0)
    
    log_msg_parts = []
    if question_id:
        log_msg_parts.append(f"QID: {question_id}")
    if worker_type:
        log_msg_parts.append(f"Worker: {worker_type}")
    log_context = " ".join(log_msg_parts) if log_msg_parts else ""
    
    if strategy_execution_stats and strategy_stats:
        for strategy_name in strategy_execution_stats.keys():
            gen_stats = strategy_stats.get(strategy_name, {})
            exec_stats = strategy_execution_stats[strategy_name]
            logger.debug(
                f"{log_prefix} - [{strategy_name}] Gen: {gen_stats.get('generated', 0)}, "
                f"Parsed: {gen_stats.get('parsed', 0)}, Exec: {exec_stats['executed']}, "
                f"Valid: {exec_stats['valid']}, Fail: {exec_stats['fail']}, "
                f"Timeout: {exec_stats['timeout']}, Empty: {exec_stats['empty']}"
            )
    
    if result_groups:
        candidate_groups_dict = result_groups
        if non_empty_keys:
            candidate_groups_dict = {k: v for k, v in result_groups.items() if k in non_empty_keys}
            
        group_sizes = [len(group) for group in candidate_groups_dict.values()]
        max_group_size = max(group_sizes)
        most_common_group_key = max(candidate_groups_dict, key=lambda k: len(candidate_groups_dict[k]))
        most_common_group = candidate_groups_dict[most_common_group_key]
        candidate_groups = [group for group in candidate_groups_dict.values() if len(group) == max_group_size]
        
        sql_with_execution_time = []
        for sql in most_common_group:
            if sql in execution_time_map:
                execution_time = execution_time_map[sql]
            else:
                try:
                    start_time = time.time()
                    exec_result = execute_sql_with_timeout(db_path, sql)
                    execution_time = time.time() - start_time
                    if is_valid_execution_result(exec_result):
                        execution_time_map[sql] = execution_time
                except Exception as e:
                    logger.warning(f"{log_prefix}Failed to measure execution time for SQL: {e}")
                    execution_time = float('inf')
            sql_with_execution_time.append((sql, execution_time))
        
        if sql_with_execution_time:
            sql_with_execution_time.sort(key=lambda x: x[1])
            best_sql, best_execution_time = sql_with_execution_time[0]
        else:
            best_sql = most_common_group[0]
            best_execution_time = None

        best_raw_content = response_map.get(best_sql)

        num_valid_queries = len(valid_sqls)
        total_valid_groups = sum(len(group) for group in candidate_groups_dict.values())
        consistency = max_group_size / total_valid_groups if total_valid_groups > 0 else 0.0
        raw_confidence = max_group_size / executed_count if executed_count > 0 else 0.0

        best_sql_metadata = None
        for sql, _, metadata in sql_candidates:
            if sql == best_sql:
                best_sql_metadata = metadata
                break

        execution_time_msg = f", Exec time: {best_execution_time:.4f}s" if best_execution_time is not None else ""
        logger.debug(
            f"{log_prefix}[SQL Consistency]{' ' + log_context if log_context else ''} "
            f"Generated: {executed_count}, Valid: {num_valid_queries}, Fail: {fail_count}, "
            f"Timeout: {timeout_count}, Empty: {empty_result_count}, ParseFail: {parse_fail_count}. "
            f"Pick highest SC group: {max_group_size}/{total_valid_groups} "
            f"(consistency={consistency:.2f}, raw_confidence={raw_confidence:.2f}). "
            f"Strategy: {best_sql_metadata or 'unknown'}"
            f"{execution_time_msg}."
        )

        stats = {
            "executed": executed_count,
            "valid": num_valid_queries,
            "fail": fail_count,
            "timeout": timeout_count,
            "empty": empty_result_count,
            "parse_fail": parse_fail_count,
            "max_group_size": max_group_size,
            "candidate_groups": len(candidate_groups),
            "best_execution_time": best_execution_time if best_execution_time is not None else None,
            "raw_confidence": raw_confidence,
            "strategy_execution_stats": strategy_execution_stats if strategy_execution_stats else {}
        }

        return best_sql, best_raw_content, consistency, True, stats
    
    if sql_candidates:
        sql_counts = defaultdict(int)
        sql_metadata_map = {}
        for sql, _, metadata in sql_candidates:
            sql_counts[sql] += 1
            if sql not in sql_metadata_map:
                sql_metadata_map[sql] = metadata
        
        best_sql = max(sql_counts, key=sql_counts.get)
        best_sql_count = sql_counts[best_sql]
        best_raw_content = response_map.get(best_sql)
        best_sql_metadata = sql_metadata_map.get(best_sql, "unknown")
        
        logger.warning(
            f"{log_prefix}[SQL Selection]{' ' + log_context if log_context else ''} "
            f"No valid execution results. Selected the most frequent SQL candidate "
            f"(count: {best_sql_count}, strategy: {best_sql_metadata})."
        )
        
        stats = {
            "executed": executed_count,
            "valid": 0,
            "fail": fail_count,
            "timeout": timeout_count,
            "empty": empty_result_count,
            "parse_fail": parse_fail_count,
            "fallback": "most_frequent_sql"
        }
        
        return best_sql, best_raw_content, 0.0, False, stats
    
    stats = {
        "executed": 0,
        "valid": 0,
        "fail": 0,
        "timeout": 0,
        "empty": 0,
        "parse_fail": parse_fail_count
    }
    return None, None, 0.0, False, stats


def build_sql_candidates(
    sql_tuples: List[Tuple[str, str, Optional[str]]],
    db_path: str,
    log_prefix: str = ""
) -> List[SQLCandidate]:
    """
    Convert internal (sql, raw_content, strategy) tuples to SQLCandidate list.
    Executes each SQL to populate is_valid and execution_time.
    
    Args:
        sql_tuples: List of (sql, raw_content, strategy) tuples
        db_path: Path to the database file
        log_prefix: Optional prefix for log messages
    
    Returns:
        List of SQLCandidate objects with execution metadata
    """
    candidates = []
    
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max(len(sql_tuples), 1), 32)) as executor:
        future_to_tuple = {}
        for sql, raw_content, strategy in sql_tuples:
            def build_candidate(s, r, st):
                start_t = time.time()
                exec_result = execute_sql_with_timeout(db_path, s)
                is_valid = is_valid_execution_result(exec_result)
                return SQLCandidate(
                    sql=s,
                    raw_content=r,
                    strategy=st,
                    is_valid=is_valid,
                    execution_time=(time.time() - start_t) if is_valid_execution_result(exec_result) else None
                )
            future = executor.submit(build_candidate, sql, raw_content, strategy)
            future_to_tuple[future] = (sql, raw_content, strategy)
            
        results_map = {}
        for future in concurrent.futures.as_completed(future_to_tuple):
            tup = future_to_tuple[future]
            results_map[tup] = future.result()
            
        for tup in sql_tuples:
            candidates.append(results_map[tup])
    
    valid_count = sum(1 for c in candidates if c.is_valid)
    logger.debug(
        f"{log_prefix}Built {len(candidates)} SQL candidates "
        f"(valid: {valid_count}, invalid: {len(candidates) - valid_count})"
    )
    
    return candidates


def select_best_from_candidates(
    candidates: List[SQLCandidate],
    db_path: str,
    log_prefix: str = ""
) -> Tuple[Optional[str], float, bool]:
    """
    Perform self-consistency selection from a list of SQLCandidate objects.
    Uses cached execution results from SQLCandidate.is_valid and execution_time.
    
    Selection Strategy:
    1. Group valid SQLs by execution result
    2. Select the group with most consistent results
    3. From that group, pick the SQL with fastest execution time
    
    Args:
        candidates: List of SQLCandidate objects with execution metadata
        db_path: Path to the database file (for result grouping)
        log_prefix: Optional prefix for log messages
    
    Returns:
        Tuple of (best_sql, confidence_score, success)
    """
    if not candidates:
        return None, 0.0, False
    
    valid_candidates = [c for c in candidates if c.is_valid]
    
    if not valid_candidates:
        sql_counts = defaultdict(list)
        for c in candidates:
            sql_counts[c.sql].append(c)
        
        if sql_counts:
            most_frequent_sql = max(sql_counts.keys(), key=lambda s: len(sql_counts[s]))
            logger.warning(
                f"{log_prefix}No valid execution results. "
                f"Selected the most frequent SQL candidate (count: {len(sql_counts[most_frequent_sql])})."
            )
            return most_frequent_sql, 0.0, False
        return None, 0.0, False
    
    result_groups = defaultdict(list)
    non_empty_keys = set()
    for c in valid_candidates:
        exec_result = cached_execute_sql_with_timeout(db_path, c.sql)
        if is_valid_execution_result(exec_result):
            result_key = frozenset(tuple(row) for row in exec_result.result)
            result_groups[result_key].append(c)
            if has_non_empty_result(exec_result):
                non_empty_keys.add(result_key)
    
    if not result_groups:
        best_candidate = min(valid_candidates, key=lambda c: c.execution_time or float('inf'))
        logger.debug(f"{log_prefix}Selecting fastest valid candidate (no grouping possible).")
        return best_candidate.sql, 1.0 / len(valid_candidates), True
        
    candidate_groups_dict = result_groups
    if non_empty_keys:
        candidate_groups_dict = {k: v for k, v in result_groups.items() if k in non_empty_keys}
    
    group_sizes = [(len(group), group) for group in candidate_groups_dict.values()]
    max_group_size, largest_group = max(group_sizes, key=lambda x: x[0])
    
    best_candidate = min(largest_group, key=lambda c: c.execution_time or float('inf'))
    
    total_valid = len(valid_candidates)
    consistency = max_group_size / total_valid if total_valid > 0 else 0.0

    exec_time_str = f"{best_candidate.execution_time:.3f}s" if best_candidate.execution_time is not None else "N/A"
    logger.debug(
        f"{log_prefix}Self-consistency selection: {max_group_size}/{total_valid} (consistency={consistency:.2f}), "
        f"execution_time={exec_time_str}"
    )

    return best_candidate.sql, consistency, True


def _parse_pairwise_llm_response(response: str) -> Optional[str]:
    """Parse output from LLM for Pairwise Selection."""
    try:
        match = re.search(r"<result>(.*?)</result>", response, flags=re.DOTALL)
        if not match:
            return None
        answer_content = match.group(1).strip().upper()
        if answer_content not in ["A", "B", "TIE"]:
            return None
        return answer_content
    except Exception as e:
        logger.error(f"Error parsing pair selection response: {e}")
        return None

async def select_best_from_candidates_with_pairwise(
    candidates: List[SQLCandidate],
    session: Any,
    llm_client: Any,
    db_schema: Any,
    db_path: str,
    log_prefix: str = ""
) -> Tuple[Optional[str], float, bool]:
    """Select SQL through pairwise LLM comparisons."""
    if not candidates:
        return None, 0.0, False
        
    system_config = config.get_system_config()
    pairwise_cfg = system_config.get("sql_selection_config", {})
    
    filter_top_k = pairwise_cfg.get("filter_top_k_sql", 3)
    sampling_budget = pairwise_cfg.get("evaluator_sampling_budget", 1)
    shortcut_threshold = pairwise_cfg.get("shortcut_consistency_score_threshold", 0.6)

    valid_candidates = []
    sql_map_to_result_str = {}
    non_empty_results = set()
    
    for c in candidates:
        exec_result = cached_execute_sql_with_timeout(db_path, c.sql)
        if is_valid_execution_result(exec_result):
            # Convert result to frozenset of tuples to be hashable and independent of row order
            result_key = frozenset(tuple(row) for row in exec_result.result)
            valid_candidates.append((c, result_key))
            sql_map_to_result_str[c.sql] = format_execution_result(exec_result)
            if has_non_empty_result(exec_result):
                non_empty_results.add(result_key)

    if not valid_candidates:
        logger.warning(f"{log_prefix}No SQL candidates with valid results, using execution-result selection.")
        return select_best_from_candidates(candidates, db_path, log_prefix)
        
    if non_empty_results:
        valid_candidates = [(c, res) for c, res in valid_candidates if res in non_empty_results]
    
    counter = Counter(res for _, res in valid_candidates)
    
    deduplicated = []
    seen = set()
    for c, res in valid_candidates:
        if res not in seen:
            score = counter[res] / len(valid_candidates)
            exec_time = c.execution_time if c.execution_time is not None else float('inf')
            deduplicated.append((c.sql, sql_map_to_result_str[c.sql], score, exec_time))
            seen.add(res)
            
    top_k = sorted(deduplicated, key=lambda x: (x[2], -x[3]), reverse=True)[:filter_top_k]
    
    if len(top_k) == 0:
        return None, 0.0, False
        
    if len(top_k) == 1:
        logger.debug(f"{log_prefix}Selected SQL with confidence {top_k[0][2]:.2f}.")
        return top_k[0][0], top_k[0][2], True
        
    if top_k[0][2] >= shortcut_threshold:
        logger.debug(f"{log_prefix}Selected SQL with confidence {top_k[0][2]:.2f}.")
        return top_k[0][0], top_k[0][2], True
        
    db_schema_profile = _build_and_condense_schema_context(db_schema.tables)
    
    pair_sqls_to_eval = []
    for i, a in enumerate(top_k):
        for j, b in enumerate(top_k):
            if i < j:
                pair_sqls_to_eval.append((a, b, i, j))
                
    win_matrix = np.zeros((len(top_k), len(top_k), sampling_budget))
    
    async def compare_pair(a, b, i, j):
        prompt = get_br_pair_selection_prompt(
            DATABASE_SCHEMA=db_schema_profile,
            QUESTION=session.question,
            HINT=session.evidence or "None",
            QUERY_A=a[0],
            RESULT_A=a[1],
            QUERY_B=b[0],
            RESULT_B=b[1]
        )
        t_tasks = []
        for _ in range(sampling_budget):
            t_tasks.append(llm_client.generate(prompt=prompt, system_prompt="You are a SQL expert.", temperature=0.3))
        
        responses = await asyncio.gather(*t_tasks, return_exceptions=True)
        votes = []
        tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        for res in responses:
            if isinstance(res, Exception):
                logger.error(f"Error checking pairwise SQL candidates: {res}")
                votes.append(None)
                continue
            if isinstance(res, LLMResponse):
                vote = _parse_pairwise_llm_response(res.content)
                votes.append(vote)
                if res.usage:
                    tokens["prompt_tokens"] += res.usage.get("prompt_tokens", 0)
                    tokens["completion_tokens"] += res.usage.get("completion_tokens", 0)
                    tokens["total_tokens"] += res.usage.get("total_tokens", 0)
        
        return i, j, votes, tokens

    compare_tasks = [compare_pair(a, b, i, j) for a, b, i, j in pair_sqls_to_eval]
    has_failure = False
    results = await asyncio.gather(*compare_tasks)
    
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for i, j, votes, tokens in results:
        total_tokens["prompt_tokens"] += tokens["prompt_tokens"]
        total_tokens["completion_tokens"] += tokens["completion_tokens"]
        total_tokens["total_tokens"] += tokens["total_tokens"]
        
        for v_idx, vote in enumerate(votes):
            if vote == "A":
                win_matrix[i, j, v_idx] = 1
                win_matrix[j, i, v_idx] = 0
            elif vote == "B":
                win_matrix[j, i, v_idx] = 1
                win_matrix[i, j, v_idx] = 0
            elif vote == "TIE":
                win_matrix[i, j, v_idx] = 0.5
                win_matrix[j, i, v_idx] = 0.5
            else:
                has_failure = True
                
    if has_failure:
        logger.error(f"{log_prefix}Pairwise selection had failures/invalid votes, using the highest-consistency candidate")
        session.total_tokens += total_tokens["total_tokens"]
        for k, v in total_tokens.items():
            session.meta_agent_usage[k] += v
        return top_k[0][0], top_k[0][2], True
        
    sql_count = len(top_k)
    robust_win_matrix = np.zeros((sql_count, sql_count))
    for x in range(sql_count):
        for y in range(sql_count):
            if x != y:
                robust_win_matrix[x, y] = np.mean(win_matrix[x, y, :])
                
    ranking_scores = np.mean(robust_win_matrix, axis=1)
    
    total_score = sum(sql[2] for sql in top_k)
    if total_score > 0:
        score_weights = np.array([sql[2] for sql in top_k]) / total_score
        ranking_scores = ranking_scores * score_weights
        
    ranking = np.argsort(-ranking_scores)
    best_idx = ranking[0]
    best_candidate = top_k[best_idx]
    
    logger.debug(f"{log_prefix}Pairwise Selected SQL idx {best_idx} with weighted score {ranking_scores[best_idx]:.3f}")
    
    session.total_tokens += total_tokens["total_tokens"]
    for k, v in total_tokens.items():
        session.meta_agent_usage[k] += v
    
    return best_candidate[0], float(best_candidate[2]), True
