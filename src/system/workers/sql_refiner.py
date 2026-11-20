import random
import asyncio
from typing import Dict, Optional, Tuple, List, Any
from collections import defaultdict
from loguru import logger

from src.database.sql_execution import (
    cached_execute_sql_with_timeout,
    is_valid_execution_result,
    has_non_empty_result,
    SQLExecutionResultType
)
from src.system.types import WorkerType, WorkerInput, WorkerResult, SQLCandidate
from src.system.prompt_factory import get_worker_prompt
from src.system.workers.base import BaseWorker
from src.system.workers.utils import (
    _build_hint_from_steps, 
    _extract_sql_query_answer,
    build_sql_candidates,
    select_best_from_candidates,
    SQL_REVISION_LLM_KWARGS_TEMPERATURE,
    SQL_REVISION_LLM_KWARGS_N,
    SQL_VALIDATION_MAX_TRIES
)

class SQLRefinerWorker(BaseWorker):
    """Worker that refines SQL."""

    def __init__(self, worker_type: WorkerType):
        super().__init__(worker_type)
        self.description = "Use *only* after `SQL Constructor` if the generated SQL is invalid, returns no results, or is logically incorrect based on execution feedback."
    
    def _build_prompt(self, input_data: WorkerInput) -> str:
        steps = input_data.context.get("steps", [])
        hint = _build_hint_from_steps(steps, input_data.context)
        
        return get_worker_prompt(
            self.worker_type,
            QUESTION=input_data.question,
            HINT=hint,
            SCHEMA_CONTEXT=input_data.schema
        )

    async def execute(self, input_data: WorkerInput) -> WorkerResult:
        self.turn_count += 1
        
        try:
            db_path = input_data.context.get("db_path")
            if not db_path:
                raise ValueError("db_path not found in WorkerInput context for SQLRefinerWorker.")

            session = input_data.context.get("session")
            if session and hasattr(session, 'sql_candidates') and session.sql_candidates:
                logger.debug(f"[{self.worker_type.value}] Refining {len(session.sql_candidates)} candidates")
                
                refined_candidates = await self.refine_candidates(session.sql_candidates, input_data)
                
                best_sql, confidence, success = select_best_from_candidates(
                    refined_candidates, db_path, log_prefix=f"[{self.worker_type.value}] "
                )
                
                return WorkerResult(
                    worker_type=self.worker_type,
                    result=f"<sql>{best_sql}</sql>" if best_sql else "",
                    turn=self.turn_count,
                    success=success,
                    error_message=None if success else "No valid refined SQL",
                    sql_candidates=refined_candidates,
                    metadata={
                        "refined_count": len(refined_candidates),
                        "valid_count": sum(1 for c in refined_candidates if c.is_valid),
                        "confidence": confidence
                    }
                )

            prompt = self._build_prompt(input_data)

            full_response, _, _, total_usage, sql_candidates = await self.generate_most_consistent_sql_query(
                prompt, 
                db_path
            )

            if not full_response:
                return WorkerResult(
                    worker_type=self.worker_type,
                    result="",
                    turn=self.turn_count,
                    success=False,
                    error_message="Failed to generate a consistent SQL query.",
                    usage=total_usage,
                    sql_candidates=sql_candidates
                )

            return WorkerResult(
                worker_type=self.worker_type,
                result=full_response,
                turn=self.turn_count,
                success=True,
                usage=total_usage,
                sql_candidates=sql_candidates
            )
            
        except Exception as e:
            logger.error(f"Worker {self.worker_type.value} execution failed: {e}")
            return WorkerResult(
                worker_type=self.worker_type,
                result="",
                turn=self.turn_count,
                success=False,
                error_message=str(e)
            )

    async def generate_most_consistent_sql_query(
        self, 
        prompt: str, 
        db_path: str
    ) -> Tuple[Optional[str], float, bool, Dict[str, int], List[SQLCandidate]]:
        all_sql_queries = []
        result_groups = defaultdict(list)
        valid_sql_query_tries = 0
        response_map = {}
        raw_responses: List[str] = []
        executed_sql_count = 0
        fail_timeout_count = 0
        empty_result_count = 0
        parse_fail_count = 0
        
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for round_idx in range(1, 11):
            if len(all_sql_queries) >= SQL_REVISION_LLM_KWARGS_N:
                break
            if valid_sql_query_tries >= SQL_VALIDATION_MAX_TRIES and len(all_sql_queries) > 0:
                break
            
            remaining = SQL_REVISION_LLM_KWARGS_N - len(all_sql_queries)
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=SQL_REVISION_LLM_KWARGS_TEMPERATURE,
                max_tokens=self.config.model_config.max_tokens,
                n=SQL_REVISION_LLM_KWARGS_N,
                stop=self.config.model_config.stop
            )
            
            if response.usage:
                if total_usage["prompt_tokens"] == 0:
                    total_usage["prompt_tokens"] = response.usage.get("prompt_tokens", 0)
                    total_usage["completion_tokens"] = response.usage.get("completion_tokens", 0)
                    total_usage["total_tokens"] = response.usage.get("total_tokens", 0)
            
            if response.error:
                logger.error(f"LLM call failed: {response.error}")
                break
            
            response_contents = response.contents or ([response.content] if response.content else [])
            for idx, content in enumerate(response_contents, 1):
                logger.trace(
                    f"[{self.worker_type.value}] Received sample {idx} in round {round_idx}."
                )
            raw_responses.extend(response_contents)
            
            for content in response_contents:
                sql_query = _extract_sql_query_answer(content)
                if sql_query is None:
                    parse_fail_count += 1
                    continue
                
                response_map[sql_query] = content
                
                sql_query_execution_result = cached_execute_sql_with_timeout(db_path, sql_query)
                executed_sql_count += 1
                if not is_valid_execution_result(sql_query_execution_result):
                    fail_timeout_count += 1
                elif not has_non_empty_result(sql_query_execution_result):
                    empty_result_count += 1
                if is_valid_execution_result(sql_query_execution_result) or valid_sql_query_tries >= SQL_VALIDATION_MAX_TRIES:
                    all_sql_queries.append(sql_query)
                    if is_valid_execution_result(sql_query_execution_result):
                        result_groups[frozenset(sql_query_execution_result.result)].append(sql_query)
                else:
                    valid_sql_query_tries += 1
        
        sql_tuples = [(sql, response_map[sql], "refiner") for sql in all_sql_queries]
        sql_candidates = build_sql_candidates(sql_tuples, db_path, log_prefix=f"[{self.worker_type.value}] ") if sql_tuples else []
        
        if result_groups:
            group_sizes = [len(group) for group in result_groups.values()]
            max_group_size = max(group_sizes)
            candidate_groups = [group for group in result_groups.values() if len(group) == max_group_size]
            most_consistent_group = random.choice(candidate_groups)
            most_consistent_sql = random.choice(most_consistent_group)
            num_valid_queries = sum(group_sizes)
            consistency_score = max_group_size / num_valid_queries if num_valid_queries > 0 else 0
            logger.debug(
                f"[{self.worker_type.value}] Selected SQL: valid={num_valid_queries}, "
                f"consistency={consistency_score:.2f}"
            )
            return response_map[most_consistent_sql], consistency_score, True, total_usage, sql_candidates

        if all_sql_queries:
            logger.warning("Selected SQL has no valid execution result.")
            selected_sql = random.choice(all_sql_queries)
            return response_map[selected_sql], 0, False, total_usage, sql_candidates
            
        raw_responses_content = [resp for resp in raw_responses if resp]
        if raw_responses_content:
            logger.warning("No SQL could be parsed from model responses.")
            return random.choice(raw_responses_content), 0, False, total_usage, []
            
        logger.error("All LLM calls failed or produced empty content. Cannot generate SQL.")
        return "Error: The language model failed to produce any output.", 0, False, total_usage, []

    async def refine_candidates(
        self,
        candidates: List[SQLCandidate],
        input_data: WorkerInput,
    ) -> List[SQLCandidate]:
        """
        Refine multiple SQL candidates in parallel.
        
        Args:
            candidates: List of SQLCandidate objects to refine
            input_data: WorkerInput containing question, schema, context
        
        Returns:
            List of refined SQLCandidate objects
        """
        if not candidates:
            return []
        
        db_path = input_data.context.get("db_path")
        if not db_path:
            logger.warning("db_path not found in context for refine_candidates")
            return candidates
        
        async def refine_single_candidate(candidate: SQLCandidate) -> SQLCandidate:
            """Refine a single SQL candidate."""
            try:
                steps = input_data.context.get("steps", [])
                hint = _build_hint_from_steps(steps, input_data.context)
                
                if hint:
                    hint = f"{hint}\n\nSQL to refine: {candidate.sql}"
                else:
                    hint = f"SQL to refine: {candidate.sql}"
                
                prompt = get_worker_prompt(
                    self.worker_type,
                    QUESTION=input_data.question,
                    HINT=hint,
                    SCHEMA_CONTEXT=input_data.schema
                )
                
                response = await self.llm_client.generate(
                    prompt=prompt,
                    temperature=SQL_REVISION_LLM_KWARGS_TEMPERATURE,
                    max_tokens=self.config.model_config.max_tokens,
                    n=1,
                    stop=self.config.model_config.stop
                )
                
                if response.error or not response.content:
                    logger.debug("Candidate refinement failed.")
                    return candidate
                
                refined_sql = _extract_sql_query_answer(response.content)
                if not refined_sql:
                    return candidate
                
                exec_result = cached_execute_sql_with_timeout(db_path, refined_sql)
                is_valid = is_valid_execution_result(exec_result)
                
                return SQLCandidate(
                    sql=refined_sql,
                    raw_content=response.content,
                    strategy=f"{candidate.strategy}_refined" if candidate.strategy else "refined",
                    is_valid=is_valid,
                    execution_time=None
                )
                
            except Exception as e:
                logger.debug(f"Error refining candidate: {e}")
                return candidate
        
        refined = await asyncio.gather(*[refine_single_candidate(c) for c in candidates])
        
        valid_count = sum(1 for c in refined if c.is_valid)
        logger.debug(
            f"[{self.worker_type.value}] Refined {len(candidates)} candidates "
            f"(valid: {valid_count}, invalid: {len(refined) - valid_count})"
        )
        
        return list(refined)
