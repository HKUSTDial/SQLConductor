from typing import Dict, Optional, Tuple, List
from collections import Counter
from loguru import logger

from src.database.sql_execution import (
    cached_execute_sql_with_timeout,
    is_valid_execution_result,
    has_non_empty_result,
    SQLExecutionResultType,
    format_execution_result
)
from .base import BaseChecker
from ..utils import _extract_sql_query_answer


class SyntaxChecker(BaseChecker):
    """Checks SQL syntax errors and revises using execution feedback"""
    
    async def check_and_revise(
        self,
        sql: str,
        question: str,
        schema: str,
        hint: str,
        db_path: str,
        llm_client,
        sampling_budget: int = 1
    ) -> Tuple[str, Dict[str, int]]:
        """
        Execute SQL and check for syntax/execution errors.
        If errors found, use LLM to revise based on execution feedback.
        """
        execution_result = cached_execute_sql_with_timeout(db_path, sql)
        
        # Empty and all-NULL results are handled by ResultChecker.
        if is_valid_execution_result(execution_result):
            logger.trace(f"[SyntaxChecker] No execution error reported (result_type={execution_result.result_type.value}).")
            return sql, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        logger.debug(f"Execution result type: {execution_result.result_type}, "
                    f"Error: {execution_result.error_message if execution_result.error_message else 'N/A'}")
        
        execution_result_str = format_execution_result(execution_result)
        if execution_result.error_message:
            execution_result_str = f"Error: {execution_result.error_message}\n\n{execution_result_str}"
        
        from src.system.workers.checkers.prompts import build_execution_checker_prompt
        prompt = build_execution_checker_prompt(
            database_schema=schema,
            question=question,
            hint=hint,
            query=sql,
            result=execution_result_str
        )
        
        total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        all_sql_candidates = []
        
        for _ in range(10):
            if len(all_sql_candidates) >= sampling_budget:
                break
            remaining = sampling_budget - len(all_sql_candidates)
            response = await llm_client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=4096,
                n=sampling_budget,
                stop=None
            )
            
            if response.usage:
                total_token_usage["prompt_tokens"] += response.usage.get("prompt_tokens", 0)
                total_token_usage["completion_tokens"] += response.usage.get("completion_tokens", 0)
                total_token_usage["total_tokens"] += response.usage.get("total_tokens", 0)
            
            if response.error:
                logger.error(f"LLM call failed in SyntaxChecker: {response.error}")
                break
            
            response_contents = response.contents or ([response.content] if response.content else [])
            for idx, content in enumerate(response_contents, 1):
                parsed_sql = self._parse_llm_response(content)
                if parsed_sql:
                    all_sql_candidates.append(parsed_sql)
                else:
                    alt_sql = _extract_sql_query_answer(content)
                    if alt_sql:
                        all_sql_candidates.append(alt_sql)
        
        selected_sql = self._select_sql_candidate(all_sql_candidates, db_path)
        if selected_sql:
            return selected_sql, total_token_usage
        else:
            logger.debug("[SyntaxChecker] No valid revision; keeping input SQL.")
            return sql, total_token_usage
    
    def _select_sql_candidate(self, all_sql_candidates: List[str], db_path: str) -> Optional[str]:
        """Select the best SQL candidate based on execution results"""
        valid_sql_candidates = []
        non_empty_results = set()
        for sql_candidate in all_sql_candidates:
            execution_result = cached_execute_sql_with_timeout(db_path, sql_candidate)
            if is_valid_execution_result(execution_result):
                # Use frozenset to handle order differences
                result_rows = execution_result.result if execution_result.result else []
                result_key = frozenset(tuple(row) for row in result_rows)
                valid_sql_candidates.append((sql_candidate, result_key))
                if has_non_empty_result(execution_result):
                    non_empty_results.add(result_key)
        
        if len(valid_sql_candidates) == 0:
            return None
        
        if non_empty_results:
            valid_sql_candidates = [(c, res) for c, res in valid_sql_candidates if res in non_empty_results]

        counter = Counter(result_key for _, result_key in valid_sql_candidates)
        return max(valid_sql_candidates, key=lambda x: counter[x[1]])[0]
