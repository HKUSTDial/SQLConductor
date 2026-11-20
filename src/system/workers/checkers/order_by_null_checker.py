from typing import Dict, Optional, Tuple
from loguru import logger
import re

from .base import BaseChecker
from ..utils import _extract_sql_query_answer


class OrderByNullChecker(BaseChecker):
    """Checks for NULL handling issues in ORDER BY clauses"""
    
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
        Check for ORDER BY NULL errors and revise if found.
        """
        order_by_null_suggestion = self._check_order_by_null(sql)
        if not order_by_null_suggestion:
            logger.trace(f"[OrderByNullChecker] No order-by-null errors found, skip.")
            return sql, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        logger.debug("[OrderByNullChecker] NULL ordering issue detected.")
        
        from src.system.workers.checkers.prompts import build_common_checker_prompt
        prompt = build_common_checker_prompt(
            database_schema=schema,
            question=question,
            hint=hint,
            query=sql,
            suggestions=order_by_null_suggestion
        )
        
        total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        parsed_sql_candidate = None
        
        attempts = 0
        while not parsed_sql_candidate and attempts < sampling_budget:
            response = await llm_client.generate(
                prompt=prompt,
                temperature=0.8,
                max_tokens=2000,
                n=1,
                stop=None
            )
            
            if response.usage:
                total_token_usage["prompt_tokens"] += response.usage.get("prompt_tokens", 0)
                total_token_usage["completion_tokens"] += response.usage.get("completion_tokens", 0)
                total_token_usage["total_tokens"] += response.usage.get("total_tokens", 0)
            
            if response.error:
                logger.error(f"LLM call failed in OrderByNullChecker: {response.error}")
                break
            
            content = response.content or (response.contents[0] if response.contents else "")
            parsed_sql_candidate = self._parse_llm_response(content)
            if not parsed_sql_candidate:
                parsed_sql_candidate = _extract_sql_query_answer(content)
            
            attempts += 1
        
        if parsed_sql_candidate:
            return parsed_sql_candidate, total_token_usage
        else:
            logger.debug("[OrderByNullChecker] No valid revision; keeping input SQL.")
            return sql, total_token_usage
    
    def _check_order_by_null(self, sql: str) -> Optional[str]:
        """
        Check for ORDER BY with LIMIT that might need NULL filtering.
        Returns suggestion string if issues found, None otherwise.
        """
        inn = re.findall(r"ORDER BY .*?(?<!DESC )LIMIT +\d+;{0,1}", sql)
        if not inn:
            return None
        
        # Skip if contains aggregation functions
        for x in inn:
            if re.findall(r"SUM\(|COUNT\(", x):
                return None
        
        suggestions = []
        for x in inn:
            suggestions.append(
                f"Please add `IS NOT NULL` condition **in the WHERE clause** "
                f"for the ORDER BY column: {x}"
            )
        
        return "\n".join(suggestions)
