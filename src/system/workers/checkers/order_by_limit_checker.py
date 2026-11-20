from typing import Dict, Optional, Tuple
from loguru import logger
import re

from .base import BaseChecker
from ..utils import _extract_sql_query_answer


class OrderByLimitChecker(BaseChecker):
    """Checks for incorrect ORDER BY with LIMIT usage"""
    
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
        Check for ORDER BY LIMIT errors and revise if found.
        """
        order_by_limit_suggestion = self._check_order_by_limit(sql)
        if not order_by_limit_suggestion:
            logger.trace(f"[OrderByLimitChecker] No order-by-limit errors found, skip.")
            return sql, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        logger.debug("[OrderByLimitChecker] ORDER BY / LIMIT issue detected.")
        
        from src.system.workers.checkers.prompts import build_common_checker_prompt
        prompt = build_common_checker_prompt(
            database_schema=schema,
            question=question,
            hint=hint,
            query=sql,
            suggestions=order_by_limit_suggestion
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
                logger.error(f"LLM call failed in OrderByLimitChecker: {response.error}")
                break
            
            content = response.content or (response.contents[0] if response.contents else "")
            parsed_sql_candidate = self._parse_llm_response(content)
            if not parsed_sql_candidate:
                parsed_sql_candidate = _extract_sql_query_answer(content)
            
            attempts += 1
        
        if parsed_sql_candidate:
            return parsed_sql_candidate, total_token_usage
        else:
            logger.debug("[OrderByLimitChecker] No valid revision; keeping input SQL.")
            return sql, total_token_usage
    
    def _check_order_by_limit(self, sql: str) -> Optional[str]:
        """
        Check for incorrect ORDER BY with MIN/MAX functions.
        Returns suggestion string if issues found, None otherwise.
        """
        identifier = r'(?:`[^`]+`|\[[^\]]+\]|"[^"]+"|[\w\.]+)'
        order_by_pattern = re.compile(
            rf"ORDER BY ((MIN|MAX)\(\s*({identifier})\s*\)).*? LIMIT \d+",
            re.IGNORECASE | re.DOTALL
        )
        res = order_by_pattern.search(sql)
        
        if res:
            return (
                f"The SQL uses the ORDER BY function incorrectly, using MIN/MAX in ORDER BY clause "
                f"is incorrect (`{res.group()}`), please correct the SQL. If the SQL contains GROUP BY, "
                f"please judge whether the content of `{res.groups()[0]}` needs to use "
                f"`SUM({res.groups()[2]})`."
            )
        return None
