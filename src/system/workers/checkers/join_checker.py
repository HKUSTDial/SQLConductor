from typing import Dict, Optional, Tuple
from loguru import logger
import re

from .base import BaseChecker
from ..utils import _extract_sql_query_answer


class JoinChecker(BaseChecker):
    """Checks for incorrect JOIN clause usage"""
    
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
        Check for JOIN errors and revise if found.
        """
        join_suggestion = self._check_join(sql)
        if not join_suggestion:
            logger.trace(f"[JoinChecker] No join errors found, skip.")
            return sql, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        logger.debug("[JoinChecker] JOIN issue detected.")
        
        from src.system.workers.checkers.prompts import build_common_checker_prompt
        prompt = build_common_checker_prompt(
            database_schema=schema,
            question=question,
            hint=hint,
            query=sql,
            suggestions=join_suggestion
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
                logger.error(f"LLM call failed in JoinChecker: {response.error}")
                break
            
            content = response.content or (response.contents[0] if response.contents else "")
            parsed_sql_candidate = self._parse_llm_response(content)
            if not parsed_sql_candidate:
                parsed_sql_candidate = _extract_sql_query_answer(content)
            
            attempts += 1
        
        if parsed_sql_candidate:
            return parsed_sql_candidate, total_token_usage
        else:
            logger.debug("[JoinChecker] No valid revision; keeping input SQL.")
            return sql, total_token_usage
    
    def _check_join(self, sql: str) -> Optional[str]:
        """
        Check for incorrect JOIN patterns.
        Returns suggestion string if issues found, None otherwise.
        """
        identifier = r'(?:`[^`]+`|\[[^\]]+\]|"[^"]+"|[\w\.]+)'
        join_pattern = re.compile(
            rf"JOIN\s+{identifier}(\s+AS\s+{identifier}){{0,1}}\s+ON(\s+{identifier}\.{identifier}\s*(=\s*{identifier}\.{identifier}(?:\s+OR\s+{identifier}\.{identifier}\s*=\s*{identifier}\.{identifier})+|IN\s+\(.*?\)))",
            re.IGNORECASE | re.DOTALL
        )
        
        if join_pattern.findall(sql):
            return (
                "The SQL uses the JOIN function incorrectly, due to using "
                "`JOIN table AS T ON Ta.column1 = Tb.column2 OR Ta.column1 = Tb.column3` "
                "or `JOIN table AS T ON Ta.column1 IN`, please only keep the highest priority "
                "group of `Ta.column = Tb.column` in `OR`."
            )
        return None
