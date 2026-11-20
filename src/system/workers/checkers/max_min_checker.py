from typing import Dict, Optional, Tuple, List
from loguru import logger
import re

from .base import BaseChecker
from ..utils import _extract_sql_query_answer


class MaxMinChecker(BaseChecker):
    """Checks for incorrect MAX/MIN function usage"""
    
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
        Check for MAX/MIN errors and revise if found.
        """
        max_min_suggestion = self._check_max_min(sql)
        if not max_min_suggestion:
            logger.trace(f"[MaxMinChecker] No max-min errors found, skip.")
            return sql, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        logger.debug("[MaxMinChecker] MAX/MIN issue detected.")
        
        from src.system.workers.checkers.prompts import build_common_checker_prompt
        prompt = build_common_checker_prompt(
            database_schema=schema,
            question=question,
            hint=hint,
            query=sql,
            suggestions=max_min_suggestion
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
                logger.error(f"LLM call failed in MaxMinChecker: {response.error}")
                break
            
            content = response.content or (response.contents[0] if response.contents else "")
            parsed_sql_candidate = self._parse_llm_response(content)
            if not parsed_sql_candidate:
                parsed_sql_candidate = _extract_sql_query_answer(content)
            
            attempts += 1
        
        if parsed_sql_candidate:
            return parsed_sql_candidate, total_token_usage
        else:
            logger.debug("[MaxMinChecker] No valid revision; keeping input SQL.")
            return sql, total_token_usage
    
    def _check_max_min(self, sql: str) -> Optional[str]:
        """
        Check for incorrect MAX/MIN patterns.
        Returns suggestion string if issues found, None otherwise.
        """
        identifier = r'(?:`[^`]+`|\[[^\]]+\]|"[^"]+"|[\w\.]+)'
        
        # Pattern 1: WHERE col = (SELECT MAX/MIN(col) FROM table)
        max_min_pattern = re.compile(
            rf"=\s*\(\s*SELECT\s*(MAX|MIN)\s*\(\s*({identifier})\s*\)\s*FROM\s*({identifier})",
            re.IGNORECASE | re.DOTALL
        )
        fun_amb = max_min_pattern.findall(sql)
        
        # Pattern 2: = (SELECT ... LIMIT)
        order_amb = set(re.findall(r"= (\(SELECT .* LIMIT \d\))", sql, re.IGNORECASE | re.DOTALL))
        
        # Pattern 3: SELECT MIN/MAX(...) ... LIMIT 1
        select_amb_pattern = re.compile(
            rf"^SELECT[^\(\)]*? ((MIN|MAX)\(\s*{identifier}\s*\)).*?LIMIT 1",
            re.IGNORECASE | re.DOTALL | re.MULTILINE
        )
        select_amb = set(select_amb_pattern.findall(sql))
        
        suggestions = []
        
        # Suggestion 1: Replace nested MAX/MIN with ORDER BY
        for fun in fun_amb:
            fuc, col, table = fun
            order = "DESC" if fuc == "MAX" else "ASC"
            suggestions.append(
                f"WHERE {col} = (SELECT {fuc}({col}) FROM {table}): "
                f"Please use ORDER BY {table}.{col} {order} LIMIT 1 instead of nested SQL"
            )
        
        # Suggestion 2: Replace nested SELECT with JOIN
        for fun in order_amb:
            suggestions.append(f"{fun}: Please use JOIN instead of nested SQL")
        
        # Suggestion 3: Remove redundant MAX/MIN with LIMIT
        for fun in select_amb:
            suggestions.append(
                f"{fun[0]}: {fun[1]} function is redundant due to LIMIT clause, "
                f"please use ORDER BY + LIMIT instead"
            )
        
        if len(suggestions) > 0:
            return "\n".join([f"{idx+1}. {suggestion}" for idx, suggestion in enumerate(suggestions)])
        return None
