from typing import Dict, Optional, Tuple
from loguru import logger
import re

from .base import BaseChecker
from ..utils import _extract_sql_query_answer


class SelectChecker(BaseChecker):
    """Checks for incorrect SELECT clause usage"""
    
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
        Check for SELECT errors and revise if found.
        """
        # Restrict concatenation handling to SELECT ... FROM.
        select_concat = re.findall(r"^SELECT.*?\|\| ' ' \|\| .*?FROM", sql, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        if select_concat:
            sql = sql.replace("|| ' ' ||", ', ')
        
        select_suggestion = self._check_select(sql)
        if not select_suggestion:
            logger.trace(f"[SelectChecker] No select errors found, skip.")
            return sql, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        logger.debug("[SelectChecker] SELECT issue detected.")
        
        from src.system.workers.checkers.prompts import build_common_checker_prompt
        prompt = build_common_checker_prompt(
            database_schema=schema,
            question=question,
            hint=hint,
            query=sql,
            suggestions=select_suggestion
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
                logger.error(f"LLM call failed in SelectChecker: {response.error}")
                break
            
            content = response.content or (response.contents[0] if response.contents else "")
            parsed_sql_candidate = self._parse_llm_response(content)
            if not parsed_sql_candidate:
                parsed_sql_candidate = _extract_sql_query_answer(content)
            
            attempts += 1
        
        if parsed_sql_candidate:
            return parsed_sql_candidate, total_token_usage
        else:
            logger.debug("[SelectChecker] No valid revision; keeping input SQL.")
            return sql, total_token_usage
    
    def _check_select(self, sql: str) -> Optional[str]:
        """
        Check for ambiguous SELECT patterns (e.g., table.*).
        Returns suggestion string if issues found, None otherwise.
        """
        identifier = r'(?:`[^`]+`|\[[^\]]+\]|"[^"]+"|[\w\.]+)'
        select_amb = re.findall(
            rf"^SELECT.*? ({identifier}\.\*).*?FROM",
            sql,
            re.IGNORECASE | re.DOTALL | re.MULTILINE
        )
        
        if select_amb:
            suggestions = []
            for idx, x in enumerate(select_amb, 1):
                suggestions.append(
                    f"{idx}. We have specified that the ambiguous query is the corresponding id column, "
                    f"please replace {x} with the corresponding id column in the above SQL"
                )
            return "\n".join(suggestions)
        return None
