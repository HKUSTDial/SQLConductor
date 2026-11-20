from typing import Dict, Optional, Tuple
from loguru import logger
import re

from .base import BaseChecker


class TimeChecker(BaseChecker):
    """Fixes time-related SQL issues (e.g., strftime comparisons)"""
    
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
        Normalize time comparisons.
        """
        time_revised_sql = self._check_time(sql)
        if time_revised_sql:
            logger.debug("[TimeChecker] Time expression normalized.")
            return time_revised_sql, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        logger.trace(f"[TimeChecker] No time errors found, skip.")
        return sql, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    def _check_time(self, sql: str) -> Optional[str]:
        """
        Fix strftime comparisons where numeric values should be strings.
        Pattern: strftime(...) >= 2024 -> strftime(...) >= '2024'
        """
        res = re.sub(
            r"(strftime *\([^\(]*?\) *[>=<]+ *)(\d{4,})",
            r"\1'\2'",
            sql
        )
        if res != sql:
            return res
        return None

