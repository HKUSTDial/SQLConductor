"""
Base class for SQL checkers
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
from loguru import logger
import re


class BaseChecker(ABC):
    """Base class for all SQL checkers"""
    
    @abstractmethod
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
        Check SQL and revise if necessary.
        
        Args:
            sql: SQL query to check
            question: Original question
            schema: Database schema
            hint: Hint from previous steps
            db_path: Path to database file
            llm_client: LLM client for revision
            sampling_budget: Number of revision attempts
            
        Returns:
            Tuple of (revised_sql, token_usage_dict)
        """
        pass
    
    def _parse_llm_response(self, response: str) -> Optional[str]:
        """Extract SQL from <result> or <sql> tags in an LLM response."""
        try:
            answer_match = re.search(r"<result>(.*?)</result>", response, re.DOTALL)
            if answer_match:
                answer_content = answer_match.group(1).strip()
            else:
                answer_match = re.search(r"<sql>(.*?)</sql>", response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                else:
                    logger.debug("No <result> or <sql> tag found in LLM response")
                    return None
            
            if answer_content.startswith("```sql"):
                answer_content = answer_content[6:]
            if answer_content.endswith("```"):
                answer_content = answer_content[:-3]
            answer_content = answer_content.strip()
            
            return answer_content
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return None

