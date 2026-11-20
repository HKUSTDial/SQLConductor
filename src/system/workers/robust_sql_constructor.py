import random
import re
import json
import asyncio
from typing import Dict, Optional, Tuple, List, Any
from collections import defaultdict
from pathlib import Path
from loguru import logger

from src.system.config import config
from src.system.types import WorkerType, WorkerInput, WorkerResult, SQLCandidate
from src.system.workers.base import BaseWorker
from src.system.prompt_factory import (
    get_robust_dc_prompt,
    get_robust_skeleton_prompt,
    get_robust_icl_prompt
)
from src.database.sql_execution import (
    cached_execute_sql_with_timeout,
    is_valid_execution_result,
    SQLExecutionResultType,
    normalize_sql
)
from src.system.workers.utils import (
    _build_hint_from_steps, 
    _extract_sql_query_answer,
    build_sql_candidates,
    get_icl_examples_path,
    SQL_GENERATION_LLM_KWARGS_TEMPERATURE,
    SQL_VALIDATION_MAX_TRIES,
    select_fastest_sql_in_sc_candidates
)

DC_SAMPLING_BUDGET = 8
SKELETON_SAMPLING_BUDGET = 8
ICL_SAMPLING_BUDGET = 8


class RobustSQLConstructorWorker(BaseWorker):
    """Generate SQL with DC, skeleton, and ICL prompts, then select by execution results."""

    def __init__(self, worker_type: WorkerType):
        super().__init__(worker_type)
        self.description = "The core worker. Use this when you have sufficient information (schema, values, logic) to draft the initial SQL query."
        self._icl_examples_cache = None
        self._icl_examples_path = get_icl_examples_path()

    def _get_icl_examples(self, question_id: str) -> List[Dict[str, str]]:
        """Load ICL examples from the config-derived path."""
        if self._icl_examples_cache is None:
            if self._icl_examples_path:
                try:
                    with open(self._icl_examples_path, "r") as f:
                        self._icl_examples_cache = json.load(f)
                        logger.trace(f"[Robust SQL Constructor] Loaded ICL examples from {self._icl_examples_path}")
                except Exception as e:
                    logger.error(f"[Robust SQL Constructor] Failed to load ICL examples: {e}")
                    self._icl_examples_cache = {}
            else:
                self._icl_examples_cache = {}

        return self._icl_examples_cache.get(str(question_id), [])

    def _format_icl_examples(self, examples: List[Dict[str, str]]) -> str:
        """Format few-shot examples for the prompt."""
        if not examples:
            return ""
        
        formatted = ""
        for i, ex in enumerate(examples, 1):
            formatted += f"Example {i}:\n"
            formatted += f"Question: {ex.get('question', '')}\n"
            formatted += f"SQL: {ex.get('sql', '')}\n\n"
        return formatted.strip()

    def _parse_llm_response(self, response: str) -> Optional[str]:
        """Parse LLM response to extract SQL."""
        
        
        content = response
        
        match = re.search(r"<result>(.*?)</result>", content, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if answer.startswith("```sql") and answer.endswith("```"):
                answer = answer[6:-3].strip()
            elif answer.startswith("```") and answer.endswith("```"):
                 answer = answer[3:-3].strip()
            return normalize_sql(answer)

        if "</reasoning>\n[result>" in content:
             content = content.replace("</reasoning>\n[result>", "</reasoning>\n<result>")

        return _extract_sql_query_answer(response)

    async def execute(self, input_data: WorkerInput) -> WorkerResult:
        self.turn_count += 1
        
        try:
            steps = input_data.context.get("steps", [])
            hint = _build_hint_from_steps(steps, input_data.context)
            
            db_path = input_data.context.get("db_path")
            if not db_path:
                raise ValueError("db_path not found in WorkerInput context.")

            question_id = input_data.context.get("question_id")
            
            schema_context = input_data.schema
            question = input_data.question
            
            dc_prompt = get_robust_dc_prompt(
                DATABASE_SCHEMA=schema_context,
                QUESTION=question,
                HINT=hint
            ).strip()
            
            skeleton_prompt = get_robust_skeleton_prompt(
                DATABASE_SCHEMA=schema_context,
                QUESTION=question,
                HINT=hint
            ).strip()
            
            icl_examples = (
                self._get_icl_examples(str(question_id))
                if input_data.context.get("use_icl_examples", True) and question_id is not None
                else []
            )
            icl_prompt = get_robust_icl_prompt(
                FEW_SHOT_EXAMPLES=self._format_icl_examples(icl_examples),
                DATABASE_SCHEMA=schema_context,
                QUESTION=question,
                HINT=hint
            ).strip()

            tasks = []
            
            tasks.append(self.llm_client.generate(
                prompt=dc_prompt,
                temperature=SQL_GENERATION_LLM_KWARGS_TEMPERATURE,
                n=DC_SAMPLING_BUDGET,
                stop=["</result>"]
            ))
            
            tasks.append(self.llm_client.generate(
                prompt=skeleton_prompt,
                temperature=SQL_GENERATION_LLM_KWARGS_TEMPERATURE,
                n=SKELETON_SAMPLING_BUDGET,
                stop=["</result>"]
            ))
            
            tasks.append(self.llm_client.generate(
                prompt=icl_prompt,
                temperature=SQL_GENERATION_LLM_KWARGS_TEMPERATURE,
                n=ICL_SAMPLING_BUDGET,
                stop=["</result>"]
            ))
            
            results = await asyncio.gather(*tasks)
            
            all_sql_candidates = []
            total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
            strategy_names = ["DC", "Skeleton", "ICL"]
            strategy_stats = {
                "DC": {"generated": 0, "parsed": 0, "errors": 0},
                "Skeleton": {"generated": 0, "parsed": 0, "errors": 0},
                "ICL": {"generated": 0, "parsed": 0, "errors": 0}
            }
            
            for strategy_idx, response in enumerate(results):
                strategy_name = strategy_names[strategy_idx]
                
                if response.usage:
                    total_usage["prompt_tokens"] += response.usage.get("prompt_tokens", 0)
                    total_usage["completion_tokens"] += response.usage.get("completion_tokens", 0)
                    total_usage["total_tokens"] += response.usage.get("total_tokens", 0)
                
                if response.error:
                    logger.error(f"[{strategy_name}][{self.worker_type.value}] LLM error: {response.error}")
                    strategy_stats[strategy_name]["errors"] += 1
                    continue
                    
                contents = response.contents or ([response.content] if response.content else [])
                for sample_idx, content in enumerate(contents, 1):
                    strategy_stats[strategy_name]["generated"] += 1
                    
                    # Restore the closing tag omitted by the stop sequence.
                    if "</result>" not in content:
                        content += "</result>"
                        
                    sql = self._parse_llm_response(content)
                    if sql:
                        strategy_stats[strategy_name]["parsed"] += 1
                        all_sql_candidates.append((sql, content, strategy_name))
                    else:
                        logger.debug(f"[{strategy_name}][{self.worker_type.value}] Failed to parse SQL from sample {sample_idx}")
            
            if not all_sql_candidates:
                 return WorkerResult(
                    worker_type=self.worker_type,
                    result="",
                    turn=self.turn_count,
                    success=False,
                    error_message="Failed to generate any valid SQL candidates.",
                    usage=total_usage,
                    metadata={"stats": {}, "confidence": 0.0, "valid_proportion": 0.0, "total_candidates": 0, "candidate_count": 0},
                    sql_candidates=[]
                )
            
            sql_candidates = build_sql_candidates(all_sql_candidates, db_path, log_prefix=f"[{self.worker_type.value}] ")
            
            best_sql, best_raw_content, confidence, success, stats = select_fastest_sql_in_sc_candidates(
                sql_candidates=all_sql_candidates,
                db_path=db_path,
                question_id=question_id,
                worker_type=self.worker_type.value,
                strategy_stats=strategy_stats,
                log_prefix=""
            )
            
            total_candidates = len(all_sql_candidates)
            valid_count = stats.get("valid", 0)
            valid_proportion = valid_count / total_candidates if total_candidates > 0 else 0.0
            
            metadata = {
                "stats": stats,
                "confidence": confidence,
                "valid_proportion": valid_proportion,
                "total_candidates": total_candidates,
                "candidate_count": len(sql_candidates)
            }
            
            if best_sql and best_raw_content:
                
                return WorkerResult(
                    worker_type=self.worker_type,
                    result=best_raw_content,
                    turn=self.turn_count,
                    success=True,
                    error_message=None,
                    usage=total_usage,
                    metadata=metadata,
                    sql_candidates=sql_candidates
                )
            
            return WorkerResult(
                worker_type=self.worker_type,
                result="",
                turn=self.turn_count,
                success=False,
                error_message="No SQL candidates generated or all failed execution.",
                usage=total_usage,
                metadata=metadata,
                sql_candidates=sql_candidates
            )

        except Exception as e:
            logger.error(f"RobustSQLConstructorWorker execution failed: {e}")
            return WorkerResult(
                worker_type=self.worker_type,
                result="",
                turn=self.turn_count,
                success=False,
                error_message=str(e)
            )
