import random
import json
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
from loguru import logger

from src.system.types import WorkerType, WorkerInput, WorkerResult, SQLCandidate
from src.system.prompt_factory import get_robust_icl_prompt
from src.system.workers.base import BaseWorker
from src.system.workers.utils import (
    _build_hint_from_steps, 
    _extract_sql_query_answer,
    build_sql_candidates,
    select_fastest_sql_in_sc_candidates,
    get_icl_examples_path,
    SQL_GENERATION_LLM_KWARGS_TEMPERATURE,
    SQL_GENERATION_LLM_KWARGS_N,
)

class ICLSQLConstructorWorker(BaseWorker):
    """SQL constructor using in-context examples when available."""

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
                        logger.trace(f"[ICL SQL Constructor] Loaded ICL examples from {self._icl_examples_path}")
                except Exception as e:
                    logger.error(f"[ICL SQL Constructor] Failed to load ICL examples: {e}")
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
    
    def _build_prompt(self, input_data: WorkerInput) -> str:
        steps = input_data.context.get("steps", [])
        hint = _build_hint_from_steps(steps, input_data.context)
        
        question_id = input_data.context.get("question_id")
        icl_examples = (
            self._get_icl_examples(str(question_id))
            if input_data.context.get("use_icl_examples", True) and question_id is not None
            else []
        )
        
        prompt = get_robust_icl_prompt(
            FEW_SHOT_EXAMPLES=self._format_icl_examples(icl_examples),
            DATABASE_SCHEMA=input_data.schema,
            QUESTION=input_data.question,
            HINT=hint
        )
        return prompt

    async def execute(self, input_data: WorkerInput) -> WorkerResult:
        self.turn_count += 1
        
        try:
            prompt = self._build_prompt(input_data)
            db_path = input_data.context.get("db_path")
            if not db_path:
                raise ValueError("db_path not found in WorkerInput context for ICLSQLConstructorWorker.")

            question_id = input_data.context.get("question_id")
            full_response, _, success, total_usage, metadata, sql_candidates = await self.generate_most_consistent_sql_query(
                prompt, 
                db_path,
                question_id=question_id
            )
            
            if not full_response:
                return WorkerResult(
                    worker_type=self.worker_type,
                    result="",
                    turn=self.turn_count,
                    success=False,
                    error_message="Failed to generate a consistent SQL query.",
                    usage=total_usage,
                    metadata=metadata,
                    sql_candidates=sql_candidates
                )

            return WorkerResult(
                worker_type=self.worker_type,
                result=full_response,
                turn=self.turn_count,
                success=True,
                error_message=None,
                usage=total_usage,
                metadata=metadata,
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
        db_path: str,
        question_id: Optional[str] = None
    ) -> Tuple[Optional[str], float, bool, Dict[str, int], Dict[str, Any], List[SQLCandidate]]:
        """Generate a batch of SQL candidates and select by execution-result voting."""
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        response = await self.llm_client.generate(
            prompt=prompt,
            temperature=SQL_GENERATION_LLM_KWARGS_TEMPERATURE,
            max_tokens=self.config.model_config.max_tokens,
            n=SQL_GENERATION_LLM_KWARGS_N,
            stop=["</result>"]
        )
        
        if response.usage:
            total_usage = {
                "prompt_tokens": response.usage.get("prompt_tokens", 0),
                "completion_tokens": response.usage.get("completion_tokens", 0),
                "total_tokens": response.usage.get("total_tokens", 0),
            }
        
        if response.error:
            logger.error(f"LLM call failed: {response.error}")
            metadata = {"stats": {}, "confidence": 0.0, "valid_proportion": 0.0, "total_candidates": 0, "candidate_count": 0}
            return None, 0.0, False, total_usage, metadata, []
        
        response_contents = response.contents or ([response.content] if response.content else [])
        all_sql_tuples = []  # (sql, raw_content, strategy)
        raw_responses = []
        parse_fail_count = 0
        
        for idx, content in enumerate(response_contents, 1):
            raw_responses.append(content)
            
            # Restore the closing tag omitted by the stop sequence.
            if "</result>" not in content:
                content += "</result>"
            
            sql_query = _extract_sql_query_answer(content)
            if sql_query is None:
                parse_fail_count += 1
                continue
            all_sql_tuples.append((sql_query, content, "icl"))
        
        if all_sql_tuples:
            sql_candidates = build_sql_candidates(all_sql_tuples, db_path, log_prefix=f"[{self.worker_type.value}] ")
            
            best_sql, best_raw_content, confidence, success, stats = select_fastest_sql_in_sc_candidates(
                sql_candidates=all_sql_tuples,
                db_path=db_path,
                question_id=question_id,
                worker_type=self.worker_type.value,
                log_prefix=""
            )
            stats["parse_fail"] = parse_fail_count
            
            total_candidates = len(all_sql_tuples)
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
                return best_raw_content, confidence, success, total_usage, metadata, sql_candidates
            
            logger.warning("Selected SQL has no valid execution result.")
            selected_sql, selected_content, _ = random.choice(all_sql_tuples)
            return selected_content, 0.0, False, total_usage, metadata, sql_candidates
        
        raw_responses_content = [r for r in raw_responses if r]
        if raw_responses_content:
            logger.warning("No SQL could be parsed from model responses.")
            metadata = {"stats": {"parse_fail": parse_fail_count}, "confidence": 0.0, "valid_proportion": 0.0, "total_candidates": 0, "candidate_count": 0}
            return random.choice(raw_responses_content), 0.0, False, total_usage, metadata, []
        
        logger.error("All LLM calls failed or produced empty content. Cannot generate SQL.")
        metadata = {"stats": {"parse_fail": parse_fail_count}, "confidence": 0.0, "valid_proportion": 0.0, "total_candidates": 0, "candidate_count": 0}
        return "Error: The language model failed to produce any output.", 0.0, False, total_usage, metadata, []
