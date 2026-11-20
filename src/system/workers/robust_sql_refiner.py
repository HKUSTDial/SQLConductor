"""
Robust SQL Refiner Worker with multiple checkers
"""
import asyncio
from typing import Dict, Optional, Tuple, List
from loguru import logger

from src.database.sql_execution import (
    cached_execute_sql_with_timeout,
    is_valid_execution_result,
    has_non_empty_result,
    SQLExecutionResultType
)
from src.system.types import WorkerType, WorkerInput, WorkerResult, SQLCandidate
from src.system.workers.base import BaseWorker
from src.system.workers.utils import _build_hint_from_steps, _extract_sql_query_answer, select_best_from_candidates
from src.system.workers.checkers import (
    SyntaxChecker,
    ResultChecker,
    JoinChecker,
    SelectChecker,
    MaxMinChecker,
    OrderByLimitChecker,
    OrderByNullChecker,
    TimeChecker
)


REFINER_SAMPLING_BUDGET = 5


class RobustSQLRefinerWorker(BaseWorker):
    """Refine SQL with syntax, result, and clause-specific checkers."""

    def __init__(self, worker_type: WorkerType):
        super().__init__(worker_type)
        self.description = "Use *only* after `SQL Constructor`. Deploys multiple specialized checkers to identify and fix potential SQL issues. If the generated SQL is invalid, returns no results, or is logically incorrect based on execution feedback, use this worker."
        
        self.checkers = [
            SyntaxChecker(),
            JoinChecker(),
            OrderByLimitChecker(),
            TimeChecker(),
            SelectChecker(),
            MaxMinChecker(),
            OrderByNullChecker(),
            ResultChecker(),
        ]
        
        logger.debug(
            f"Initialized {self.__class__.__name__} with {len(self.checkers)} checkers: "
            f"{', '.join([checker.__class__.__name__ for checker in self.checkers])}"
        )
    
    async def execute(self, input_data: WorkerInput) -> WorkerResult:
        """
        Execute the SQL refinement process using all checkers.
        """
        self.turn_count += 1
        
        try:
            db_path = input_data.context.get("db_path")
            if not db_path:
                raise ValueError("db_path not found in WorkerInput context for RobustSQLRefinerWorker.")
            
            session = input_data.context.get("session")
            if session and hasattr(session, 'sql_candidates') and session.sql_candidates:
                logger.debug(f"[{self.worker_type.value}] Refining {len(session.sql_candidates)} candidates")
                
                refined_candidates = await self.refine_candidates(session.sql_candidates, input_data)
                
                best_sql, confidence, success = select_best_from_candidates(
                    refined_candidates, db_path, log_prefix=f"[{self.worker_type.value}] "
                )
                
                return WorkerResult(
                    worker_type=self.worker_type,
                    result=f"<result>{best_sql}</result>" if best_sql else "",
                    turn=self.turn_count,
                    success=success,
                    error_message=None if success else "No valid refined SQL",
                    sql_candidates=refined_candidates,
                    metadata={
                        "refined_count": len(refined_candidates),
                        "valid_count": sum(1 for c in refined_candidates if c.is_valid),
                        "confidence": confidence
                    }
                )
            
            sql = self._extract_sql_from_context(input_data)
            if not sql:
                logger.warning("No SQL found in context to refine")
                return WorkerResult(
                    worker_type=self.worker_type,
                    result="",
                    turn=self.turn_count,
                    success=False,
                    error_message="No SQL found in context to refine.",
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                )
            
            steps = input_data.context.get("steps", [])
            hint = _build_hint_from_steps(steps, input_data.context)
            question = input_data.question
            schema = input_data.schema
            
            current_sql = sql
            total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
            for checker in self.checkers:
                checker_name = checker.__class__.__name__
                try:
                    revised_sql, checker_usage = await checker.check_and_revise(
                        sql=current_sql,
                        question=question,
                        schema=schema,
                        hint=hint,
                        db_path=db_path,
                        llm_client=self.llm_client,
                        sampling_budget=REFINER_SAMPLING_BUDGET
                    )
                    
                    total_usage["prompt_tokens"] += checker_usage.get("prompt_tokens", 0)
                    total_usage["completion_tokens"] += checker_usage.get("completion_tokens", 0)
                    total_usage["total_tokens"] += checker_usage.get("total_tokens", 0)
                    
                    if revised_sql != current_sql:
                        current_sql = revised_sql
                    
                except Exception as e:
                    logger.error(f"[RobustSQLRefiner] Error in {checker_name}: {e}")
                    continue
            
            final_result = cached_execute_sql_with_timeout(db_path, current_sql)
            success = has_non_empty_result(final_result)
            
            if success:
                logger.debug("[RobustSQLRefiner] Refinement returned a non-empty result.")
            else:
                logger.debug(
                    f"[RobustSQLRefiner] Refined SQL still has issues. "
                    f"Result type: {final_result.result_type.value}, "
                    f"Error: {final_result.error_message or 'N/A'}"
                )
            
            result_content = f"<result>{current_sql}</result>"
            
            return WorkerResult(
                worker_type=self.worker_type,
                result=result_content,
                turn=self.turn_count,
                success=success,
                error_message=None if success else "Refined SQL still has execution issues.",
                usage=total_usage
            )
            
        except Exception as e:
            logger.error(f"Worker {self.worker_type.value} execution failed: {e}", exc_info=True)
            return WorkerResult(
                worker_type=self.worker_type,
                result="",
                turn=self.turn_count,
                success=False,
                error_message=str(e),
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )
    
    def _extract_sql_from_context(self, input_data: WorkerInput) -> Optional[str]:
        """Extract the latest SQL query from the session or worker context."""
        steps = input_data.context.get("steps", [])
        
        # Check both successful and failed results (failed results may still contain SQL)
        for step in reversed(steps):
            if step.worker_result and step.worker_result.result:
                worker_type = step.worker_result.worker_type
                if worker_type in [WorkerType.SQL_CONSTRUCTOR, WorkerType.SQL_REFINER]:
                    result = step.worker_result.result
                    if result:
                        sql = _extract_sql_query_answer(result)
                        if sql:
                            return sql
        
        session = input_data.context.get("session")
        if session:
            if hasattr(session, 'constructed_sql') and session.constructed_sql:
                return session.constructed_sql
            if hasattr(session, 'current_sql') and session.current_sql:
                return session.current_sql
        
        current_sql = input_data.context.get("current_sql")
        if current_sql:
            return current_sql
        
        return None

    def _normalize_sql(self, sql: str) -> str:
        """Normalize whitespace and case for candidate comparison."""
        if not sql:
            return ""
        return " ".join(sql.split()).strip().lower()

    async def refine_candidates(
        self,
        candidates: List[SQLCandidate],
        input_data: WorkerInput,
    ) -> List[SQLCandidate]:
        """
        Refine unique SQL candidates in parallel and preserve their frequencies.
        
        Args:
            candidates: List of SQLCandidate objects to refine
            input_data: WorkerInput containing question, schema, context
        
        Returns:
            List of refined SQLCandidate objects
        """
        if not candidates:
            return []
        
        db_path = input_data.context.get("db_path")
        if not db_path:
            logger.warning("db_path not found in context for refine_candidates")
            return candidates
            
        unique_candidates_map = {}
        for candidate in candidates:
            norm_sql = self._normalize_sql(candidate.sql)
            if norm_sql not in unique_candidates_map:
                unique_candidates_map[norm_sql] = candidate.sql
        
        async def refine_single_sql(sql: str) -> Tuple[str, bool]:
            """Refine a single SQL string using all checkers."""
            try:
                steps = input_data.context.get("steps", [])
                hint = _build_hint_from_steps(steps, input_data.context)
                question = input_data.question
                schema = input_data.schema
                
                current_sql = sql
                
                for checker in self.checkers:
                    try:
                        revised_sql, _ = await checker.check_and_revise(
                            sql=current_sql,
                            question=question,
                            schema=schema,
                            hint=hint,
                            db_path=db_path,
                            llm_client=self.llm_client,
                            sampling_budget=REFINER_SAMPLING_BUDGET
                        )
                        current_sql = revised_sql
                    except Exception as e:
                        logger.debug(f"Error in checker {checker.__class__.__name__}: {e}")
                        continue
                
                exec_result = cached_execute_sql_with_timeout(db_path, current_sql)
                is_valid = is_valid_execution_result(exec_result)
                
                return current_sql, is_valid
                
            except Exception as e:
                logger.debug(f"Error refining candidate sql: {e}")
                return sql, False
                
        unique_norms = list(unique_candidates_map.keys())
        tasks = [refine_single_sql(unique_candidates_map[norm]) for norm in unique_norms]
        
        refined_results = await asyncio.gather(*tasks)
        
        norm_to_result = {}
        for norm, result in zip(unique_norms, refined_results):
            norm_to_result[norm] = result
            
        # Reconstruct the full list of candidates (preserving frequency distribution)
        final_refined_candidates = []
        for candidate in candidates:
            norm = self._normalize_sql(candidate.sql)
            revised_sql, is_valid = norm_to_result.get(norm, (candidate.sql, False))
            
            final_refined_candidates.append(SQLCandidate(
                sql=revised_sql,
                raw_content=f"<result>{revised_sql}</result>",
                strategy=f"{candidate.strategy}_robust_refined" if candidate.strategy else "robust_refined",
                is_valid=is_valid,
                execution_time=None
            ))
        
        valid_count = sum(1 for c in final_refined_candidates if c.is_valid)
        logger.debug(
            f"[{self.worker_type.value}] Refined {len(final_refined_candidates)} candidates "
            f"(valid: {valid_count})"
        )
        
        return final_refined_candidates
