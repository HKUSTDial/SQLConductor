"""
Implementation of MetaAgent in SQLConductor Reasoning System
"""

from loguru import logger
import time
from typing import Dict, Any, List, Optional, Tuple

from src.system.config import config
from src.system.llm_client import get_llm_client, LLMResponse
from src.system.workers import WorkerPool, WorkerType, _build_and_condense_schema_context
from src.system.workers.factory import WorkerFactory
from src.system.workers.base import BaseWorker
from src.system.prompt_factory import get_meta_agent_system_prompt, get_meta_agent_user_prompt, get_meta_agent_user_prompt_unrestricted
from src.database.schema import DatabaseSchema
from src.system.types import ReasoningStep, ReasoningSession, VALID_WORKER_TRANSITIONS, VALID_WORKER_TRANSITIONS_UNRESTRICTED, WorkerInput, WorkerResult, FALLBACK_PIPELINE, FALLBACK_WORKER_OVERRIDES, SQLCandidate
from src.system.tools import XMLOutputParser
from src.database.sql_execution import format_execution_result, cached_execute_sql_with_timeout, has_non_empty_result
from src.system.workers.utils import build_worker_input, update_session_state, VALID_SQL_THRESHOLD, CONSISTENCY_THRESHOLD, CONFIDENCE_THRESHOLD_UNRESTRICTED, select_best_from_candidates, select_best_from_candidates_with_pairwise


class MetaAgent:
    """Plan worker calls and maintain the reasoning session."""
    
    def __init__(self, max_turns: int = None):
        self.config = config.get_meta_agent_config()
        self.max_turns = max_turns or self.config.get("max_turns", 8)
        self.llm_client = get_llm_client(self.config["model_config"])
        self.worker_pool = WorkerPool()
        self.output_parser = XMLOutputParser()
        self.unrestricted = config.unrestricted
        self.worker_transitions = VALID_WORKER_TRANSITIONS_UNRESTRICTED if self.unrestricted else VALID_WORKER_TRANSITIONS
        
        all_tools = self.worker_pool.get_all_tools()
        worker_descriptions_lines = []
        for tool in all_tools:
            worker_descriptions_lines.append(f"*   **{tool.name}**: {tool.description}")
        worker_descriptions_block = "\n".join(worker_descriptions_lines)
        
        self.system_prompt = get_meta_agent_system_prompt(worker_descriptions=worker_descriptions_block)
        
        self.fallback_workers: Dict[WorkerType, BaseWorker] = {}
        
        logger.info(f"MetaAgent initialized: model={self.config['model_config'].model_name}, max_turns={self.max_turns}")
        if self.unrestricted:
            logger.info("MetaAgent running in unrestricted mode.")

    def _build_initial_schema(self, db_schema: DatabaseSchema) -> str:
        """Builds the initial schema context and condenses it if necessary."""
        return _build_and_condense_schema_context(db_schema.tables)

    def _update_usage(self, target_dict: Dict[str, int], new_usage: Optional[Dict[str, int]]):
        if not new_usage: return
        target_dict["prompt_tokens"] += new_usage.get("prompt_tokens", 0)
        target_dict["completion_tokens"] += new_usage.get("completion_tokens", 0)
        target_dict["total_tokens"] += new_usage.get("total_tokens", 0)

    def _get_last_successful_worker(self, steps: List[ReasoningStep]) -> Optional[WorkerType]:
        """Finds the last worker that executed successfully."""
        for step in reversed(steps):
            if step.worker_result and step.worker_result.success:
                return step.worker_result.worker_type
        return None

    def _build_prompt_for_turn(self, session: ReasoningSession, steps: List[ReasoningStep]) -> str:
        """Build a complete, structured prompt for the LLM from the current session state"""
        history_lines = []
        for step in steps:
            history_lines.append(f"Turn {step.turn}:")
            if step.worker_call:
                history_lines.append(f"  - Call: {step.worker_call}")
            if step.worker_result:
                if step.worker_result.success:
                    history_lines.append(f"  - Result: Success")
                else:
                    history_lines.append(f"  - Result: Error -> {step.worker_result.error_message}")
        history_block = "\n".join(history_lines) if history_lines else "No reasoning history yet."

        last_worker = self._get_last_successful_worker(steps)
        
        available_tools = self.worker_pool.get_available_tools(last_worker, transitions=self.worker_transitions)
        valid_next_workers_block = ""
        if available_tools:
            workers_list = "\n".join([f"- {tool.name}" for tool in available_tools])
            valid_next_workers_block = f"{workers_list}"

        intermediate_sql_result = None
        sql_to_execute = session.current_sql
        if sql_to_execute:
            exec_result = cached_execute_sql_with_timeout(session.db_path, sql_to_execute)
            exec_result_str = format_execution_result(exec_result)
            
            candidates_summary = ""
            candidates = session.refined_candidates or session.sql_candidates
            if candidates:
                total = len(candidates)
                valid = sum(1 for c in candidates if c.is_valid)
                consistency = session.sql_consistency_score
                if consistency is not None:
                    candidates_summary = (
                        f"SQL Constructor generated {total} SQL candidates ({valid} valid). "
                        f"Most consistent group: {consistency*100:.0f}%. "
                        f"The most consistent SQL is:\n"
                    )
                else:
                    candidates_summary = (
                        f"SQL Constructor generated {total} SQL candidates ({valid} valid). "
                        f"The most consistent SQL is:\n"
                    )
            
            intermediate_sql_result = f"{candidates_summary}{sql_to_execute}\n\nExecution Result:\n{exec_result_str}"

        prompt_fn = get_meta_agent_user_prompt_unrestricted if self.unrestricted else get_meta_agent_user_prompt
        user_prompt = prompt_fn(
            question=session.question,
            schema=session.schema,
            rephrased_question=session.rephrased_question,
            selected_schema_context=session.selected_schema_context,
            identified_column_values=session.identified_column_values,
            identified_column_functions=session.identified_column_functions,
            intermediate_sql_result=intermediate_sql_result,
            history=history_block,
            valid_next_workers_block=valid_next_workers_block
        )
        return user_prompt

    def needs_worker_call(self, llm_response: str) -> bool:
        """Check whether a worker should be called"""
        return self.output_parser.needs_tool_call(llm_response)
    
    def parse_worker_call(self, llm_response: str) -> Optional[Tuple[str, WorkerType]]:
        """Parse the worker call and return its name and type."""
        tool_call = self.output_parser.parse(llm_response)
        if tool_call:
            return tool_call.tool_name, tool_call.worker_type
        return None, None

    def _get_last_sql(self, session: ReasoningSession) -> Optional[str]:
        """Get the last generated SQL query from the session state"""
        return session.current_sql

    def _has_generated_sql(self, session: ReasoningSession) -> bool:
        """Check whether the SQL constructor has already succeeded"""
        return session.constructed_sql is not None

    def _has_valid_sql(self, session: ReasoningSession) -> bool:
        """Check whether a valid SQL statement exists (executed successfully, not timed out, and not empty)"""
        if not session.final_sql:
            return False
        exec_result = cached_execute_sql_with_timeout(session.db_path, session.final_sql)
        return has_non_empty_result(exec_result)

    async def _select_final_sql_from_candidates(self, session: ReasoningSession, db_schema: DatabaseSchema = None) -> Optional[str]:
        """Select from refined candidates when available, otherwise generated candidates.

        Use pairwise comparison when configured and a schema is available;
        otherwise use self-consistency.
        """
        candidates = session.refined_candidates or session.sql_candidates
        if not candidates:
            return None
        
        system_config = config.get_system_config()
        use_pairwise = system_config.get("use_pairwise_selection", True)
        
        if use_pairwise and db_schema is not None:
            logger.debug(f"[Final Selection] Using pairwise LLM comparison for SQL selection")
            best_sql, confidence, success = await select_best_from_candidates_with_pairwise(
                candidates, session, self.llm_client, db_schema, session.db_path, log_prefix="[Final Selection Pairwise] "
            )
        else:
            if use_pairwise and db_schema is None:
                logger.debug(f"[Final Selection] Pairwise selection requested but db_schema not available, using self-consistency")
            else:
                logger.debug(f"[Final Selection] Using self-consistency selection")
            best_sql, confidence, success = select_best_from_candidates(
                candidates, session.db_path, log_prefix="[Final Selection] "
            )
        
        if best_sql:
            logger.debug(f"[Final Selection] Selected SQL with confidence {confidence:.2f}")
        else:
            logger.warning("[Final Selection] No valid SQL found in candidates")
        
        return best_sql

    async def _refine_sql_candidates(
        self,
        session: ReasoningSession,
        db_schema: DatabaseSchema,
        steps: List[ReasoningStep]
    ) -> Tuple[Optional[WorkerResult], List[SQLCandidate]]:
        """
        Refine SQL candidates in parallel using the SQL Refiner worker's refine_candidates method.
        
        Args:
            session: Current reasoning session with sql_candidates populated
            db_schema: Database schema
            steps: List of reasoning steps for context
        
        Returns:
            Tuple of (refiner_result, refined_candidates)
        """
        if not session.sql_candidates:
            logger.warning("[Refine Candidates] No SQL candidates to refine")
            return None, []
        
        refiner_worker = self.worker_pool.get_worker(WorkerType.SQL_REFINER)
        if not refiner_worker:
            logger.error("[Refine Candidates] Could not get SQL Refiner worker")
            return None, []
        
        if not hasattr(refiner_worker, 'refine_candidates'):
            logger.warning("[Refine Candidates] SQL Refiner worker does not support refine_candidates")
            return await self._execute_worker(WorkerType.SQL_REFINER, session, db_schema, steps), []
        
        input_data = WorkerInput(
            question=session.current_question,
            schema=session.current_schema,
            context={
                "table_schema_dict": db_schema.tables,
                "db_path": session.db_path,
                "session": session,
                "steps": steps,
                "question_id": session.question_id
            }
        )
        
        logger.debug(f"[Refine Candidates] Refining {len(session.sql_candidates)} candidates in parallel")
        
        refined_candidates = await refiner_worker.refine_candidates(session.sql_candidates, input_data)
        
        session.refined_candidates = refined_candidates
        
        valid_count = sum(1 for c in refined_candidates if c.is_valid)
        logger.debug(
            f"[Refine Candidates] Completed: {len(refined_candidates)} refined "
            f"(valid: {valid_count}, invalid: {len(refined_candidates) - valid_count})"
        )
        
        best_sql, confidence, success = select_best_from_candidates(
            refined_candidates, session.db_path, log_prefix="[Refine Candidates] "
        )
        
        result = WorkerResult(
            worker_type=WorkerType.SQL_REFINER,
            result=f"<sql>{best_sql}</sql>" if best_sql else "",
            turn=refiner_worker.turn_count if hasattr(refiner_worker, 'turn_count') else 0,
            success=success,
            error_message=None if success else "No valid refined SQL",
            sql_candidates=refined_candidates,
            metadata={
                "refined_count": len(refined_candidates),
                "valid_count": valid_count,
                "confidence": confidence
            }
        )
        
        return result, refined_candidates

    async def _execute_fixed_pipeline(self, question: str, evidence: str, db_schema: DatabaseSchema, db_path: str, question_id: Optional[str] = None) -> Tuple[ReasoningSession, List[ReasoningStep]]:
        """Run the configured worker sequence with an independent reasoning session."""
        initial_schema = self._build_initial_schema(db_schema)
        fallback_session = ReasoningSession(
            question=question,
            schema=initial_schema,
            evidence=evidence,
            db_id=db_schema.db_id,
            db_path=db_path,
            question_id=question_id
        )
        
        fallback_steps: List[ReasoningStep] = []

        for turn, worker_type in enumerate(FALLBACK_PIPELINE, start=1):
            logger.debug(f"[Workflow] Turn {turn}: Executing {worker_type.value}")

            worker_override = None
            if worker_type in FALLBACK_WORKER_OVERRIDES:
                if worker_type not in self.fallback_workers:
                    try:
                        impl_name = FALLBACK_WORKER_OVERRIDES[worker_type]
                        self.fallback_workers[worker_type] = WorkerFactory.create_worker(worker_type, impl_name=impl_name)
                        logger.debug(f"Initialized {impl_name} worker: {worker_type.value}")
                    except Exception as e:
                        logger.error(f"Failed to initialize {impl_name} worker for {worker_type.value}: {e}")
                
                worker_override = self.fallback_workers.get(worker_type)

            worker_result = await self._execute_worker(worker_type, fallback_session, db_schema, [], worker_override=worker_override)
            self._update_session_state(fallback_session, worker_type, worker_result)

            if worker_result.usage:
                fallback_session.total_tokens += worker_result.usage.get("total_tokens", 0)
                self._update_usage(fallback_session.worker_usage, worker_result.usage)

            step = ReasoningStep(
                turn=turn,
                llm_response=f"<call>{worker_type.value}</call>",
                worker_call=worker_type.value,
                worker_result=worker_result
            )
            fallback_steps.append(step)

        final_sql = None
        if fallback_session.sql_candidates or fallback_session.refined_candidates:
            final_sql = await self._select_final_sql_from_candidates(fallback_session, db_schema)
        
        if not final_sql:
            final_sql = self._get_last_sql(fallback_session)
        
        if final_sql:
            fallback_session.final_sql = final_sql
            if self._has_valid_sql(fallback_session):
                fallback_session.success = True
                logger.debug(f"[Workflow] Successfully generated valid SQL.")
            else:
                logger.error(f"[Workflow] Generated SQL but execution failed.")
        else:
            logger.error("[Workflow] Failed to extract SQL from pipeline result.")

        return fallback_session, fallback_steps

    async def reason(self, question: str, schema: str, evidence: str, db_schema: DatabaseSchema, db_path: str, question_id: Optional[str] = None, retrieved_values: Optional[Dict[str, Dict[str, list]]] = None) -> Tuple[ReasoningSession, List[ReasoningStep]]:
        """Run inference"""
        start_time = time.time()

        initial_schema = self._build_initial_schema(db_schema)

        session = ReasoningSession(
            question=question,
            schema=initial_schema,
            evidence=evidence,
            db_id=db_schema.db_id,
            db_path=db_path,
            question_id=question_id,
            retrieved_values=retrieved_values
        )
        
        steps: List[ReasoningStep] = []
        max_turns_reached = False

        for turn in range(self.max_turns):
            logger.debug(f"Turn {turn + 1}/{self.max_turns}")
            
            user_prompt = self._build_prompt_for_turn(session, steps)
            llm_response = await self._call_llm_for_reasoning(user_prompt)

            step = ReasoningStep(
                turn=turn + 1,
                llm_response=llm_response.content,
                reasoning=llm_response.content
            )

            step_appended = False

            if self.needs_worker_call(llm_response.content):
                
                worker_name, worker_type = self.parse_worker_call(llm_response.content)
                if worker_name:
                    last_worker = self._get_last_successful_worker(steps)
                    valid_next_workers = self.worker_transitions.get(last_worker)

                    if valid_next_workers is not None and worker_type not in valid_next_workers:
                        error_msg = (
                            f"Invalid worker call '{worker_name}'. "
                            f"The last successful worker was '{last_worker.value if last_worker else 'None'}'. "
                            f"Valid next workers are: {[w.value for w in valid_next_workers]}."
                        )
                        logger.warning(error_msg)
                        step.worker_call = worker_name
                        step.worker_result = WorkerResult(
                            worker_type=worker_type, result="", turn=turn + 1, success=False,
                            error_message=error_msg
                        )
                        steps.append(step)
                        continue

                    if worker_type == WorkerType.END:
                        if not self._has_generated_sql(session):
                            logger.warning("No SQL at termination; requesting SQL construction.")
                            step.worker_call = WorkerType.SQL_CONSTRUCTOR.value
                            step.worker_result = WorkerResult(
                                worker_type=WorkerType.END, result="", turn=turn, success=False,
                                error_message="Call `SQL Constructor` before ending."
                            )
                            self._update_session_state(session, WorkerType.END, step.worker_result)
                            worker_type = WorkerType.SQL_CONSTRUCTOR
                        else:
                            final_sql = None
                            if session.sql_candidates or session.refined_candidates:
                                final_sql = await self._select_final_sql_from_candidates(session, db_schema)

                            if not final_sql:
                                final_sql = self._get_last_sql(session)

                            if final_sql:
                                step.worker_call = worker_name
                                if self.unrestricted and session.sql_candidates:
                                    last_constructor_metadata = next(
                                        (s.worker_result.metadata for s in reversed(steps)
                                         if s.worker_result and s.worker_result.worker_type == WorkerType.SQL_CONSTRUCTOR
                                         and s.worker_result.metadata),
                                        None
                                    )
                                    raw_confidence = (last_constructor_metadata or {}).get("raw_confidence", 1.0)
                                    if raw_confidence < CONFIDENCE_THRESHOLD_UNRESTRICTED:
                                        logger.debug(
                                            f"[SQL Refiner] raw_confidence={raw_confidence:.2f} < "
                                            f"{CONFIDENCE_THRESHOLD_UNRESTRICTED}. Triggering refinement before end."
                                        )
                                        steps.append(step)
                                        step_appended = True
                                        refiner_result, _ = await self._refine_sql_candidates(session, db_schema, steps)
                                        if refiner_result:
                                            self._update_session_state(session, WorkerType.SQL_REFINER, refiner_result)
                                            if refiner_result.usage:
                                                session.total_tokens += refiner_result.usage.get("total_tokens", 0)
                                                self._update_usage(session.worker_usage, refiner_result.usage)
                                            refiner_step = ReasoningStep(
                                                turn=turn + 1,
                                                llm_response=f"<call>{WorkerType.SQL_REFINER.value}</call>",
                                                worker_call=WorkerType.SQL_REFINER.value,
                                                worker_result=refiner_result,
                                            )
                                            steps.append(refiner_step)
                                            refined_sql = await self._select_final_sql_from_candidates(session, db_schema)
                                            if not refined_sql:
                                                refined_sql = self._get_last_sql(session)
                                            if refined_sql:
                                                final_sql = refined_sql

                                logger.debug("Terminating reasoning.")
                                session.final_sql = final_sql
                                session.success = True
                                if not step_appended:
                                    steps.append(step)
                                break
                            else:
                                logger.warning("No valid SQL at termination; requesting SQL construction.")
                                worker_type = WorkerType.SQL_CONSTRUCTOR
                                step.worker_call = WorkerType.SQL_CONSTRUCTOR.value

                    step.worker_call = worker_name if not step.worker_call else step.worker_call
                    logger.debug(f"Worker call: {worker_name}")
                    
                    worker_result = await self._execute_worker(worker_type, session, db_schema, steps)
                    step.worker_result = worker_result
                    
                    self._update_session_state(session, worker_type, worker_result)
                    
                    if worker_result.usage:
                        session.total_tokens += worker_result.usage.get("total_tokens", 0)
                        self._update_usage(session.worker_usage, worker_result.usage)

                    if (not self.unrestricted) and (worker_type == WorkerType.SQL_CONSTRUCTOR) and worker_result.metadata:
                        metadata = worker_result.metadata
                        valid_proportion = metadata.get("valid_proportion", 0.0)
                        raw_confidence = metadata.get("raw_confidence", 0.0)

                        needs_refinement = (
                            valid_proportion < VALID_SQL_THRESHOLD or
                            raw_confidence < CONSISTENCY_THRESHOLD
                        )

                        if needs_refinement:
                            logger.debug(f"[SQL Refiner] SQL quality below threshold (valid={valid_proportion:.2f}, raw_confidence={raw_confidence:.2f}).")
                            
                            steps.append(step)
                            step_appended = True
                            
                            if session.sql_candidates:
                                logger.debug(f"[SQL Refiner] Using parallel candidate refinement for {len(session.sql_candidates)} candidates")
                                refiner_result, refined_candidates = await self._refine_sql_candidates(session, db_schema, steps)
                                if refiner_result:
                                    self._update_session_state(session, WorkerType.SQL_REFINER, refiner_result)
                                    if refiner_result.usage:
                                        session.total_tokens += refiner_result.usage.get("total_tokens", 0)
                                        self._update_usage(session.worker_usage, refiner_result.usage)
                            else:
                                logger.warning("[SQL Refiner] No SQL candidates available, using regular refiner")
                                refiner_result = await self._execute_worker(WorkerType.SQL_REFINER, session, db_schema, steps)
                                self._update_session_state(session, WorkerType.SQL_REFINER, refiner_result)
                                if refiner_result.usage:
                                    session.total_tokens += refiner_result.usage.get("total_tokens", 0)
                                    self._update_usage(session.worker_usage, refiner_result.usage)
                            
                            refiner_step = ReasoningStep(
                                turn=turn + 1,
                                llm_response=f"<call>{WorkerType.SQL_REFINER.value}</call>",
                                worker_call=WorkerType.SQL_REFINER.value,
                                worker_result=refiner_result
                            )
                            steps.append(refiner_step)
                            
                            refined_sql = None
                            if session.sql_candidates or session.refined_candidates:
                                refined_sql = await self._select_final_sql_from_candidates(session, db_schema)
                            if not refined_sql:
                                refined_sql = self._get_last_sql(session)
                            if refined_sql:
                                logger.debug(f"SQL refinement completed; ending workflow (success={refiner_result.success if refiner_result else False}).")
                                session.final_sql = refined_sql
                                session.success = refiner_result.success if refiner_result else False
                                break
                            else:
                                logger.warning("SQL refinement returned no SQL; continuing.")

                    # Restricted mode ends after SQL refinement; unrestricted mode follows the policy.
                    if not self.unrestricted and worker_type == WorkerType.SQL_REFINER:
                        refined_sql = None
                        if session.sql_candidates or session.refined_candidates:
                            refined_sql = await self._select_final_sql_from_candidates(session, db_schema)
                        if not refined_sql:
                            refined_sql = self._get_last_sql(session)
                        if refined_sql:
                            logger.debug(f"SQL refinement completed; ending workflow (success={worker_result.success}).")
                            session.final_sql = refined_sql
                            session.success = worker_result.success
                            step.worker_call = worker_name
                            steps.append(step)
                            break
                    
                else:
                    logger.warning("Failed to parse worker call. Terminating.")
                    step.worker_call = None
                    steps.append(step)
                    break
            
            if llm_response.usage:
                p_tokens = llm_response.usage.get("prompt_tokens", 0)
                c_tokens = llm_response.usage.get("completion_tokens", 0)
                session.meta_agent_usage["prompt_tokens"] += p_tokens
                session.meta_agent_usage["completion_tokens"] += c_tokens
                session.meta_agent_usage["total_tokens"] = (
                    session.meta_agent_usage["prompt_tokens"] +
                    session.meta_agent_usage["completion_tokens"]
                )
                session.total_tokens += p_tokens + c_tokens

            if not step_appended:
                steps.append(step)
        
        else:
            if self.unrestricted:
                logger.warning(
                    f"Turn limit ({self.max_turns}) reached; generating and refining final SQL."
                )
                forced_turn = self.max_turns + 1

                if not self._has_generated_sql(session):
                    worker_result = await self._execute_worker(WorkerType.SQL_CONSTRUCTOR, session, db_schema, steps)
                    self._update_session_state(session, WorkerType.SQL_CONSTRUCTOR, worker_result)
                    if worker_result.usage:
                        session.total_tokens += worker_result.usage.get("total_tokens", 0)
                        self._update_usage(session.worker_usage, worker_result.usage)
                    steps.append(ReasoningStep(
                        turn=forced_turn,
                        llm_response=f"<call>{WorkerType.SQL_CONSTRUCTOR.value}</call>",
                        worker_call=WorkerType.SQL_CONSTRUCTOR.value,
                        worker_result=worker_result,
                        exclude_from_training=True,
                    ))
                    forced_turn += 1

                if session.sql_candidates:
                    refiner_result, _ = await self._refine_sql_candidates(session, db_schema, steps)
                    if refiner_result:
                        self._update_session_state(session, WorkerType.SQL_REFINER, refiner_result)
                        if refiner_result.usage:
                            session.total_tokens += refiner_result.usage.get("total_tokens", 0)
                            self._update_usage(session.worker_usage, refiner_result.usage)
                        steps.append(ReasoningStep(
                            turn=forced_turn,
                            llm_response=f"<call>{WorkerType.SQL_REFINER.value}</call>",
                            worker_call=WorkerType.SQL_REFINER.value,
                            worker_result=refiner_result,
                            exclude_from_training=True,
                        ))

                final_sql = None
                if session.sql_candidates or session.refined_candidates:
                    final_sql = await self._select_final_sql_from_candidates(session, db_schema)
                if not final_sql:
                    final_sql = self._get_last_sql(session)
                if final_sql:
                    session.final_sql = final_sql

                if not self._has_valid_sql(session):
                    max_turns_reached = True
            else:
                logger.warning(f"Max turns ({self.max_turns}) reached. Starting another workflow.")
                max_turns_reached = True

        if not session.final_sql:
            potential_sql = self._get_last_sql(session)
            if potential_sql:
                session.final_sql = potential_sql

        needs_fallback = False
        fallback_reason = None

        if max_turns_reached:
            needs_fallback = True
            fallback_reason = "Max turns reached"
        elif not session.final_sql:
            needs_fallback = True
            fallback_reason = "No SQL generated"
        elif not self._has_valid_sql(session):
            needs_fallback = True
            fallback_reason = "SQL execution failed (error/timeout/empty)"

        if not needs_fallback and steps:
            last_step = steps[-1]
            if last_step.worker_call is None and self.needs_worker_call(last_step.llm_response):
                needs_fallback = True
                fallback_reason = "Failed to parse worker call, terminating early"

        if needs_fallback:
            logger.debug(f"[Workflow] Triggered: {fallback_reason}")
            fallback_session, fallback_steps = await self._execute_fixed_pipeline(
                question=session.question,
                evidence=session.evidence,
                db_schema=db_schema,
                db_path=session.db_path,
                question_id=session.question_id
            )
            session = fallback_session
            steps = fallback_steps

            session.fallback_triggered = True
            session.fallback_reason = fallback_reason

        session.execution_time = time.time() - start_time
        return session, steps
    
    def _update_session_state(self, session: ReasoningSession, worker_type: WorkerType, result: WorkerResult):
        """Update the session from the worker result."""
        update_session_state(session, worker_type, result)

    async def _call_llm_for_reasoning(self, user_prompt: str) -> LLMResponse:
        """Call the policy and estimate missing prompt usage for failed responses."""
        response = await self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=self.system_prompt,
            temperature=self.config.get("temperature", 0.1),
            max_tokens=self.config.get("max_tokens", 2048)
        )
        if response.error and response.usage is None:
            system_len = len(self.system_prompt) if self.system_prompt else 0
            estimated_prompt_tokens = (system_len + len(user_prompt)) // 4
            logger.warning(
                f"[Meta Agent] LLM call failed: {response.error}. "
                f"Estimated prompt_tokens={estimated_prompt_tokens}."
            )
            response.usage = {
                "prompt_tokens": estimated_prompt_tokens,
                "completion_tokens": 0,
                "total_tokens": estimated_prompt_tokens,
            }
        return response
    
    async def _execute_worker(self, worker_type: WorkerType, session: ReasoningSession, db_schema: DatabaseSchema, steps: List[ReasoningStep], worker_override: Optional[BaseWorker] = None) -> WorkerResult:
        """Execute the worker"""
        input_data = build_worker_input(session, db_schema.tables, steps)

        if worker_override:
            return await worker_override.execute(input_data)
        
        return await self.worker_pool.execute_worker(worker_type, input_data)
    
    def update_worker_model(self, worker_type: WorkerType, new_port: int):
        """Update worker model ports"""
        self.worker_pool.update_worker_model(worker_type, new_port)
    
    def get_reasoning_summary(self, session: ReasoningSession) -> Dict[str, Any]:
        """Get the inference summary"""
        return {
            "question": session.question,
            "success": session.success,
            "final_sql": session.final_sql,
            "total_turns": len(session.steps),
            "total_tokens": session.total_tokens,
            "steps": [
                {
                    "turn": step.turn,
                    "worker_call": step.worker_call,
                    "worker_success": step.worker_result.success if step.worker_result else None,
                    "has_sql": bool(step.worker_result and "sql" in step.worker_result.result.lower()) if step.worker_result else False
                }
                for step in session.steps
            ]
        }
