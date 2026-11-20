import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger
from tqdm import tqdm
import multiprocessing as mp

from src.system.types import (
    ReasoningSession, ReasoningStep, WorkerType, WorkerResult, 
    VALID_WORKER_TRANSITIONS_UNRESTRICTED, ActionName, ACTION_WORKER_MAP
)
from src.system.prompt_factory import (
    get_meta_agent_user_prompt_unrestricted, get_meta_agent_system_prompt, get_prompt
)
from src.system.workers import WorkerPool
from src.system.workers.utils import update_session_state
from src.database.sql_execution import (
    format_execution_result, cached_execute_sql_with_timeout
)
from src.database.utils import build_table_profile_statement
from src.system.workers import _build_and_condense_schema_context
from src.llm_call.openai_llm import call_openai

from src.mcts.mcts_action import (
    RaphraseQuestionAction, SchemaSelectionAction, 
    IdentifyColumnValuesAction, IdentifyColumnFunctionsAction, 
    SQLGenerationAction, SQLRevisionAction, EndAction
)
from src.mcts.serialization import load_trajectories

_WORKER_DESCRIPTIONS_CACHE: Optional[str] = None

def _get_worker_descriptions() -> str:
    """Return the cached worker descriptions."""
    global _WORKER_DESCRIPTIONS_CACHE
    if _WORKER_DESCRIPTIONS_CACHE is None:
        logger.disable("src.system.workers")
        try:
            worker_pool = WorkerPool()
            all_tools = worker_pool.get_all_tools()
            worker_descriptions_lines = []
            for tool in all_tools:
                worker_descriptions_lines.append(f"*   **{tool.name}**: {tool.description}")
            _WORKER_DESCRIPTIONS_CACHE = "\n".join(worker_descriptions_lines)
        finally:
            logger.enable("src.system.workers")
    return _WORKER_DESCRIPTIONS_CACHE

# Worker function for multiprocessing - must be at module level to be picklable
def _process_pkl_file_reasoning_worker(args):
    """Process a trajectory file in a worker process."""
    pkl_path, summary, llm_kwargs = args
    
    processor = StatefulTrajectoryProcessor(
        model_path=llm_kwargs["model"],
        base_url=llm_kwargs.get("base_url"),
        api_key=llm_kwargs.get("api_key"),
        temperature=llm_kwargs.get("temperature", 0.7),
        top_p=llm_kwargs.get("top_p", 0.9),
        max_tokens=llm_kwargs.get("max_tokens", 1000)
    )
    
    return processor.process_pkl_file_reasoning(Path(pkl_path), summary)

class StatefulTrajectoryProcessor:
    """Generate SFT examples by replaying MCTS trajectories."""
    
    def __init__(self, model_path: str, base_url: str = None, api_key: str = None, 
                 temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 1000, 
                 cache_dir: Optional[Path] = None):
        self.llm_kwargs = {
            "model": model_path,
            "base_url": base_url,
            "api_key": api_key,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": 1,
            "n_strategy": "single"
        }
        self.cache_dir = cache_dir

    def _get_worker_type(self, action: Any) -> Optional[WorkerType]:
        """Maps MCTS Action to WorkerType"""
        action_name_str = action.__class__.__name__
        for action_enum in ActionName:
            if action_enum.value == action_name_str:
                return ACTION_WORKER_MAP.get(action_enum)
        
        if isinstance(action, EndAction):
            return WorkerType.END
            
        return None

    def _get_action_result_string(self, action: Any, node: Any, db_path: Optional[Path]) -> str:
        """Extract the worker result from a trajectory node."""
        if isinstance(action, RaphraseQuestionAction):
            return node.rephrased_question
        elif isinstance(action, SchemaSelectionAction):
            selected_schema_dict = getattr(node, 'selected_schema_dict', None)
            if selected_schema_dict:
                ddl_list = [
                    build_table_profile_statement(
                        selected_schema_dict[t].to_dict(),
                        add_value_description=True,
                        add_column_description=True,
                        add_value_examples=True,
                        add_expanded_column_name=True,
                    )
                    for t in selected_schema_dict
                ]
                return "\n".join(ddl_list)
            return node.selected_schema_context
        elif isinstance(action, IdentifyColumnValuesAction):
            return node.identified_column_values
        elif isinstance(action, IdentifyColumnFunctionsAction):
            return node.identified_column_functions
        elif isinstance(action, SQLGenerationAction):
            return node.sql_query
        elif isinstance(action, SQLRevisionAction):
            return node.revised_sql_query
        elif isinstance(action, EndAction):
            return ""
        return ""

    def _get_last_successful_worker(self, steps: List[ReasoningStep]) -> Optional[WorkerType]:
        """Finds the last worker that executed successfully."""
        for step in reversed(steps):
            if step.worker_result and step.worker_result.success:
                return step.worker_result.worker_type
        return None

    def _build_prompt_for_turn(self, session: ReasoningSession, steps: List[ReasoningStep]) -> str:
        """Build the policy prompt from the trajectory state."""
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
        
        valid_next_worker_types = VALID_WORKER_TRANSITIONS_UNRESTRICTED.get(last_worker, [])
        valid_next_workers_block = ""
        if valid_next_worker_types:
            workers_list = "\n".join([f"- {wt.value}" for wt in valid_next_worker_types])
            valid_next_workers_block = f"{workers_list}"

        intermediate_sql_result = None
        sql_to_execute = session.current_sql
        if sql_to_execute:
            try:
                exec_result = cached_execute_sql_with_timeout(session.db_path, sql_to_execute)
                exec_result_str = format_execution_result(exec_result)
                candidates_summary = ""
                candidates = session.refined_candidates or session.sql_candidates
                if candidates:
                    total = len(candidates)
                    valid = sum(1 for candidate in candidates if candidate.is_valid)
                    candidates_summary = f"SQL Constructor generated {total} SQL candidates ({valid} valid). "
                    if session.sql_consistency_score is not None:
                        candidates_summary += f"Most consistent group: {session.sql_consistency_score*100:.0f}%. "
                    candidates_summary += "The most consistent SQL is:\n"
                intermediate_sql_result = f"{candidates_summary}{sql_to_execute}\n\nExecution Result:\n{exec_result_str}"
            except Exception as e:
                logger.warning(f"Failed to execute intermediate SQL: {e}")
                intermediate_sql_result = f"Error executing SQL: {e}"

        user_prompt = get_meta_agent_user_prompt_unrestricted(
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

    def _update_session_state(self, session: ReasoningSession, worker_type: WorkerType, result: WorkerResult):
        """Updates session state based on worker result."""
        update_session_state(session, worker_type, result)

    def _generate_reasoning(self, system_prompt: str, user_prompt: str, action: str, result: str) -> str:
        """Generates reasoning for the step using LLM."""
        prompt_template = get_prompt(
            "stateful_step_reasoning",
            META_AGENT_SYSTEM_PROMPT=system_prompt,
            META_AGENT_USER_PROMPT=user_prompt,
            ACTION=action,
            RESULT=result
        )
        
        try:
            response = call_openai(prompt=prompt_template, cost_recorder=None, **self.llm_kwargs)
            reasoning = response[0].strip()
            
            reasoning = reasoning.replace("<reasoning>", "").replace("</reasoning>", "").strip()
            reasoning = reasoning.replace("<call>", "").replace("</call>", "").strip()
            reasoning = re.sub(r"<think>.*?</think>", "", reasoning, flags=re.DOTALL).strip()
            
            return f"<reasoning>\n{reasoning}\n</reasoning>"
        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            return "<reasoning>Error generating reasoning.</reasoning>"

    def process_trajectory(self, trajectory_nodes: List[Any], question: str, 
                           initial_schema: str, db_path: Path) -> List[Dict[str, Any]]:
        """
        Process a single trajectory to generate SFT training examples.
        Each step in the trajectory becomes a training example.
        """
        session = ReasoningSession(
            question=question,
            schema=initial_schema,
            db_id=db_path.parent.name,
            db_path=str(db_path)
        )
        
        steps: List[ReasoningStep] = []
        training_examples = []
        
        for i, node in enumerate(trajectory_nodes[1:]):
            if not hasattr(node, 'parent_action'):
                continue
            
            action = node.parent_action
            worker_type = self._get_worker_type(action)
            
            if not worker_type:
                logger.warning(f"Unknown worker type for action {action}")
                continue

            action_name = worker_type.value
            result_str = self._get_action_result_string(action, node, db_path)
            
            
            user_prompt = self._build_prompt_for_turn(session, steps)
            worker_descriptions = _get_worker_descriptions()
            system_prompt = get_meta_agent_system_prompt(worker_descriptions=worker_descriptions)
            
            reasoning = self._generate_reasoning(system_prompt, user_prompt, action_name, result_str)
            
            target_response = f"{reasoning}\n<call>{action_name}</call>"
            
            training_examples.append({
                "turn": i + 1,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "target_response": target_response,
                "worker_result": result_str,
                "reasoning": reasoning,
                "action": action_name
            })
            
            worker_result = getattr(node, "worker_result", None) or WorkerResult(
                worker_type=worker_type,
                result=result_str,
                turn=i + 1,
                success=True
            )
            
            step = ReasoningStep(
                turn=i + 1,
                llm_response=target_response,
                worker_call=action_name,
                worker_result=worker_result,
                reasoning=reasoning
            )
            steps.append(step)
            
            self._update_session_state(session, worker_type, worker_result)
            
        return training_examples

    def process_pkl_file_reasoning(self, pkl_path: Path, summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate reasoning for all selected trajectories based on pre-computed summary."""
        selected_trajectories = summary.get("selected_trajectories", [])
        if not selected_trajectories:
            legacy_idx = summary.get("selected_trajectory_index")
            if legacy_idx is not None:
                selected_trajectories = [{"index": legacy_idx, "rank": 1, "metadata": {}}]

        if not summary.get("has_valid_trajectory") or not selected_trajectories:
            return None
            
        try:
            with open(pkl_path, 'rb') as f:
                data = load_trajectories(f)
            
            first_node = data[0][0]
            question = summary["question"]
            table_schema_dict_node = getattr(first_node, 'table_schema_dict', None)
            if table_schema_dict_node:
                initial_schema = _build_and_condense_schema_context(table_schema_dict_node)
            else:
                initial_schema = getattr(first_node, 'schema_context', "")
            
            db_root_dir = getattr(first_node, 'db_root_dir', "")
            db_id = summary["db_id"]
            db_path = Path(db_root_dir) / db_id / f"{db_id}.sqlite"
            
            trajectories_result = []
            for selected in selected_trajectories:
                traj_idx = selected["index"]
                rank = selected.get("rank", 1)
                metadata = selected.get("metadata", {})

                trajectory = data[traj_idx]
                steps = self.process_trajectory(trajectory, question, initial_schema, db_path)

                trajectories_result.append({
                    "trajectory_index": traj_idx,
                    "rank": rank,
                    "metadata": metadata,
                    "steps": steps,
                })

            return {
                "question_id": summary["question_id"],
                "db_id": db_id,
                "question": question,
                "trajectories": trajectories_result,
            }
        except Exception as e:
            logger.error(f"Error generating reasoning for {pkl_path}: {e}")
            return None

    def process_dataset(self, input_dir: Path, output_file: Path, ref_data_path: Path, n_processes: int = 1):
        """Process all PKL files in directory to generate reasoning (requires summary file)."""
        input_dir = Path(input_dir)
        pkl_files = sorted(list(input_dir.glob("*.pkl")))
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file = output_file.with_name(f"{output_file.stem}_summary.json")
        
        logger.info("Starting Reasoning Phase...")
        
        if not summary_file.exists():
            logger.error(f"Summary file not found: {summary_file}. Please run summary generation first.")
            return

        with open(summary_file, 'r') as f:
            summaries = json.load(f)
        
        summary_map = {s["question_id"]: s for s in summaries if "question_id" in s}
        
        existing_data = []
        processed_data_ids = set()
        if output_file.exists():
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
            processed_data_ids = {item["question_id"] for item in existing_data}
            logger.info(f"Loaded {len(existing_data)} existing data items.")

        tasks = []
        for pkl_file in pkl_files:
            qid = int(pkl_file.stem)
            if qid in summary_map and qid not in processed_data_ids:
                summ = summary_map[qid]
                has_selected = bool(summ.get("selected_trajectories")) or (
                    summ.get("selected_trajectory_index") is not None
                )
                if summ.get("has_valid_trajectory") and has_selected:
                    tasks.append((pkl_file, summ))
        
        logger.info(f"Generating reasoning for {len(tasks)} tasks.")
        
        if tasks:
            batch_size = 10
            processed_count = 0
            new_data = []
            
            with tqdm(total=len(tasks), desc="Generating Reasoning") as pbar:
                if n_processes > 1:
                    worker_args = [
                        (str(pkl_file), summ, self.llm_kwargs)
                        for pkl_file, summ in tasks
                    ]
                    
                    with mp.Pool(n_processes) as pool:
                        for task_data in pool.imap(_process_pkl_file_reasoning_worker, worker_args):
                            if task_data:
                                new_data.append(task_data)
                            
                            processed_count += 1
                            if processed_count % batch_size == 0:
                                full_data = existing_data + new_data
                                full_data.sort(key=lambda x: x["question_id"])
                                with open(output_file, 'w') as f:
                                    json.dump(full_data, f, indent=2, ensure_ascii=False)
                            pbar.update(1)
                else:
                    for pkl_file, summ in tasks:
                        task_data = self.process_pkl_file_reasoning(pkl_file, summ)
                        if task_data:
                            new_data.append(task_data)
                        
                        processed_count += 1
                        if processed_count % batch_size == 0:
                            full_data = existing_data + new_data
                            full_data.sort(key=lambda x: x["question_id"])
                            with open(output_file, 'w') as f:
                                json.dump(full_data, f, indent=2, ensure_ascii=False)
                        pbar.update(1)
            
            full_data = existing_data + new_data
            full_data.sort(key=lambda x: x["question_id"])
            with open(output_file, 'w') as f:
                json.dump(full_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Reasoning phase complete. Saved to {output_file}")
        else:
            logger.info("No new tasks to process.")
