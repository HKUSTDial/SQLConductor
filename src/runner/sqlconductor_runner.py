#!/usr/bin/env python3
"""Run SQLConductor inference on preprocessed tasks."""

import asyncio
import argparse
from loguru import logger
import sys
from pathlib import Path
import json
from typing import Optional
import pickle
import concurrent.futures
from tqdm import tqdm

from src.system.reasoning_engine import get_reasoning_engine
from src.system.config import config
from src.database.database_manager import DatabaseManager
from src.system.types import TaskResult
from src.eval.compile_eval_json import compile_results


# Worker function for multiprocessing. Must be at the top level.
def _process_task_worker(task_with_index):
    """
    A synchronous wrapper that processes a single task in a separate process.
    It sets up its own asyncio event loop to run the async reasoning.
    """
    i, task = task_with_index
    try:
        # Each process needs its own reasoning engine instance.
        engine = get_reasoning_engine()
        question = task.question
        evidence = getattr(task, 'evidence', '')
        db_id = getattr(task, 'db_id', 'N/A')
        question_id = getattr(task, 'question_id', i)

        with logger.contextualize(question_id=question_id, db_id=db_id):
            database_root_dir = config.config_data.get('system', {}).get('data', {}).get('database_root_dir')
            if not database_root_dir:
                raise ValueError("'database_root_dir' is not set in config (system.data.database_root_dir). Please specify it in your config file.")
            db_schema = DatabaseManager.get_database_schema(db_id, database_root_dir)
            db_path = Path(database_root_dir) / db_id / f"{db_id}.sqlite"
            
            table_schema_dict = getattr(task, 'table_schema_dict', None)
            if table_schema_dict is not None:
                db_schema.tables = table_schema_dict
            
            session_state, reasoning_trace = asyncio.run(engine.reason(question, task.schema_context, evidence, db_schema, db_path.as_posix(), question_id=str(question_id), retrieved_values=getattr(task, 'retrieved_values', None)))

            task_result = TaskResult(
                task_index=i,
                question_id=question_id,
                db_id=db_id,
                difficulty=getattr(task, 'difficulty', 'N/A'),
                question=question,
                schema=task.schema_context,
                session_state=session_state,
                reasoning_trace=reasoning_trace
            )
            return task_result

    except Exception as e:
        logger.error(f"Failed to process task {i+1} in worker process: {e}", file=sys.stderr)
        return None


async def run_real_data_reasoning(max_tasks: Optional[int] = None, num_workers: int = 1, task_index: Optional[int] = None):
    """Run inference on real data and save one .pkl file per task"""
    logger.info("SQLConductor inference")
    
    engine = get_reasoning_engine()
    
    status = engine.get_system_status()
    logger.info(f"System status: {status['status']}")
    
    if status['status'] != 'initialized':
        logger.error("System not initialized")
        return
    
    validation = engine.validate_config()
    logger.info(f"Configuration: {'valid' if validation['overall'] else 'invalid'}")
    
    if not validation['overall']:
        logger.error("Configuration validation failed. Please check your config file.")
        return
    
    try:
        data_file = config.config_data.get('system', {}).get('data', {}).get('real_tasks_file', 'data/preprocessed/bird/dev/sample_tasks.pkl')
        output_dir = config.config_data.get('system', {}).get('data', {}).get('output_dir', 'outputs/reasoning_results')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Data file path: {data_file}")
        logger.info(f"Output directory for .pkl files: {output_dir}")
        logger.info(f"Parallel workers: {num_workers}")

        with open(data_file, 'rb') as f:
            real_data = pickle.load(f)
        logger.info(f"Total task number: {len(real_data)} in {data_file}")
        
        if task_index is not None:
            if task_index < 0 or task_index >= len(real_data):
                logger.error(f"Task index {task_index} out of range [0, {len(real_data) - 1}]")
                return
            tasks_to_process = [real_data[task_index]]
            logger.info(f"Running task at index {task_index}")
        else:
            tasks_to_process = real_data[:max_tasks] if max_tasks is not None else real_data
            
        logger.info(f"Task number to process: {len(tasks_to_process)}")

        processed_count = 0
        tasks_with_indices = list(enumerate(tasks_to_process))
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_task = {executor.submit(_process_task_worker, task): task for task in tasks_with_indices}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks_to_process), desc="Processing tasks"):
                task_result = future.result()
                
                if task_result:
                    output_path = Path(output_dir) / f"task_{task_result.question_id}.pkl"
                    with open(output_path, 'wb') as f:
                        pickle.dump(task_result, f)
                    processed_count += 1

        logger.info(f"\n{processed_count}/{len(tasks_to_process)} tasks processed and saved to {output_dir}")
        
    except FileNotFoundError:
        logger.error(f"Task file not found: {data_file}")
    except Exception as e:
        logger.error(f"Reasoning failed: {e}")
        raise


def export_candidates_to_json(results_dir: str, output_path: str):
    """
    Extract SQL candidates from .pkl files and save them to standalone JSON files
    Includes, for each question, the selected SQL, all candidate SQL statements,
    and their related metadata.
    """
    results_path = Path(results_dir)
    if not results_path.is_dir():
        logger.error(f"Results directory not found: {results_dir}")
        return

    all_candidates = []
    pkl_files = sorted(list(results_path.glob("*.pkl")))

    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            try:
                task_result: TaskResult = pickle.load(f)
                all_candidates.append(task_result.to_candidates_dict())
            except (pickle.UnpicklingError, EOFError, AttributeError) as e:
                logger.warning(f"Could not load {pkl_file}, skipping. Error: {e}")
    
    all_candidates.sort(key=lambda x: x.get('question_id', 0))

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_candidates, f, indent=2, ensure_ascii=False)
        
    logger.info(f"SQL candidates exported to {output_path}")


def _debug_evaluate_single_task(output_dir: str, answer_json_path: str):
    """Evaluate a single task and print its SQL execution results."""
    from src.eval.bird_eval import execute_model
    import sqlite3

    try:
        with open(answer_json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load compiled answer.json: {e}")
        return

    if not results:
        logger.error("No results found in answer.json")
        return
    
    result = results[0]

    predicted_sql = result.get('final_sql', 'SELECT 1')
    gold_sql = result.get('gold_sql', '')
    db_id = result.get('db_id', '')
    question_id = result.get('question_id', 'N/A')
    question = result.get('question', '')

    db_root = config.config_data.get('system', {}).get('data', {}).get('database_root_dir')
    if not db_root:
        raise ValueError("'database_root_dir' is not set in config (system.data.database_root_dir). Please specify it in your config file.")
    db_path = f"{db_root}/{db_id}/{db_id}.sqlite"

    eval_result = execute_model(predicted_sql, gold_sql, db_path, 0, meta_time_out=30.0)

    pred_rows, gold_rows = None, None
    try:
        conn = sqlite3.connect(db_path)
        from src.database.sql_execution import _register_custom_functions
        _register_custom_functions(conn)
        cursor = conn.cursor()
        cursor.execute(predicted_sql)
        pred_rows = cursor.fetchall()
        cursor.execute(gold_sql)
        gold_rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        logger.warning(f"Execution error while fetching result sets: {e}")

    status = eval_result['status']
    is_correct = status == 'correct'
    banner = "✅ CORRECT" if is_correct else f"❌ {status.upper()}"

    def format_rows(rows):
        if rows is None:
            return "Execution Error or Timeout"
        if not rows:
            return "Empty Result Set"
        s = repr(rows[:5])
        if len(rows) > 5:
            s += f" ... (and {len(rows)-5} more rows)"
        return s

    report = f"""
{'=' * 80}
SINGLE-TASK EVALUATION
{'=' * 80}
Question ID : {question_id}
Question    : {question}
DB          : {db_id}
Result      : {banner}
{'-' * 80}
Predicted SQL:
{predicted_sql}

Predicted Result (top 5 rows):
{format_rows(pred_rows)}
{'-' * 80}
Gold SQL:
{gold_sql}

Gold Result (top 5 rows):
{format_rows(gold_rows)}
{'=' * 80}"""
    print(report)


async def main():
    parser = argparse.ArgumentParser(description="Text-to-SQL Reasoning System")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a custom YAML configuration file."
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of tasks for real-data inference (default: process all tasks)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel processes for real-data reasoning."
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="Run a single task by its 0-based index from the data file and auto-evaluate."
    )
    parser.add_argument(
        "--compile-results",
        action="store_true",
        help="Compile existing .pkl results into a final JSON file without running reasoning."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing the .pkl result files. If not provided, defaults to output_dir from config."
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/final_summary.json",
        help="Path to save the final compiled JSON file."
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        default="meta_agent",
        choices=["meta_agent"],
        help="Type of agent to use for reasoning. Overrides config file setting."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Override log level from config file (e.g., DEBUG, INFO, WARNING)."
    )
    parser.add_argument(
        "--real-tasks-file",
        type=str,
        default=None,
        help="Override real tasks file path from config file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory path from config file."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--worker-impl",
        type=str,
        action="append",
        default=[],
        help="Override worker implementation. Format: 'Worker Type=implementation'. Can be used multiple times. Example: --worker-impl 'SQL Constructor=robust' --worker-impl 'Schema Pruner=default'"
    )
    parser.add_argument(
        "--unrestricted",
        action="store_true",
        default=False,
        help="Enable unrestricted mode: relax meta agent orchestration constraints and let the LLM more freely decide the reasoning flow. Only applies to --agent-type meta_agent."
    )

    args = parser.parse_args()

    if args.config:
        logger.info(f"Loading custom configuration from: {args.config}")
        config.load_config(args.config)

    if args.log_level:
        config.config_data.setdefault('system', {})['log_level'] = args.log_level.upper()
        logger.info(f"Overriding log_level with: {args.log_level.upper()}")

    if args.real_tasks_file:
        config.config_data.setdefault('system', {}).setdefault('data', {})['real_tasks_file'] = args.real_tasks_file
        logger.info(f"Overriding real_tasks_file with: {args.real_tasks_file}")

    if args.output_dir:
        config.config_data.setdefault('system', {}).setdefault('data', {})['output_dir'] = args.output_dir
        logger.info(f"Overriding output_dir with: {args.output_dir}")

    if args.task_index is not None:
        current_out_dir = config.config_data.get('system', {}).get('data', {}).get('output_dir', 'outputs/reasoning_results')
        debug_out_dir = str(Path(current_out_dir) / f"debug/q_{args.task_index}")
        config.config_data.setdefault('system', {}).setdefault('data', {})['output_dir'] = debug_out_dir
        logger.info(f"Single-task output directory: {debug_out_dir}")

    if args.unrestricted:
        config.config_data.setdefault('system', {})['unrestricted'] = True
        logger.info("Unrestricted mode enabled.")

    config.set_agent_type(args.agent_type)

    if args.worker_impl:
        worker_impls = {}
        for worker_impl_arg in args.worker_impl:
            try:
                if '=' not in worker_impl_arg:
                    logger.warning(f"Invalid worker-impl format: '{worker_impl_arg}'. Expected format: 'Worker Type=implementation'. Skipping.")
                    continue
                worker_type, impl_name = worker_impl_arg.split('=', 1)
                worker_type = worker_type.strip()
                impl_name = impl_name.strip()
                worker_impls[worker_type] = impl_name
                logger.info(f"Overriding worker implementation: '{worker_type}' = '{impl_name}'")
            except Exception as e:
                logger.warning(f"Failed to parse worker-impl argument '{worker_impl_arg}': {e}. Skipping.")
        
        if worker_impls:
            config.set_worker_impls_from_dict(worker_impls)

    config.setup_logging()

    if args.results_dir is None:
        args.results_dir = config.config_data.get('system', {}).get('data', {}).get('output_dir', 'outputs/reasoning_results')

    data_file_path = config.config_data.get('system', {}).get('data', {}).get('data_file_path')

    if args.compile_results:
        compile_results(args.results_dir, args.output_json, data_file_path=data_file_path)
    else:
        try:
            await run_real_data_reasoning(args.max_tasks, args.num_workers, args.task_index)

            logger.info("Reasoning complete. Compiling results...")
            output_dir = config.config_data.get('system', {}).get('data', {}).get('output_dir', 'outputs/reasoning_results')
            final_json_path = str(Path(output_dir) / "answer.json")
            compile_results(output_dir, final_json_path, data_file_path=data_file_path)
            
            candidates_json_path = str(Path(output_dir) / "sql_candidates.json")
            export_candidates_to_json(output_dir, candidates_json_path)
            
            if args.task_index is not None:
                _debug_evaluate_single_task(output_dir, final_json_path)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except asyncio.CancelledError:
            logger.debug("Event loop shutdown cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

if __name__ == "__main__":
    asyncio.run(main())
