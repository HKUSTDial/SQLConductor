#!/usr/bin/env python3
"""
Compile task-result pickles into a JSON file for evaluation.

Usage:
    python -m src.eval.compile_eval_json --input-dir /path/to/pkl/files --output-file results.json
"""

import pickle
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm

from src.system.types import TaskResult, ReasoningSession, ReasoningStep, WorkerResult, WorkerType


def load_gold_data(data_file_path: str = None) -> Dict[int, Dict[str, Any]]:
    """
    Loads gold data (SQL and evidence) from the dataset's dev.json file.
    
    Args:
        data_file_path: Path to the dataset's dev.json file (e.g. data/spider/test/dev.json). If None, falls back to the default Bird dev path.
    """
    if data_file_path:
        possible_paths = [Path(data_file_path)]
    else:
        possible_paths = [
            Path("data/bird/dev/dev.json"),
            Path(__file__).resolve().parent.parent.parent / "data/bird/dev/dev.json"
        ]
    for path in possible_paths:
        if path.exists():
            print(f"Loading gold data from '{path}'...")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return {item['question_id']: item for item in data}
            except Exception as e:
                print(f"Warning: Failed to load gold data from '{path}': {e}", file=sys.stderr)
                return {}
    searched = [str(p) for p in possible_paths]
    print(f"Warning: Could not find gold data file at {searched}. Output will not include 'gold_sql' and 'evidence'.", file=sys.stderr)
    return {}

def compile_results(input_dir: str, output_file: str, data_file_path: str = None):
    """
    Compiles results from a directory of .pkl files into a single JSON file.
    
    Args:
        input_dir: Directory containing .pkl result files.
        output_file: Path to save the compiled JSON.
        data_file_path: Path to the dataset's dev.json for gold SQL lookup.
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"Error: Input directory '{input_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    pkl_files = sorted(list(input_path.glob("*.pkl")))

    if not pkl_files:
        print(f"Warning: No .pkl files found in '{input_dir}'.", file=sys.stderr)
        return

    print(f"Found {len(pkl_files)} .pkl files to process in '{input_dir}'.")

    gold_data_map = load_gold_data(data_file_path)

    for pkl_file in tqdm(pkl_files, desc="Compiling results"):
        try:
            with open(pkl_file, 'rb') as f:
                task_result: TaskResult = pickle.load(f)

            worker_call_sequence = [
                step.worker_call for step in task_result.reasoning_trace if step.worker_call
            ]

            gold_info = gold_data_map.get(task_result.question_id, {})

            result_obj = {
                "question_id": task_result.question_id,
                "question": task_result.question,
                "gold_sql": gold_info.get("SQL", ""),
                "evidence": gold_info.get("evidence", ""),
                "db_id": task_result.db_id,
                "difficulty": task_result.difficulty,
                "success": task_result.session_state.success,
                "final_sql": task_result.session_state.final_sql,
                "total_turns": len(task_result.reasoning_trace),
                "fallback_triggered": getattr(task_result.session_state, "fallback_triggered", False),
                "fallback_reason": getattr(task_result.session_state, "fallback_reason", None),
                "worker_call_sequence": worker_call_sequence,
                "total_tokens": task_result.session_state.total_tokens,
                "meta_agent_usage": getattr(task_result.session_state, 'meta_agent_usage', {}),
                "worker_usage": getattr(task_result.session_state, 'worker_usage', {}),
                "execution_time": task_result.session_state.execution_time,
            }
            all_results.append(result_obj)

        except (pickle.UnpicklingError, EOFError, AttributeError, ModuleNotFoundError) as e:
            print(f"\nWarning: Could not process file '{pkl_file}'. It may be corrupted. Error: {e}", file=sys.stderr)

    try:
        all_results.sort(key=lambda x: x['question_id'])
    except (TypeError, KeyError):
        print("\nWarning: Could not sort results by 'question_id'. It might be missing or of an inconsistent type.", file=sys.stderr)


    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully compiled {len(all_results)} results to '{output_path}'.")


def main():
    parser = argparse.ArgumentParser(description="Compile evaluation data from .pkl files.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing the .pkl result files."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save the final compiled JSON file."
    )
    
    args = parser.parse_args()
    compile_results(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()
