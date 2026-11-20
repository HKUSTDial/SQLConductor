import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger
from tqdm import tqdm
import multiprocessing as mp

from src.system.types import (
    WorkerType, ActionName, ACTION_WORKER_MAP
)
from src.mcts.mcts_action import EndAction
from src.mcts.serialization import load_trajectories
from src.database.sql_execution import (
    format_execution_result, cached_execute_sql_with_timeout, is_valid_execution_result
)
from src.trajectories.trajectory_selection_strategy import (
    TrajectorySelectionStrategy, 
    ShortestTrajectoryStrategy,
    get_strategy
)

class TrajectorySummaryProcessor:
    """
    Trajectory Summary Processor:
    Extracts summary and statistical information from trajectories obtained from MCTS
    Identify valid paths and select paths for further processing.
    """
    
    def __init__(self, selection_strategy: Optional[TrajectorySelectionStrategy] = None):
        """
        Initialize the processor.
        
        Args:
            selection_strategy: Strategy for selecting trajectories when multiple
                               correct trajectories exist. If None, uses ShortestTrajectoryStrategy
                               (default behavior). Can also pass a string like "shortest", "longest", etc.
        """
        if selection_strategy is None:
            self.selection_strategy = ShortestTrajectoryStrategy()
        elif isinstance(selection_strategy, str):
            self.selection_strategy = get_strategy(selection_strategy)
        elif isinstance(selection_strategy, TrajectorySelectionStrategy):
            self.selection_strategy = selection_strategy
        else:
            raise TypeError(
                f"selection_strategy must be TrajectorySelectionStrategy, str, or None, "
                f"got {type(selection_strategy)}"
            )

    def _get_worker_type(self, action: Any) -> Optional[WorkerType]:
        """Maps MCTS Action to WorkerType"""
        action_name_str = action.__class__.__name__
        for action_enum in ActionName:
            if action_enum.value == action_name_str:
                return ACTION_WORKER_MAP.get(action_enum)
        
        if isinstance(action, EndAction):
            return WorkerType.END
            
        return None

    def process_pkl_file_summary(self, pkl_path: Path, gold_sqls: Dict[int, str]) -> Dict[str, Any]:
        """Process a PKL file and return only the summary."""
        try:
            with open(pkl_path, 'rb') as f:
                data = load_trajectories(f)
            
            if not data:
                return None
                
            first_node = data[0][0]
            question_id = int(pkl_path.stem)
            question = first_node.original_question
            
            db_id = getattr(first_node, 'db_id', "")
            db_root_dir = getattr(first_node, 'db_root_dir', "")
            
            db_path = None
            if db_root_dir and db_id:
                db_path = Path(db_root_dir) / db_id / f"{db_id}.sqlite"

            trajectories_info = []
            correct_trajectories = []
            
            for idx, trajectory in enumerate(data):
                final_node = trajectory[-1]
                final_sql = getattr(final_node, 'final_sql_query', None)
                
                is_correct = False
                if final_sql and question_id in gold_sqls and db_path:
                    gold_sql = gold_sqls[question_id]
                    try:
                        pred_res = cached_execute_sql_with_timeout(db_path, final_sql)
                        gold_res = cached_execute_sql_with_timeout(db_path, gold_sql)
                        
                        if (is_valid_execution_result(pred_res) and
                            is_valid_execution_result(gold_res)):
                            # Compare as sets to ignore order and column names
                            pred_set = set(pred_res.result)
                            gold_set = set(gold_res.result)
                            if pred_set == gold_set:
                                is_correct = True
                    except:
                        pass
                
                actions = []
                for node in trajectory[1:]:
                    if hasattr(node, 'parent_action'):
                        action = node.parent_action
                        worker_type = self._get_worker_type(action)
                        if worker_type:
                            actions.append(worker_type.value)

                traj_info = {
                    "index": idx,
                    "length": len(trajectory),
                    "correct": is_correct,
                    "actions": actions,
                    "selected": False
                }
                trajectories_info.append(traj_info)
                
                if is_correct:
                    correct_trajectories.append((idx, trajectory))
            
            trajectory_clusters = {}
            index_to_length = {}
            for traj_info in trajectories_info:
                actions = traj_info["actions"]
                idx = traj_info["index"]
                index_to_length[idx] = traj_info["length"]
                
                cluster_key = tuple(actions)
                
                if cluster_key not in trajectory_clusters:
                    trajectory_clusters[cluster_key] = {
                        "worker_sequence": list(actions),
                        "trajectory_indices": [],
                        "trajectory_lengths": {},
                        "total_count": 0,
                        "success_count": 0
                    }
                
                cluster = trajectory_clusters[cluster_key]
                cluster["trajectory_indices"].append(idx)
                cluster["trajectory_lengths"][idx] = traj_info["length"]
                cluster["total_count"] += 1
                if traj_info["correct"]:
                    cluster["success_count"] += 1
            
            trajectory_clusters_list = []
            for cluster_key, cluster_data in trajectory_clusters.items():
                total = cluster_data["total_count"]
                success = cluster_data["success_count"]
                accuracy = success / total if total > 0 else 0.0
                # Laplace smoothing: P = (s + 1) / (n + 2)
                laplace_probability = (success + 1) / (total + 2) if total >= 0 else 0.5
                
                trajectory_clusters_list.append({
                    "worker_sequence": cluster_data["worker_sequence"],
                    "trajectory_indices": sorted(cluster_data["trajectory_indices"]),
                    "trajectory_lengths": {str(k): v for k, v in cluster_data["trajectory_lengths"].items()},
                    "total_count": total,
                    "success_count": success,
                    "accuracy": round(accuracy, 4),
                    "laplace_probability": round(laplace_probability, 4)
                })
            
            trajectory_clusters_list.sort(
                key=lambda x: (x["laplace_probability"], x["total_count"]), 
                reverse=True
            )
            
            index_to_cluster = {}
            for cluster in trajectory_clusters_list:
                for idx in cluster["trajectory_indices"]:
                    index_to_cluster[idx] = cluster
            
            current_question_stats = {}
            for cluster in trajectory_clusters_list:
                worker_sequence = cluster["worker_sequence"]
                signature = tuple(worker_sequence)
                current_question_stats[signature] = {
                    "success": cluster["success_count"],
                    "total": cluster["total_count"]
                }
            
            if hasattr(self.selection_strategy, 'trajectory_stats'):
                from src.trajectories.trajectory_selection_strategy import LaplaceSmoothingStrategy
                if isinstance(self.selection_strategy, LaplaceSmoothingStrategy):
                    current_strategy = LaplaceSmoothingStrategy(
                        trajectory_stats=current_question_stats,
                        epsilon=self.selection_strategy.epsilon,
                        seed=None
                    )
                    current_strategy.random = self.selection_strategy.random
                else:
                    current_strategy = self.selection_strategy
            else:
                current_strategy = self.selection_strategy
            
            reconstructed_trajectories_info = []
            for traj_info in trajectories_info:
                idx = traj_info["index"]
                if idx in index_to_cluster:
                    cluster = index_to_cluster[idx]
                    reconstructed_trajectories_info.append({
                        "index": idx,
                        "length": traj_info["length"],
                        "correct": traj_info["correct"],
                        "actions": cluster["worker_sequence"]
                    })
            
            summary = {
                "question_id": question_id,
                "db_id": db_id,
                "question": question,
                "total_trajectories": len(data),
                "correct_trajectories_count": len(correct_trajectories),
                "has_valid_trajectory": len(correct_trajectories) > 0,
                "selected_trajectories": [],
                "trajectory_clusters": trajectory_clusters_list,
            }
            
            if correct_trajectories:
                top_k = current_strategy.select_topk(
                    3, correct_trajectories, reconstructed_trajectories_info
                )
                if top_k:
                    summary["selected_trajectories"] = top_k
                else:
                    logger.warning(
                        f"Selection strategy returned empty list for question_id {question_id}. "
                        f"No trajectory selected."
                    )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error processing summary for {pkl_path}: {e}")
            return {
                "question_id": int(pkl_path.stem) if pkl_path.stem.isdigit() else pkl_path.stem,
                "error": str(e)
            }

    def print_analysis(self, summaries: List[Dict[str, Any]], output_file: Path):
        """Print brief analysis of the dataset and save to file."""
        if not summaries:
            logger.info("No summaries to analyze.")
            return
            
        total_tasks = len(summaries)
        
        total_correct_trajectories = 0
        tasks_with_valid = 0
        
        selected_lengths = []
        db_valid_counts = {}
        all_dbs = set()
        
        for s in summaries:
            db_id = s.get("db_id")
            if db_id:
                all_dbs.add(db_id)
            
            if s.get("has_valid_trajectory"):
                tasks_with_valid += 1

                selected_trajs = s.get("selected_trajectories", [])
                if selected_trajs:
                    rank1 = next((t for t in selected_trajs if t.get("rank") == 1), selected_trajs[0])
                    rank1_len = rank1.get("metadata", {}).get("length")
                    if rank1_len is None:
                        rank1_idx = rank1.get("index")
                        if rank1_idx is not None and "trajectory_clusters" in s:
                            for cluster in s["trajectory_clusters"]:
                                tl = cluster.get("trajectory_lengths", {})
                                if str(rank1_idx) in tl:
                                    rank1_len = tl[str(rank1_idx)]
                                    break
                    if rank1_len is not None:
                        selected_lengths.append(rank1_len)
                else:
                    selected_idx = s.get("selected_trajectory_index")
                    if selected_idx is not None:
                        if "trajectory_clusters" in s:
                            for cluster in s["trajectory_clusters"]:
                                trajectory_lengths = cluster.get("trajectory_lengths", {})
                                if str(selected_idx) in trajectory_lengths:
                                    selected_lengths.append(trajectory_lengths[str(selected_idx)])
                                    break
                        elif "trajectories_info" in s:
                            for t_info in s["trajectories_info"]:
                                if t_info["index"] == selected_idx:
                                    selected_lengths.append(t_info["length"])
                                    break

                if db_id:
                    db_valid_counts[db_id] = db_valid_counts.get(db_id, 0) + 1
            
            if "correct_trajectories_count" in s:
                total_correct_trajectories += s["correct_trajectories_count"]
            elif "trajectory_clusters" in s:
                count = sum(cluster.get("success_count", 0) for cluster in s["trajectory_clusters"])
                total_correct_trajectories += count
            elif "trajectories_info" in s:
                count = sum(1 for t in s["trajectories_info"] if t.get("correct"))
                total_correct_trajectories += count
        
        avg_correct = total_correct_trajectories / total_tasks if total_tasks > 0 else 0
        valid_percentage = (tasks_with_valid / total_tasks * 100) if total_tasks > 0 else 0

        total_selected = sum(
            len(s.get("selected_trajectories", []) or ([s["selected_trajectory_index"]] if s.get("selected_trajectory_index") is not None else []))
            for s in summaries if s.get("has_valid_trajectory")
        )
        avg_selected = total_selected / tasks_with_valid if tasks_with_valid > 0 else 0

        length_stats_str = "No valid trajectories selected."
        if selected_lengths:
            min_len = min(selected_lengths)
            max_len = max(selected_lengths)
            avg_len = sum(selected_lengths) / len(selected_lengths)
            
            length_counts = {}
            for l in selected_lengths:
                length_counts[l] = length_counts.get(l, 0) + 1
            sorted_lengths = sorted(length_counts.items())
            length_dist_str = ", ".join([f"Len {l}: {c}" for l, c in sorted_lengths])
            
            length_stats_str = (
                f"Min: {min_len}, Max: {max_len}, Avg: {avg_len:.2f}\n"
                f"Distribution: {length_dist_str}"
            )

        db_stats_str = "No DB info available."
        if all_dbs:
            db_counts_list = []
            for db in all_dbs:
                count = db_valid_counts.get(db, 0)
                db_counts_list.append((db, count))
            
            sorted_dbs = sorted(db_counts_list, key=lambda x: x[1], reverse=True)
            db_stats_lines = [f"{db}: {count}" for db, count in sorted_dbs]
            db_stats_str = "\n".join(db_stats_lines)

        analysis_text = (
            f"{'=' * 40}\n"
            f"Dataset Analysis Report\n"
            f"{'=' * 40}\n"
            f"Total Tasks: {total_tasks}\n"
            f"Total Correct Trajectories: {total_correct_trajectories}\n"
            f"Average Correct Trajectories per Task: {avg_correct:.2f}\n"
            f"Tasks with Valid Trajectory: {tasks_with_valid} ({valid_percentage:.2f}%)\n"
            f"Avg Selected Trajectories per Valid Task: {avg_selected:.2f}\n"
            f"{'-' * 40}\n"
            f"Rank-1 Selected Trajectory Lengths:\n{length_stats_str}\n"
            f"{'-' * 40}\n"
            f"Valid Cases Database Distribution:\n{db_stats_str}\n"
            f"{'=' * 40}\n"
        )
        
        for line in analysis_text.split('\n'):
            if line:
                logger.info(line)
        
        analysis_file = output_file.with_name(f"{output_file.stem}_analysis.txt")
        try:
            with open(analysis_file, 'w') as f:
                f.write(analysis_text)
            logger.info(f"Analysis saved to {analysis_file}")
        except Exception as e:
            logger.error(f"Failed to save analysis file: {e}")

    def process_dataset(self, input_dir: Path, output_file: Path, ref_data_path: Path, n_processes: int = 1):
        """Process all PKL files in directory to generate summaries."""
        input_dir = Path(input_dir)
        pkl_files = sorted(list(input_dir.glob("*.pkl")))
        
        with open(ref_data_path, 'r') as f:
            ref_data = json.load(f)
        gold_sqls = {item['question_id']: item['SQL'] for item in ref_data}
        
        logger.info(f"Found {len(pkl_files)} PKL files.")
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file = output_file.with_name(f"{output_file.stem}_summary.json")
        
        summaries = []
        logger.info("Starting Summary Phase...")
        
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summaries = json.load(f)
            processed_ids = {s["question_id"] for s in summaries if "question_id" in s}
            logger.info(f"Loaded {len(summaries)} existing summaries.")
        else:
            processed_ids = set()
        
        files_to_summarize = [p for p in pkl_files if int(p.stem) not in processed_ids]
        
        if files_to_summarize:
            batch_size = 50
            processed_count = 0
            new_summaries = []
            
            with tqdm(total=len(files_to_summarize), desc="Generating Summaries") as pbar:
                if n_processes > 1:
                    with mp.Pool(n_processes) as pool:
                        results = []
                        for pkl_file in files_to_summarize:
                            results.append(pool.apply_async(self.process_pkl_file_summary, args=(pkl_file, gold_sqls)))
                        
                        for res in results:
                            summary = res.get()
                            if summary:
                                new_summaries.append(summary)
                            
                            processed_count += 1
                            if processed_count % batch_size == 0:
                                full_list = summaries + new_summaries
                                full_list.sort(key=lambda x: x.get("question_id", 0))
                                with open(summary_file, 'w') as f:
                                    json.dump(full_list, f, indent=2, ensure_ascii=False)
                            pbar.update(1)
                else:
                    for pkl_file in files_to_summarize:
                        summary = self.process_pkl_file_summary(pkl_file, gold_sqls)
                        if summary:
                            new_summaries.append(summary)
                        
                        processed_count += 1
                        if processed_count % batch_size == 0:
                            full_list = summaries + new_summaries
                            full_list.sort(key=lambda x: x.get("question_id", 0))
                            with open(summary_file, 'w') as f:
                                json.dump(full_list, f, indent=2, ensure_ascii=False)
                        pbar.update(1)
            
            summaries.extend(new_summaries)
            summaries.sort(key=lambda x: x.get("question_id", 0))
            with open(summary_file, 'w') as f:
                json.dump(summaries, f, indent=2, ensure_ascii=False)
            logger.info(f"Summary phase complete. Saved to {summary_file}")
            
            self.print_analysis(summaries, output_file)
        else:
            logger.info("All files already summarized.")
            self.print_analysis(summaries, output_file)

