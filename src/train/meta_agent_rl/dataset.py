
import json
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

class RLDataset(Dataset):
    """
    Dataset for MetaAgent RL training.
    Loads training questions, database schemas, and trajectory metadata.
    """
    def __init__(self, tasks_path: str, trajectory_summary_path: str,
                 curriculum_threshold: Optional[float] = None):
        if curriculum_threshold is not None and not 0.0 <= curriculum_threshold <= 1.0:
            raise ValueError("curriculum_threshold must be between 0 and 1")
        self.all_tasks = self._load_tasks(tasks_path)
        summary_path = Path(trajectory_summary_path)
        if summary_path.exists():
            with summary_path.open('r', encoding='utf-8') as f:
                summary = json.load(f)
            summary_question_ids = set()
            for item in summary:
                qid = item.get("question_id")
                if qid is None:
                    continue
                if curriculum_threshold is not None:
                    total = item.get("total_trajectories")
                    correct = item.get("correct_trajectories_count")
                    if (type(total) is not int or type(correct) is not int
                            or not 0 <= correct <= total):
                        raise ValueError(
                            f"Invalid trajectory counts for question {qid}: expected integer "
                            "counts with 0 <= correct_trajectories_count <= total_trajectories"
                        )
                    # A success ratio requires at least one explored trajectory.
                    if total == 0 or correct / total >= curriculum_threshold:
                        continue
                summary_question_ids.add(qid)
        else:
            logger.warning(f"Trajectory summary file not found: {summary_path}")
            summary_question_ids = set()
        
        # Keep tasks present in the trajectory summary.
        self.valid_tasks = []
        for task in self.all_tasks:
            if hasattr(task, "question_id"):
                qid = task.question_id
                if hasattr(task, "model_dump"):
                    task_data = task.model_dump()
                elif hasattr(task, "dict"):
                    task_data = task.dict()
                else:
                    task_data = task.__dict__
            else:
                qid = task.get("question_id")
                task_data = task

            try:
                qid_int = int(qid)
                if qid_int in summary_question_ids:
                    self.valid_tasks.append(task_data)
            except (ValueError, TypeError):
                continue
                
        logger.info(f"Loaded {len(self.valid_tasks)} training tasks from {len(self.all_tasks)} total.")

    def _load_tasks(self, path: str) -> List[Dict[str, Any]]:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Task file not found: {path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith('.json'):
                content = json.load(f)
                return content if isinstance(content, list) else []
            elif path.endswith('.jsonl'):
                return [json.loads(line) for line in f]
            else:
                import pickle
                with open(path, 'rb') as pf:
                    return pickle.load(pf)

    def __len__(self):
        return len(self.valid_tasks)

    def __getitem__(self, idx):
        task = self.valid_tasks[idx]
        qid = task.get("question_id")
        
        return {
            "question_id": qid,
            "question": task.get("question", ""),
            "db_id": task.get("db_id", ""),
            "evidence": task.get("evidence", ""),
            "ground_truth_sql": task.get("sql", ""),
            "table_schema_dict": task.get("table_schema_dict", {})
        }

    def collate_fn(self, batch):
        return {
            "question_id": [x["question_id"] for x in batch],
            "question": [x["question"] for x in batch],
            "db_id": [x["db_id"] for x in batch],
            "evidence": [x["evidence"] for x in batch],
            "ground_truth_sql": [x["ground_truth_sql"] for x in batch],
            "table_schema_dict": [x["table_schema_dict"] for x in batch]
        }
