from collections import Counter
from typing import Optional
from loguru import logger

from src.database.sql_execution import cached_execute_sql_with_timeout, is_valid_execution_result


def compute_outcome_reward(predicted_sql: Optional[str], ground_truth_sql: str, db_path: str) -> float:
    """
    Outcome Reward (OR): Cell-multiset ExecF1 plus the standard EX bonus.

        OR = 1 + ExecF1   if execution result exactly matches ground truth
        OR = ExecF1       otherwise

    Range: [0.0, 2.0]
    """
    if not predicted_sql or not ground_truth_sql:
        return 0.0

    try:
        pred_result = cached_execute_sql_with_timeout(db_path, predicted_sql)
        if not is_valid_execution_result(pred_result):
            return 0.0

        gt_result = cached_execute_sql_with_timeout(db_path, ground_truth_sql)
        if not is_valid_execution_result(gt_result):
            return 0.0

        # SQLite scalars preserve text/NULL semantics; numeric 1 and 1.0 compare
        # equally. Count every cell, including repeated values and rows.
        pred_cells = Counter(value for row in pred_result.result for value in row)
        gt_cells = Counter(value for row in gt_result.result for value in row)
        overlap = sum((pred_cells & gt_cells).values())
        cell_count = sum(pred_cells.values()) + sum(gt_cells.values())
        f1 = 2.0 * overlap / cell_count if cell_count else 1.0

        # EX compares complete rows; ExecF1 compares individual cells.
        bonus = float(set(pred_result.result) == set(gt_result.result))
        return f1 + bonus

    except Exception as e:
        logger.warning(f"Error computing outcome reward: {e}")
        return 0.0


def compute_executability_reward(predicted_sql: Optional[str], db_path: str) -> float:
    """
    Execution Reward (ER): Measures whether the predicted SQL can be executed.

        ER = 1.0   if execution completes without errors or timeout (including empty results)
        ER = 0.0   otherwise
    """
    if not predicted_sql:
        return 0.0

    try:
        pred_result = cached_execute_sql_with_timeout(db_path, predicted_sql)
        return float(is_valid_execution_result(pred_result))

    except Exception as e:
        logger.warning(f"Error computing executability reward: {e}")
        return 0.0
