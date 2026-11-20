"""Strategies for selecting correct trajectories for each task."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict, Optional, Set
from loguru import logger
import random
from collections import defaultdict
import json
from pathlib import Path


class TrajectorySelectionStrategy(ABC):
    """
    Abstract base class for trajectory selection strategies.
    
    All strategies must implement the select method which takes a list of
    correct trajectories and returns the index of the selected one.
    """
    
    @abstractmethod
    def select(self, correct_trajectories: List[Tuple[int, Any]], 
               trajectories_info: List[Dict[str, Any]]) -> Optional[int]:
        """
        Select the best trajectory from a list of correct trajectories.
        
        Args:
            correct_trajectories: List of tuples (index, trajectory) for correct trajectories
            trajectories_info: List of trajectory info dicts (for additional metadata)
        
        Returns:
            Index of the selected trajectory, or None if no selection can be made
        """
        pass
    
    def select_topk(self, k: int,
                    correct_trajectories: List[Tuple[int, Any]],
                    trajectories_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Select top-k trajectories, one per unique cluster (worker sequence).

        Steps:
          1. Group correct_trajectories by their actions (worker sequence) as the
             cluster key.
          2. Within each cluster pick one representative using select().
          3. Iteratively rank the cluster representatives with select(), returning
             up to k results.

        Args:
            k: Maximum number of trajectories (clusters) to select
            correct_trajectories: List of tuples (index, trajectory) for correct trajectories
            trajectories_info: List of trajectory info dicts with additional metadata

        Returns:
            List of dicts (up to k items) in ranked order, each containing:
                "index"    : trajectory index in the original PKL data
                "rank"     : 1-based rank (1 = best)
                "metadata" : dict with at least "length" and "actions"
        """
        if not correct_trajectories:
            return []

        cluster_map: Dict[tuple, List[Tuple[int, Any]]] = {}
        for idx, traj in correct_trajectories:
            traj_info = next(
                (info for info in trajectories_info if info["index"] == idx), None
            )
            actions_key = tuple(traj_info.get("actions", [])) if traj_info else ()
            cluster_map.setdefault(actions_key, []).append((idx, traj))

        cluster_reps: List[Tuple[int, Any]] = []
        for members in cluster_map.values():
            best_idx = self.select(members, trajectories_info)
            if best_idx is not None:
                rep_traj = next(t for i, t in members if i == best_idx)
                cluster_reps.append((best_idx, rep_traj))

        results = []
        remaining = list(cluster_reps)
        for rank in range(1, k + 1):
            if not remaining:
                break
            best_idx = self.select(remaining, trajectories_info)
            if best_idx is None:
                break
            traj_info = next(
                (info for info in trajectories_info if info["index"] == best_idx), {}
            )
            results.append({
                "index": best_idx,
                "rank": rank,
                "metadata": {
                    "length": traj_info.get("length", 0),
                    "actions": traj_info.get("actions", []),
                }
            })
            remaining = [(idx, traj) for idx, traj in remaining if idx != best_idx]

        return results

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of this strategy for logging/identification."""
        pass


class ShortestTrajectoryStrategy(TrajectorySelectionStrategy):
    """Select the shortest correct trajectory by number of steps."""
    
    def select(self, correct_trajectories: List[Tuple[int, Any]], 
               trajectories_info: List[Dict[str, Any]]) -> Optional[int]:
        """
        Select the shortest trajectory from correct trajectories.
        
        If multiple trajectories have the same length, selects the one
        with the smallest index (first encountered).
        """
        if not correct_trajectories:
            return None
        
        correct_trajectories.sort(key=lambda t: len(t[1]))
        best_idx, _ = correct_trajectories[0]
        
        return best_idx
    
    def get_strategy_name(self) -> str:
        return "shortest"


class LongestTrajectoryStrategy(TrajectorySelectionStrategy):
    """Select the longest correct trajectory."""
    
    def select(self, correct_trajectories: List[Tuple[int, Any]], 
               trajectories_info: List[Dict[str, Any]]) -> Optional[int]:
        """Select the longest trajectory from correct trajectories."""
        if not correct_trajectories:
            return None
        
        correct_trajectories.sort(key=lambda t: len(t[1]), reverse=True)
        best_idx, _ = correct_trajectories[0]
        
        return best_idx
    
    def get_strategy_name(self) -> str:
        return "longest"


class FirstTrajectoryStrategy(TrajectorySelectionStrategy):
    """Select the correct trajectory with the smallest index."""
    
    def select(self, correct_trajectories: List[Tuple[int, Any]], 
               trajectories_info: List[Dict[str, Any]]) -> Optional[int]:
        """Select the first correct trajectory by index."""
        if not correct_trajectories:
            return None
        
        correct_trajectories.sort(key=lambda t: t[0])
        best_idx, _ = correct_trajectories[0]
        
        return best_idx
    
    def get_strategy_name(self) -> str:
        return "first"


class LastTrajectoryStrategy(TrajectorySelectionStrategy):
    """
    Selects the last correct trajectory encountered (by original index).
    """
    
    def select(self, correct_trajectories: List[Tuple[int, Any]], 
               trajectories_info: List[Dict[str, Any]]) -> Optional[int]:
        """Select the last correct trajectory by index."""
        if not correct_trajectories:
            return None
        
        correct_trajectories.sort(key=lambda t: t[0], reverse=True)
        best_idx, _ = correct_trajectories[0]
        
        return best_idx
    
    def get_strategy_name(self) -> str:
        return "last"


class RandomTrajectoryStrategy(TrajectorySelectionStrategy):
    """
    Randomly selects one of the correct trajectories.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: Optional random seed for reproducibility
        """
        import random
        # Keep the random generator pickleable.
        self.random = random.Random(seed) if seed is not None else random.Random()
    
    def select(self, correct_trajectories: List[Tuple[int, Any]], 
               trajectories_info: List[Dict[str, Any]]) -> Optional[int]:
        """Randomly select one correct trajectory."""
        if not correct_trajectories:
            return None
        
        selected = self.random.choice(correct_trajectories)
        return selected[0]
    
    def get_strategy_name(self) -> str:
        return "random"


class LaplaceSmoothingStrategy(TrajectorySelectionStrategy):
    """Rank trajectories by Laplace-smoothed success probability and tie-breakers."""
    
    SCHEMA_PRUNER = "Schema Pruner"
    SQL_REFINER = "SQL Refiner"
    
    def __init__(self, 
                 trajectory_stats: Optional[Dict[str, Dict[str, int]]] = None,
                 epsilon: float = 0.02,
                 seed: Optional[int] = None):
        """
        Args:
            trajectory_stats: Optional dict mapping trajectory signature (worker sequence)
                            to {"success": count, "total": count}. If None, will use
                            uniform prior (all trajectories have same probability).
            epsilon: Threshold for considering probabilities as equal.
            seed: Random seed for tie-breaking (default: None)
        """
        self.trajectory_stats = trajectory_stats or {}
        self.epsilon = epsilon
        # Keep the random generator pickleable.
        self.random = random.Random(seed) if seed is not None else random.Random()
    
    def _get_trajectory_signature(self, actions: List[str]) -> Tuple[str, ...]:
        """
        Generate a signature for a trajectory based on its worker sequence.
        This signature is used to look up statistics.
        """
        return tuple(actions)
    
    def _calculate_laplace_probability(self, signature: Tuple[str, ...]) -> Tuple[float, int]:
        """
        Calculate Laplace-smoothed probability: P = (s + 1) / (n + 2)
        where s is success count, n is total count.
        
        Returns:
            Tuple of (probability, n) where n is the sample size used
        """
        if signature in self.trajectory_stats:
            stats = self.trajectory_stats[signature]
            s = stats.get("success", 0)
            n = stats.get("total", 0)
        else:
            # Uniform prior: no data means s=0, n=0
            s = 0
            n = 0
        
        probability = (s + 1) / (n + 2) if n >= 0 else 0.5
        return probability, n
    
    def _has_worker(self, actions: List[str], worker_name: str) -> bool:
        """Check if trajectory contains a specific worker."""
        return worker_name in actions
    
    def _has_both_workers(self, actions: List[str]) -> bool:
        """Check if trajectory has both Schema Pruner and SQL Refiner."""
        return (self._has_worker(actions, self.SCHEMA_PRUNER) and 
                self._has_worker(actions, self.SQL_REFINER))
    
    def _has_refiner(self, actions: List[str]) -> bool:
        """Check if trajectory has SQL Refiner."""
        return self._has_worker(actions, self.SQL_REFINER)
    
    def select(self, correct_trajectories: List[Tuple[int, Any]], 
               trajectories_info: List[Dict[str, Any]]) -> Optional[int]:
        """
        Select trajectory using multi-criteria decision making.
        """
        if not correct_trajectories:
            return None
        
        candidates = []
        for idx, trajectory in correct_trajectories:
            traj_info = None
            for info in trajectories_info:
                if info["index"] == idx:
                    traj_info = info
                    break
            
            if traj_info is None:
                logger.warning(f"Could not find trajectory info for index {idx}")
                continue
            
            actions = traj_info.get("actions", [])
            signature = self._get_trajectory_signature(actions)
            probability, n = self._calculate_laplace_probability(signature)
            length = traj_info.get("length", len(trajectory))
            
            candidates.append({
                "index": idx,
                "probability": probability,
                "n": n,
                "length": length,
                "actions": actions,
                "has_both": self._has_both_workers(actions),
                "has_refiner": self._has_refiner(actions)
            })
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x["probability"], reverse=True)
        
        max_prob = candidates[0]["probability"]
        candidates = [c for c in candidates if abs(c["probability"] - max_prob) <= self.epsilon]
        
        if len(candidates) == 1:
            return candidates[0]["index"]
        
        has_both = [c for c in candidates if c["has_both"]]
        if has_both:
            candidates = has_both
        else:
            has_refiner = [c for c in candidates if c["has_refiner"]]
            if has_refiner:
                candidates = has_refiner
        
        if len(candidates) == 1:
            return candidates[0]["index"]
        
        candidates.sort(key=lambda x: x["n"], reverse=True)
        max_n = candidates[0]["n"]
        candidates = [c for c in candidates if c["n"] == max_n]
        
        if len(candidates) == 1:
            return candidates[0]["index"]
        
        candidates.sort(key=lambda x: x["length"])
        min_length = candidates[0]["length"]
        candidates = [c for c in candidates if c["length"] == min_length]
        
        if len(candidates) == 1:
            return candidates[0]["index"]
        
        selected = self.random.choice(candidates)
        return selected["index"]

    def select_topk(self, k: int,
                    correct_trajectories: List[Tuple[int, Any]],
                    trajectories_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Select top-k trajectories, one per unique cluster (worker sequence).

        Rank clusters by probability, worker coverage, sample count, and length.
        Each representative is the shortest trajectory in its cluster.

        Returns:
            List of up to k dicts in ranked order, each with:
                "index", "rank", "metadata":
                    "length", "actions",
                    "laplace_probability", "n",
                    "has_both_workers", "has_refiner"
        """
        if not correct_trajectories:
            return []

        cluster_map: Dict[tuple, list] = {}  # signature -> [{index, length}, ...]
        for idx, trajectory in correct_trajectories:
            traj_info = next(
                (info for info in trajectories_info if info["index"] == idx), None
            )
            if traj_info is None:
                logger.warning(f"Could not find trajectory info for index {idx}")
                continue
            actions = traj_info.get("actions", [])
            signature = self._get_trajectory_signature(actions)
            length = traj_info.get("length", len(trajectory))
            cluster_map.setdefault(signature, []).append({"index": idx, "length": length})

        if not cluster_map:
            return []

        cluster_candidates = []
        for signature, members in cluster_map.items():
            actions = list(signature)
            probability, n = self._calculate_laplace_probability(signature)
            # Representative: shortest trajectory within the cluster (then smallest index)
            rep = min(members, key=lambda m: (m["length"], m["index"]))
            cluster_candidates.append({
                "index": rep["index"],
                "length": rep["length"],
                "actions": actions,
                "probability": probability,
                "n": n,
                "has_both": self._has_both_workers(actions),
                "has_refiner": self._has_refiner(actions),
            })

        sort_key = lambda c: (
            c["probability"],
            int(c["has_both"]),
            int(c["has_refiner"]),
            c["n"],
            -c["length"],
        )
        cluster_candidates.sort(key=sort_key, reverse=True)

        # Sample among tied clusters at the selection boundary.
        if len(cluster_candidates) > k:
            cutoff_score = sort_key(cluster_candidates[k - 1])
            top_clear = [c for c in cluster_candidates if sort_key(c) > cutoff_score]
            tied = [c for c in cluster_candidates if sort_key(c) == cutoff_score]
            remaining_slots = k - len(top_clear)
            if remaining_slots > 0 and len(tied) > remaining_slots:
                tied = self.random.sample(tied, remaining_slots)
            cluster_candidates = top_clear + tied

        results = []
        for rank, c in enumerate(cluster_candidates[:k], 1):
            results.append({
                "index": c["index"],
                "rank": rank,
                "metadata": {
                    "length": c["length"],
                    "actions": c["actions"],
                    "laplace_probability": round(c["probability"], 4),
                    "n": c["n"],
                    "has_both_workers": c["has_both"],
                    "has_refiner": c["has_refiner"],
                },
            })

        return results

    def get_strategy_name(self) -> str:
        return "laplace_smoothing"


STRATEGY_REGISTRY = {
    "shortest": ShortestTrajectoryStrategy,
    "longest": LongestTrajectoryStrategy,
    "first": FirstTrajectoryStrategy,
    "last": LastTrajectoryStrategy,
    "random": RandomTrajectoryStrategy,
    "laplace_smoothing": LaplaceSmoothingStrategy,
}


def collect_trajectory_statistics(summaries: List[Dict[str, Any]]) -> Dict[Tuple[str, ...], Dict[str, int]]:
    """
    Collect trajectory statistics from summary data for Laplace smoothing.
    
    Args:
        summaries: List of summary dicts from TrajectorySummaryProcessor
    
    Returns:
        Dict mapping trajectory signature (tuple of worker names) to 
        {"success": count, "total": count}
    """
    stats = defaultdict(lambda: {"success": 0, "total": 0})
    
    for summary in summaries:
        trajectory_clusters = summary.get("trajectory_clusters", [])
        if trajectory_clusters:
            for cluster in trajectory_clusters:
                worker_sequence = cluster.get("worker_sequence", [])
                signature = tuple(worker_sequence)
                total_count = cluster.get("total_count", 0)
                success_count = cluster.get("success_count", 0)
                
                stats[signature]["total"] += total_count
                stats[signature]["success"] += success_count
        else:
            trajectories_info = summary.get("trajectories_info", [])
            for traj_info in trajectories_info:
                actions = traj_info.get("actions", [])
                signature = tuple(actions)
                
                stats[signature]["total"] += 1
                if traj_info.get("correct", False):
                    stats[signature]["success"] += 1
    
    return dict(stats)


def load_trajectory_statistics_from_summary_file(summary_file_path: Path) -> Dict[Tuple[str, ...], Dict[str, int]]:
    """
    Load trajectory statistics from a summary JSON file.
    
    Args:
        summary_file_path: Path to the summary JSON file
    
    Returns:
        Dict mapping trajectory signature to statistics
    """
    if not summary_file_path.exists():
        logger.warning(f"Summary file not found: {summary_file_path}. Using empty statistics.")
        return {}
    
    try:
        with open(summary_file_path, 'r') as f:
            summaries = json.load(f)
        
        stats = collect_trajectory_statistics(summaries)
        return stats
    except Exception as e:
        logger.error(f"Failed to load statistics from {summary_file_path}: {e}")
        return {}


def get_strategy(strategy_name: str, **kwargs) -> TrajectorySelectionStrategy:
    """
    Factory function to get a strategy by name.
    
    Args:
        strategy_name: Name of the strategy (e.g., "shortest", "longest", "first", "last", "random", "laplace_smoothing")
        **kwargs: Additional arguments to pass to strategy constructor
    
    Returns:
        An instance of the requested strategy
    
    Raises:
        ValueError: If strategy_name is not recognized
    """
    if strategy_name not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Available strategies: {available}"
        )
    
    strategy_class = STRATEGY_REGISTRY[strategy_name]
    return strategy_class(**kwargs)
