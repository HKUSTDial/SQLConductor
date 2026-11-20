import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from torch.utils.data import Dataset
from loguru import logger
import torch
from transformers import PreTrainedTokenizer
from tqdm import tqdm
from src.train.meta_agent_sft.collator import DataCollatorForSupervisedFineTuning

class MetaAgentSFTDataset(Dataset):
    """SFT dataset containing per-step messages and trajectory weights."""
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer: PreTrainedTokenizer, 
        max_seq_length: int = 4096,
        use_bucketing: bool = False,
        bucket_boundaries: Optional[List[int]] = None,
        alpha: float = 2.0
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.alpha = alpha
        self.examples = []  # List of (messages, trajectory weight * step token fraction)
        self.use_bucketing = use_bucketing
        self.bucket_boundaries = bucket_boundaries or [2048, 4096, 8192, 16384]
        self.lengths = []
        self.bucket_indices = []
        
        trajectory_ranges = self._load_data(data_path)
        self._normalize_trajectory_weights(trajectory_ranges)
        
        if self.use_bucketing:
            self._compute_lengths_and_buckets()
        
    def _load_data(self, data_path: str):
        """Load and process the JSON data."""
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        logger.info(f"Loading data from {data_path}")
        logger.info(f"Found {len(data)} tasks")
        
        count = 0
        skipped = 0
        trajectory_ranges = []
        for task in data:
            if "trajectories" in task:
                # Multiple trajectories per question, each with laplace_probability.
                trajectories = task["trajectories"]
                if not trajectories:
                    continue
                probs = [t["metadata"].get("laplace_probability", 1.0) for t in trajectories]
                powered = [p ** self.alpha for p in probs]
                total = sum(powered)
                weights = [p / total for p in powered] if total > 0 else [1.0 / len(powered)] * len(powered)

                for traj, weight in zip(trajectories, weights):
                    start = len(self.examples)
                    for step in traj.get("steps", []):
                        if not all(key in step for key in ["system_prompt", "user_prompt", "target_response"]):
                            skipped += 1
                            continue
                        if not step.get("target_response", "").strip():
                            skipped += 1
                            continue
                        messages = [
                            {"role": "system", "content": step["system_prompt"]},
                            {"role": "user", "content": step["user_prompt"]},
                            {"role": "assistant", "content": step["target_response"]}
                        ]
                        self.examples.append((messages, weight))
                        count += 1
                    trajectory_ranges.append((start, len(self.examples)))
            else:
                # Single trajectory with steps at the top level and unit weight.
                start = len(self.examples)
                for step in task.get("steps", []):
                    if not all(key in step for key in ["system_prompt", "user_prompt", "target_response"]):
                        skipped += 1
                        continue
                    if not step.get("target_response", "").strip():
                        skipped += 1
                        continue
                    messages = [
                        {"role": "system", "content": step["system_prompt"]},
                        {"role": "user", "content": step["user_prompt"]},
                        {"role": "assistant", "content": step["target_response"]}
                    ]
                    self.examples.append((messages, 1.0))
                    count += 1
                trajectory_ranges.append((start, len(self.examples)))
        
        if skipped > 0:
            logger.warning(f"Skipped {skipped} invalid examples during loading")
                
        logger.info(f"Processed {count} training examples")
        return trajectory_ranges

    def _normalize_trajectory_weights(self, trajectory_ranges):
        """Scale step weights by their share of trajectory target tokens."""
        # Count shifted, unmasked labels after truncation.
        collator = DataCollatorForSupervisedFineTuning(
            self.tokenizer, max_length=self.max_seq_length, pad_to_multiple_of=0
        )
        examples = []
        self.lengths = []
        skipped = 0
        for start, end in tqdm(trajectory_ranges, desc="Normalizing trajectory weights", unit="trajectory"):
            trajectory_examples = []
            for messages, weight in self.examples[start:end]:
                batch = collator([{"messages": messages}])
                token_count = (batch["labels"][:, 1:] != collator.ignore_index).sum().item()
                if token_count == 0:
                    skipped += 1
                    continue
                trajectory_examples.append((messages, weight, token_count))
                self.lengths.append(batch["input_ids"].size(1))

            total_tokens = sum(count for _, _, count in trajectory_examples)
            examples.extend(
                (messages, weight * count / total_tokens)
                for messages, weight, count in trajectory_examples
            )

        self.examples = examples
        if skipped:
            logger.warning(f"Skipped {skipped} examples with no supervised tokens after truncation")
    
    def _compute_lengths_and_buckets(self):
        """
        Assign the lengths already computed during weight normalization to buckets.
        """
        self.bucket_indices = []
        
        bucket_counts = [0] * (len(self.bucket_boundaries) + 1)
        
        for length in self.lengths:
            bucket_idx = len(self.bucket_boundaries)
            for i, boundary in enumerate(self.bucket_boundaries):
                if length <= boundary:
                    bucket_idx = i
                    break
            self.bucket_indices.append(bucket_idx)
            bucket_counts[bucket_idx] += 1
        
        logger.info("Bucket distribution:")
        for i, count in enumerate(bucket_counts):
            if i == 0:
                bucket_range = f"<= {self.bucket_boundaries[0]}"
            elif i == len(self.bucket_boundaries):
                bucket_range = f"> {self.bucket_boundaries[-1]}"
            else:
                bucket_range = f"{self.bucket_boundaries[i-1]+1} - {self.bucket_boundaries[i]}"
            logger.info(f"  Bucket {i} ({bucket_range} tokens): {count} examples")
    
    def get_bucket_info(self) -> Tuple[List[int], List[int]]:
        """
        Returns (lengths, bucket_indices) for use by BucketSampler.
        """
        return self.lengths, self.bucket_indices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """Return raw messages and sample weight for the training collator."""
        messages, weight = self.examples[idx]

        try:
            return {
                "messages": messages,
                "sample_weight": weight
            }

        except Exception as e:
            logger.error(f"Error processing example {idx}: {e}")
            return {
                "messages": [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": ""}
                ],
                "sample_weight": 1.0
            }
