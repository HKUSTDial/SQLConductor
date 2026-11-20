import torch
from torch.utils.data import Sampler
from typing import List, Optional
from loguru import logger
import random

class BucketSampler(Sampler):
    """
    Sampler that groups examples into buckets by sequence length,
    ensuring each batch only contains examples from the same bucket.
    
    Args:
        lengths: List of tokenized lengths for each example
        bucket_indices: List of bucket assignments for each example
        batch_size: Number of examples per batch
        shuffle: Whether to shuffle within buckets and across batches
        drop_last: Whether to drop the last incomplete batch
    """
    
    def __init__(
        self,
        lengths: List[int],
        bucket_indices: List[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: Optional[int] = None
    ):
        if len(lengths) != len(bucket_indices):
            raise ValueError(f"lengths and bucket_indices must have same length, got {len(lengths)} and {len(bucket_indices)}")
        
        self.lengths = lengths
        self.bucket_indices = bucket_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        
        self.bucket_to_indices = {}
        for idx, bucket_idx in enumerate(bucket_indices):
            if bucket_idx == -1:
                continue
            if bucket_idx not in self.bucket_to_indices:
                self.bucket_to_indices[bucket_idx] = []
            self.bucket_to_indices[bucket_idx].append(idx)
        
        logger.info(f"BucketSampler initialized with {len(self.bucket_to_indices)} buckets")
        for bucket_idx in sorted(self.bucket_to_indices.keys()):
            count = len(self.bucket_to_indices[bucket_idx])
            logger.info(f"  Bucket {bucket_idx}: {count} examples")
    
    def __iter__(self):
        """
        Generate batches, ensuring each batch contains only examples from the same bucket.
        """
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
        
        all_batches = []
        
        for bucket_idx in sorted(self.bucket_to_indices.keys()):
            indices = self.bucket_to_indices[bucket_idx].copy()
            
            if self.shuffle:
                random.shuffle(indices)
            
            bucket_batches = []
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    bucket_batches.append(batch)
            
            all_batches.extend(bucket_batches)
        
        if self.shuffle:
            random.shuffle(all_batches)
        
        for batch in all_batches:
            yield from batch
    
    def __len__(self):
        """
        Return the number of samples that will be yielded.
        Only counts valid samples (bucket_idx != -1), not invalid ones.
        """
        total_valid_samples = sum(len(indices) for indices in self.bucket_to_indices.values())
        
        if self.drop_last:
            total_batches = 0
            for bucket_idx in self.bucket_to_indices.keys():
                bucket_size = len(self.bucket_to_indices[bucket_idx])
                total_batches += bucket_size // self.batch_size
            return total_batches * self.batch_size
        else:
            return total_valid_samples

