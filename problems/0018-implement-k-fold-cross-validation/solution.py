import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    
    base_size = n_samples//k
    remainder = n_samples%k

    fold_sizes = np.full(k, base_size)
    fold_sizes[:remainder] += 1

    splits = []
    current_idx = 0
    
    for size in fold_sizes:
        start, end = current_idx, current_idx + size
        test_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])
        splits.append((train_indices.tolist(), test_indices.tolist())) 
        current_idx = end
    return splits