import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    arr = arr.flatten()
    if norm_type == "frobenius":
        return np.sqrt(sum(arr**2))
    if norm_type == "l1":
        return sum(abs(arr))
    if norm_type=="l2":
        return np.sqrt(sum(arr**2))