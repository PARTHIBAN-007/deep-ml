import numpy as np

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> list:
    return x/(np.sum(x**2,axis=axis,keepdims = True)**0.5 + eps)
