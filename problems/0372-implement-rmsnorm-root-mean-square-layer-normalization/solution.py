import numpy as np

def rmsnorm(x: np.ndarray, g: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    rms = np.sqrt(np.mean(x**2,axis=-1,keepdims = True) + eps)
    rms =  x / rms
    return rms * g