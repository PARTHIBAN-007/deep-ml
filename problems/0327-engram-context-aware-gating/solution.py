import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))


def engram_context_gating(h: np.ndarray, e: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    t,d = h.shape
    k = e @ W_K
    v = e @ W_V
    h_rms = np.sqrt(np.mean(h**2) + eps )
    k_rms = np.sqrt(np.mean(k**2) + eps )
    h_norm = h / h_rms
    k_norm = k / k_rms
    dot = np.sum(h_norm * k_norm) / np.sqrt(d)
    gate = sigmoid(dot)
    output = gate * v
    return output

