import numpy as np

def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    B, C, H, W = X.shape
    channels_per_group = C // num_groups
    X_reshaped = X.reshape(B, num_groups, channels_per_group, H, W)
    mean = np.mean(X_reshaped, axis=(2, 3, 4), keepdims=True)
    var = np.var(X_reshaped, axis=(2, 3, 4), keepdims=True)
    X_norm = (X_reshaped - mean) / np.sqrt(var + epsilon)
    X_norm = X_norm.reshape(B, C, H, W)
    gamma = np.array(gamma).reshape(1, C, 1, 1)
    beta = np.array(beta).reshape(1, C, 1, 1)
    out = X_norm * gamma + beta
    return out