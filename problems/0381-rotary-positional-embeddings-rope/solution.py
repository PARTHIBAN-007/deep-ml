import numpy as np

def apply_rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    seq_len, d = x.shape
    i = np.arange(d // 2)
    frequencies = 1.0 / (base ** (2 * i / d))
    angles = np.outer(positions, frequencies)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]
    x_even_new = x_even * cos_angles - x_odd * sin_angles
    x_odd_new = x_even * sin_angles + x_odd * cos_angles
    x_rope = np.empty_like(x)
    x_rope[:, 0::2] = x_even_new
    x_rope[:, 1::2] = x_odd_new
    
    return x_rope