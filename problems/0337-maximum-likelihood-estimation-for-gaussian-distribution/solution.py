import numpy as np

def gaussian_mle(data: np.ndarray) -> tuple:
    mu_mle = float(np.mean(data))
    var_mle = float(np.var(data, ddof=0))
    return mu_mle, var_mle