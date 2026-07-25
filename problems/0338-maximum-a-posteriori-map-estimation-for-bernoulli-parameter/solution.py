import numpy as np

def map_estimate_bernoulli(observations: list, alpha: float, beta: float) -> float:
    k = np.sum(observations)
    n = len(observations)
    _map = (alpha + k -1)/(alpha+beta+n-2)
    return _map