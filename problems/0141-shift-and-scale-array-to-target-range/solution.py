import numpy as np    


def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    a = np.min(values)
    b = np.max(values)
    res = []
    f_x = lambda x: c + ((d-c)/(b-a))*(x-a)
    res = f_x(values)
    return res
    