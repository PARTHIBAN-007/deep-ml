import numpy as np

def svm_margin_width(w: np.ndarray) -> float:
    l2_norm =  np.sum((w**2))**0.5
    return 2/l2_norm