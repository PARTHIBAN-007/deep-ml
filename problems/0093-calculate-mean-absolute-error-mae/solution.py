import numpy as np

def mae(y_true, y_pred):
    error = np.abs(y_true-y_pred)
    return np.mean(error)