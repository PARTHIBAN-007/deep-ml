import numpy as np

def hinge_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    loss = 0
    n = len(y_true)
    for y,p in zip(y_true,y_pred):
        loss += max(0,1- y*p)
    return loss/n