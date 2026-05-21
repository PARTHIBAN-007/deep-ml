import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def SwiGLU(x: np.ndarray) -> np.ndarray:
    n,m = x.shape
    X1 = x[:,:m//2]
    X2 = x[:,m//2:]
    res = []
    for x1,x2 in zip(X1,X2):
        swish = x2 * sigmoid(x2)
        swiglu = x1 * swish
        res.append(swiglu)
    return np.round(res,4)