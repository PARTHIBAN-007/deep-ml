import numpy as np

def gemma_rmsnorm(x, eps=1e-6, scale=None):
    rms = np.sqrt(np.mean(x**2,axis=-1,keepdims = True)) + eps
    if scale is not None:
        y = (x/rms) * (1+scale)
    else:
        y = x/rms
    return y
