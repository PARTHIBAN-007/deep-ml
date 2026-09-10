import numpy as np

def activation(x):
    x = np.array(x)
    shape = x.shape
    nums = x.flatten()
    activation = np.array([num + np.exp(num) for num in nums])
    return activation.reshape(shape)