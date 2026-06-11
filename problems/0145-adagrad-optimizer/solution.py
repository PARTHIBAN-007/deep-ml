import numpy as np

def adagrad_optimizer(parameter, grad, G, learning_rate=0.01, epsilon=1e-8):
    G = G  + grad **2
    parameter = parameter - (learning_rate/((G + epsilon))**0.5) * grad
    return np.round(parameter, 5), np.round(G, 5)