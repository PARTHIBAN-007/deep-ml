import numpy as np

def nag_optimizer(parameter, grad_fn, velocity, learning_rate=0.01, momentum=0.9):
    w_param = parameter - momentum * velocity
    grad = grad_fn(w_param)
    velocity = momentum * velocity + learning_rate * grad
    parameter = parameter - velocity
    return np.round(parameter, 5), np.round(velocity, 5)