import torch

def sgd_step(w, grad, lr):
    return w - lr * grad