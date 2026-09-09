import numpy as np

def train(X, y, W, b):
    num_samples = X.shape[0]
    learning_rate = 0.001
    for _ in range(10000):
        y_pred = X @ W + b
        residuals = y - y_pred
        w_grad = - (1 / num_samples) * (X.T @ residuals) 
        b_grad = - (1 / num_samples) * np.sum(residuals)
        W = W - learning_rate * w_grad
        b = b - learning_rate * b_grad
    return (W,b)
