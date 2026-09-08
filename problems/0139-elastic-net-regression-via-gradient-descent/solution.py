import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    num_samples = X.shape[0]
    weights = np.zeros(X.shape[1])
    bias = 0.0
    for _ in range(max_iter):
        y_pred = X @ weights + bias
        residuals = y - y_pred
        w_grad = - (1 / num_samples) * (X.T @ residuals) + alpha1 * np.sign(weights) + 2 * alpha2 * weights
        b_grad = - (1 / num_samples) * np.sum(residuals)
        if np.linalg.norm(w_grad) < tol:
            break
        weights = weights - learning_rate * w_grad
        bias = bias - learning_rate * b_grad
    return (weights,bias)
