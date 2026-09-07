import numpy as np

def soft_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(w) * np.maximum(np.abs(w) - threshold, 0.0)

def l1_regularization_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4) -> tuple:
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for i in range(max_iter):
        y_pred = X @ weights + bias
        error = y_pred - y

        weight_gradient = (1 / n_samples) * (X.T @ error)
        bias_gradient = (1 / n_samples) * np.sum(error)

        new_weights = weights - learning_rate * weight_gradient
        new_weights = soft_threshold(new_weights, learning_rate * alpha)
        new_bias = bias - learning_rate * bias_gradient

        if (np.linalg.norm(new_weights - weights) < tol and abs(new_bias - bias) < tol):
            weights = new_weights
            bias = new_bias
            break

        weights = new_weights
        bias = new_bias

    return weights, bias