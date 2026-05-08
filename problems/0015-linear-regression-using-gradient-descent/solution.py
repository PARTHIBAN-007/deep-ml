import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    m, n = X.shape
    y = y.reshape(-1, 1)  
    theta = np.zeros((n, 1))  

    for i in range(iterations):
        y_pred = np.dot(X,theta)
        mse = (1/2*m) * np.sum((y_pred-y)**2)
        d_loss = (1/m) * np.dot(X.T,y_pred-y)
        theta = theta - alpha * d_loss
    return theta.flatten()