import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    n_samples = X.shape[0]
    weights = weights.copy()
    
    for _ in range(n_epochs):
        if method == 'batch':
            predictions = np.dot(X, weights)
            errors = predictions - y
            gradient = (2 / n_samples) * np.dot(X.T, errors)
            weights -= learning_rate * gradient
            
        elif method == 'stochastic':
            for i in range(n_samples):
                xi = X[i]
                yi = y[i]
                prediction = np.dot(xi, weights)
                error = prediction - yi
                gradient = 2 * xi * error
                weights -= learning_rate * gradient
                
        elif method == 'mini_batch':
            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                batch_n = X_batch.shape[0]
                predictions = np.dot(X_batch, weights)
                errors = predictions - y_batch
                gradient = (2 / batch_n) * np.dot(X_batch.T, errors)
                weights -= learning_rate * gradient
                
    return weights