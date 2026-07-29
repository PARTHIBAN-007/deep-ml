import numpy as np

def pdf(x, mean, variance):
    coef = 1.0 / np.sqrt(2 * np.pi * variance)
    return coef * np.exp(-0.5 * ((x - mean) ** 2) / variance)

def fit_gmm_1d(X, K, initial_means, initial_variances, initial_weights, n_iterations):
    X = np.array(X)
    means = np.array(initial_means, dtype=float)
    variances = np.array(initial_variances, dtype=float)
    weights = np.array(initial_weights, dtype=float)
    
    N = len(X)
    for _ in range(n_iterations):
        pdf_matrix = np.column_stack([pdf(X, means[i], variances[i]) for i in range(K)])
        weighted_pdfs = pdf_matrix * weights
        
        base = np.sum(weighted_pdfs, axis=1, keepdims=True)
        base = np.maximum(base, 1e-10)
        
        responsibilities = weighted_pdfs / base
        
        effective_counts = np.sum(responsibilities, axis=0)
        
        for i in range(K):
            if effective_counts[i] > 1e-10:
                means[i] = np.sum(responsibilities[:, i] * X) / effective_counts[i]
                variances[i] = np.sum(responsibilities[:, i] * (X - means[i]) ** 2) / effective_counts[i]
                variances[i] = max(variances[i], 1e-6)
                
        weights = effective_counts / N
        
    return {'means': np.round(means,4).tolist(), 'variances': np.round(variances,4).tolist(), 'weights': np.round(weights,4).tolist()}

        

    