import numpy as np

def simulate_clt(distribution: str, n: int, runs: int = 10000, seed: int = 42) -> dict:
    np.random.seed(seed)

    if distribution == "uniform":
        mu = 0.5
        sigma = np.sqrt(1 / 12)
        samples = np.random.uniform(0, 1, size=(runs, n))
    elif distribution == "exponential":
        mu = 1.0
        sigma = 1.0
        samples = np.random.exponential(1.0, size=(runs, n))
    elif distribution == "bernoulli":
        p = 0.3
        mu = p
        sigma = np.sqrt(p * (1 - p))
        samples = (np.random.uniform(0, 1, size=(runs, n)) < p).astype(int)
        
    sample_means = np.mean(samples, axis=1)
    
    standard_error = sigma / np.sqrt(n)
    z_scores = (sample_means - mu) / standard_error
    
    return {
        "mean": float(np.mean(z_scores)),
        "std": float(np.std(z_scores))
    }