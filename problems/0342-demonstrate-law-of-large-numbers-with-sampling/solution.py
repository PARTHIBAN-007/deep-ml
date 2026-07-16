import numpy as np

def law_of_large_numbers(n_samples: int, population_mean: float, population_std: float) -> float:
    samples = np.random.normal(population_mean, population_std,n_samples)
    return np.mean(samples)