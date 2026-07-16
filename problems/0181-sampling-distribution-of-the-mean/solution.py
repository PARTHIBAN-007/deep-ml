import numpy as np

def simulate_clt(num_samples: int, sample_size: int, distribution: str = 'uniform') -> float:
    avg = []
    for _ in range(num_samples):
        if distribution=="uniform":
            samples = np.random.uniform(0,1,sample_size)
        else:
            samples = np.random.exponential(1,sample_size)
        avg.append(np.mean(samples))
    return np.mean(avg)
