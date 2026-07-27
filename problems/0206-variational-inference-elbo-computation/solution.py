import numpy as np

def compute_elbo(x: list[float], q_mean: float, q_std: float, 
                 prior_mean: float, prior_std: float,
                 likelihood_std: float, n_samples: int = 1000) -> float:
    z_samples = np.random.normal(q_mean, q_std, n_samples)
    x_arr = np.array(x)
    diff = x_arr - z_samples[:, np.newaxis]
    log_likelihoods = -0.5 * np.log(2 * np.pi * (likelihood_std ** 2)) - (diff ** 2) / (2 * (likelihood_std ** 2))
    e_log_likelihood = np.mean(np.sum(log_likelihoods, axis=1))
    log_priors = -0.5 * np.log(2 * np.pi * (prior_std ** 2)) - ((z_samples - prior_mean) ** 2) / (2 * (prior_std ** 2))
    e_log_prior = np.mean(log_priors)
    entropy = 0.5 * np.log(2 * 3.14 * np.exp(1) * (q_std**2))
    elbo = e_log_likelihood - e_log_prior
    elbo_value = e_log_likelihood + e_log_prior + entropy
    return float(elbo_value)