import numpy as np

def multivariate_kl_divergence(mu_p: np.ndarray, Cov_p: np.ndarray, mu_q: np.ndarray, Cov_q: np.ndarray) -> float:
    d = len(mu_p)
    
    sign_p, logdet_p = np.linalg.slogdet(Cov_p)
    sign_q, logdet_q = np.linalg.slogdet(Cov_q)
    log_det_term = logdet_q - logdet_p
    
    inv_cov_q = np.linalg.inv(Cov_q)
    
    trace_term = np.trace(inv_cov_q @ Cov_p)
    
    diff = mu_q - mu_p
    mahalanobis_term = diff.T @ inv_cov_q @ diff
    
    kl_div = 0.5 * (log_det_term - d + trace_term + mahalanobis_term)
    
    return float(kl_div)