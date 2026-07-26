import numpy as np

def bayesian_inference_beta_binomial(prior_alpha: float, prior_beta: float, successes: int, trials: int) -> tuple[float, float, float]:
	alpha = prior_alpha + successes
	beta = prior_beta + (trials - successes)
	mean = alpha / (alpha + beta)
	return [alpha,beta,mean]