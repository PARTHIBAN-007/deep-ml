def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
	positive = 0
	for prior,likelihood in zip(priors,likelihoods):
		positive += prior * likelihood
	res = []
	for prior,likelihood in zip(priors,likelihoods):
		prob = (prior * likelihood )/ positive
		res.append(prob)
	return res