import numpy as np

def jensen_shannon_divergence(P: list[float], Q: list[float]) -> float:
	M = [(p+q)/2 for p,q in zip(P,Q)]
	kl_p = sum(p*np.log(p/q) if p/q!=0 else 0 for p,q in zip(P,M))
	kl_m = sum(p*np.log(p/q) if p/q!=0 else 0 for p,q in zip(Q,M))
	res = 0.5 * kl_p + 0.5* kl_m
	return res