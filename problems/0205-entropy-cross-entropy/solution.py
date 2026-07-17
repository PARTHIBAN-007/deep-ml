import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	entropy = sum(-p * np.log(p) if p!=0 else 0 for p in P)
	cross_entropy = sum(-p*np.log(q) for p,q in zip(P,Q))
	return (entropy,cross_entropy)