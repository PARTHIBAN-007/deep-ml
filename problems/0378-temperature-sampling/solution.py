import numpy as np

def temperature_sampling(logits: np.ndarray, temperature: float) -> list:
	if temperature <= 0:
		res = np.zeros_like(logits, dtype=float)
		max_idx = np.argmax(logits)
		res[max_idx] = 1.0
		return res.tolist()

	scaled = logits / temperature
	shifted = scaled - np.max(scaled)
	exp_vals = np.exp(shifted)
	probs = exp_vals / np.sum(exp_vals)
	return probs.tolist()