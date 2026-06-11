def rmsprop_update(params: list[float], grads: list[float], cache: list[float], 
                   lr: float = 0.01, beta: float = 0.9, epsilon: float = 1e-8) -> tuple[list[float], list[float]]:
	c_t = []
	p_t = []
	for c,g,p in zip(cache,grads,params):
		c = beta * c + (1-beta)*(g**2)
		c_t.append(c)
		p = p - (lr/(c**0.5))*(g)
		p_t.append(p)
	return p_t ,  c_t