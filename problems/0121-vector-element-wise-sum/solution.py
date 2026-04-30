def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
	if len(a)!=len(b):
		return -1
	res = []
	for v1,v2 in zip(a,b):
		res.append(v1+v2)
	return res
	