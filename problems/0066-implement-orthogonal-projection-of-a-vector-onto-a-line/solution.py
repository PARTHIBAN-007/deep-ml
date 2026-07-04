
def orthogonal_projection(v, L):
	projection = []
	for u,v in zip(v,L):
		try:
			point = ((u*v) / (v**2))*v
		except:
			point = 0
		projection.append(point)
	return projection
	