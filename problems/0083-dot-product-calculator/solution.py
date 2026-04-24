import numpy as np

def calculate_dot_product(vec1, vec2):
	dot_product = 0
	for u,v in zip(vec1,vec2):
		dot_product += u*v
	return dot_product