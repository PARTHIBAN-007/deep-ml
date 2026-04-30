import numpy as np

def cosine_similarity(v1, v2):
	dot = np.dot(v1,v2)
	norm_1 = np.linalg.norm(v1)
	norm_2 = np.linalg.norm(v2)

	if norm_1==0 or norm_2==0:
		return None
	return dot/(norm_1*norm_2)	