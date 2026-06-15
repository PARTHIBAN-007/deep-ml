import numpy as np

def mutual_information(joint_prob: list[list[float]]) -> float:
	n = len(joint_prob)
	px = []
	py = []
	for prob in joint_prob:
		px.append(sum(prob))
	
	for i in range(n):
		summ = 0
		for j in range(n):
			summ+=joint_prob[j][i]
		py.append(summ)
	
	res = 0
	for i,prob in enumerate(joint_prob):
		p_x , p_y = px[i] , py[i]
		for p in prob:
			res += p * (np.log((p/(p_x * p_y)) + 0.0000001))
	return res

