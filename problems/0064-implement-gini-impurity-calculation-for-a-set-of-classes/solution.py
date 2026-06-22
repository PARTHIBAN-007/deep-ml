
import numpy as np
def gini_impurity(y):
	n = len(y)
	mpp = {}
	for num in y:
		if num not in mpp:
			mpp[num] = 1
		else:
			mpp[num]+=1
	prob = []
	for key,val in mpp.items():
		prob.append(val/n)
	prob = sum([p**2 for p in prob])
	val = 1 - prob
	return round(val,3)