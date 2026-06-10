import numpy as np

def kernel_function(x1, x2):
	res = 0
	for num1,num2 in zip(x1,x2):
		res += num1 * num2
	return res