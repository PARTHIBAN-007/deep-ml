import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape
	n,m = input_height + 2 *padding , input_width + 2*padding
	x = [[0]*m for _ in range(n)]
	for i in range(input_height):
		for j in range(input_width):
			# print(i,j,i+1,j+1)
			# print("------------")
			x[i+padding][j+padding] = input_matrix[i][j]
	res = []
	for i in range(0,n,stride):
		layer = []
		for j in range(0,m,stride):
			if i+kernel_height<=n and j+kernel_width<=m:
				summ = 0
				for k1 in range(kernel_height):
					for k2 in range(kernel_width):
						summ += kernel[k1][k2] * x[i+k1][j+k2]
				layer.append(summ)
		if layer:
			res.append(layer)
	return res
