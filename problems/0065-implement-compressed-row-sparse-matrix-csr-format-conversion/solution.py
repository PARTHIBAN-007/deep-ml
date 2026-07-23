import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
	vals = []
	col_idx = []
	row_ptr = [0]

	running_nnz = 0
	for row in dense_matrix:
		for c, val in enumerate(row):
			if val != 0:
				vals.append(val)
				col_idx.append(c)
				running_nnz += 1
		row_ptr.append(running_nnz)
		
	return vals, col_idx, row_ptr
