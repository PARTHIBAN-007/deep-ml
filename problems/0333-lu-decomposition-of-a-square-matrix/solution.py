import numpy as np

def lu_decomposition(A: list) -> tuple:
	A = np.array(A,dtype = float)
	n = A.shape[0]

	L = np.eye(n)
	U = np.zeros((n,n))

	for i in range(n):
		for j in range(i,n):
			sum_u = sum(L[i,k]*U[k,j] for k in range(i))
			U[i,j] = A[i,j] - sum_u
		
		for j in range(i+1,n):
			sum_l = sum(L[j,k]*U[k,i] for k in range(i))
			L[j,i] = (A[j,i] -sum_l) / U[i,i]
	return L,U
