import numpy as np

def gaussian_elimination(A, b):
	A = np.array(A,dtype = float)
	b = np.array(b,dtype = float)
	n = len(b)

	for i in range(n):
		max_row = np.argmax(np.abs(A[i:n,i])) + i
		if i!=max_row:
			A[[i,max_row]] = A[[max_row,i]]
			b[[i,max_row]] = b[[max_row,i]]
		
		
		for j in range(i+1,n):
			factor = A[j,i] / A[i,i]
			A[j,i:] -= factor * A[i,i:]
			b[j] -= factor * b[i]
	
	x = np.zeros(n)
	for i in range(n-1,-1,-1):
		x[i] = (b[i] - np.dot(A[i,i+1:],x[i+1:])) / A[i,i]
	return x