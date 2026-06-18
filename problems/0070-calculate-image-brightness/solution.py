import numpy as np
def calculate_brightness(img):
	if not img:
		return -1
	n,m = len(img) , len(img[0])
	summ = 0
	for i in range(n):
		col_length = len(img[i])
		if col_length!=m:
			return -1
		for j in range(m):
			if img[i][j]>255 or img[i][j]<0:
				return -1
			summ += img[i][j]
	mean = summ / (n*m)
	return round(mean,2)
	
	