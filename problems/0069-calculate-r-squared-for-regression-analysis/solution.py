
import numpy as np

def r_squared(y_true, y_pred):
	mean = np.mean(y_pred)
	ssr = np.sum(np.power(y_true-y_pred,2))
	sst = np.sum(np.power(y_true-mean,2))
	return 1 - (ssr/sst)