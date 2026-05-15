
import numpy as np

def rmse(y_true, y_pred):
	error = y_true-y_pred
	squared_error = error**2
	mean_squared_error = np.mean(squared_error)
	return round(np.sqrt(mean_squared_error),3)
	
