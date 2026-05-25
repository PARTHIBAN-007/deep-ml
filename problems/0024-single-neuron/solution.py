import math
import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	n = len(features)
	y_pred = []
	for feature in features:
		pred = np.dot(feature,weights) + bias
		y_pred.append(sigmoid(pred))
	mse = 0
	for pred,label in zip(y_pred,labels):
		error = label - pred
		squared_error = error**2
		mse += squared_error
	mse = mse/n
	
	return (y_pred,mse)
