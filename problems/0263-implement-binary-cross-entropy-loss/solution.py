
import math
def binary_cross_entropy(y_true: list[float], y_pred: list[float], epsilon: float = 1e-15) -> float:
	loss = 0
	n = len(y_true)
	for true,pred in zip(y_true,y_pred):
		loss += - ( true*math.log(pred) + (1-true)*math.log(1-pred))
	return loss/n