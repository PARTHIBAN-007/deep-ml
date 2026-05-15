import numpy as np

def accuracy_score(y_true, y_pred):
	accuracy = 0
	for true,pred in zip(y_true,y_pred):
		if true==pred:
			accuracy+=1
	return accuracy/len(y_true)