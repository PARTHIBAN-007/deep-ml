import numpy as np
def precision(y_true, y_pred):
	true_positive = 0
	false_positive = 0
	for true,pred in zip(y_true,y_pred):
		if true==1 and pred==1:
			true_positive +=1
		if true==0 and pred==1:
			false_positive+=1
	return (true_positive)/(true_positive+false_positive)