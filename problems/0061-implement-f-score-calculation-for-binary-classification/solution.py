import numpy as np

def f_score(y_true, y_pred, beta):
	true_positive = 0
	false_positive = 0
	false_negative = 0
	for true,pred in zip(y_true,y_pred):
		if true==1 and pred==1:
			true_positive+=1
		if true==1 and pred==0:
			false_negative+=1
		if true==0 and pred==1:
			false_positive+=1
	precision = (true_positive)/(true_positive+false_positive)
	recall = (true_positive)/(true_positive+false_negative)
	f_score = (1+(beta**2)) * ((precision*recall)/((beta**2 * precision)+recall))
	return round(f_score,3)