def calculate_f1_score(y_true, y_pred):
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
	try:
		precision = (true_positive)/(true_positive+false_positive)
	except:
		precision = 0.0
	try:
		recall = (true_positive)/(true_positive+false_negative)
	except:
		recall = 0.0
	try:
		f1_score =  (2*precision*recall)/(precision+recall)
	except:
		f1_score = 0.0
	return round(f1_score,3)