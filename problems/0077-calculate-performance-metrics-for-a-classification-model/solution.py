
def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
	tp, tn , fp,fn = 0,0,0,0
	for true,pred in zip(actual,predicted):
		if true==1 and pred==1:
			tp+=1
		elif true==0 and pred==0:
			tn += 1
		elif true ==0 and pred==1:
			fp += 1
		else:
			fn +=1
	confusion_matrix = [[tp,fn],[fp,tn]]
	accuracy = (tp+tn)/(tp+tn+fp+fn)
	precision = (tp)/(tp+fp)
	recall = tp/(tp+fn)
	f1 = (2*precision*recall)/(precision+recall)
	specificity = tn/(tn+fp)
	negative_predictive = tn/(tn+fn)

	return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negative_predictive, 3)
