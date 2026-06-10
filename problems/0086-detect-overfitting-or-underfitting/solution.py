def model_fit_quality(training_accuracy, test_accuracy):
	diff = abs(training_accuracy - test_accuracy)
	if diff>0.2:
		return 1
	elif training_accuracy<0.7 and test_accuracy<0.7:
		return -1
	else:
		return 0