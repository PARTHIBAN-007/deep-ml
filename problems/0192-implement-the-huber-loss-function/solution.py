def huber_loss(y_true, y_pred, delta=1.0):
	if not hasattr(y_true, '__iter__'):
		y_true = [y_true]
	if not hasattr(y_pred, '__iter__'):
		y_pred = [y_pred]
	res = 0
	n = 0
	for y,p in zip(y_true,y_pred):
		diff = abs(y-p)
		if diff<delta:
			loss = 0.5*((y-p)**2)
		else:
			loss = delta * (abs(y-p) - 0.5 * delta)
		res+= loss
		n+=1
	return res/n

