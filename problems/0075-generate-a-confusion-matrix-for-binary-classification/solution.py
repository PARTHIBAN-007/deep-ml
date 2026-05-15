
from collections import Counter

def confusion_matrix(data):
	conf = [[0,0],[0,0]]
	for true,pred in data:
		if true==1 and pred==1:
			conf[0][0]+=1
		if true==0 and pred==0:
			conf[1][1] += 1
		if true==0 and pred==1:
			conf[1][0]+=1
		if true==1 and pred==0:
			conf[0][1]+=1
	return conf

