import numpy as np
from collections import defaultdict
def compute_tf_idf(corpus, query):
	n = len(corpus)
	df = 0
	mpp = defaultdict(lambda : defaultdict(int))
	df = []
	for i,document in enumerate(corpus):
		for word in document:
			mpp[i][word] +=1
		
	for word in query:
		cnt = 0
		for i in range(len(corpus)):
			if mpp[i][word]>=1:
				cnt += 1
		df.append(cnt)
			
	output = []
	for i,document in enumerate(corpus):
		res = []
		for j,target in enumerate(query):
			t = mpp[i][target]
			d = len(document)
			tf = t/d
			idf = (n+1) / (df[j] + 1) 
			tf_idf = tf * (np.log(idf) + 1)
			res.append(round(tf_idf,5))
		output.append(res)
	return output
	
				