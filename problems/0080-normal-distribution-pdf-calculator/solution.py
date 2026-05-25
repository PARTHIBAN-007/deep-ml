import math

def normal_pdf(x, mean, std_dev):
	var = std_dev**2
	phi = 22/7
	exp = math.exp(-((x-mean)**2)/(2*var))
	normal = 1 / ((2*phi*var)**0.5)
	pdf = normal * exp
	return round(pdf,5)
