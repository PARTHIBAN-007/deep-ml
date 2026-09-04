def phi_corr(x: list[int], y: list[int]) -> float:
	x00 = x01 = x10 = x11 = 0
	for num_x,num_y in zip(x,y):
		x00 += 1 if num_x == 0 and num_y == 0 else 0 
		x01 += 1 if num_x == 0 and num_y == 1 else 0
		x10 += 1 if num_x == 1 and num_y == 0 else 0
		x11 += 1 if num_x == 1 and num_y == 1 else 0
	try:
		val = ((x00*x11) - (x01*x10)) / (((x00+x01) * (x10 + x11) * (x00 + x10) * (x01 + x11))**0.5 )
	except:
		val = 0

	return round(val,4)