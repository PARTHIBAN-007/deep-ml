def dice_statistics(n: int) -> tuple[float, float]:
	expectation = round((n+1)/2,4)
	variance = round((((n**2)-1)/12),4)
	return (expectation,variance)