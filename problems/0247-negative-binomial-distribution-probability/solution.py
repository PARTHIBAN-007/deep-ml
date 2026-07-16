import math

def negative_binomial_pmf(k: int, r: int, p: float) -> float:
    pmf = math.comb(k+r-1,k) * (p**r ) *((1-p)**k)
    return pmf