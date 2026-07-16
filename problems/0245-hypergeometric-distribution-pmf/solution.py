import math
def hypergeometric_pmf(N: int, K: int, n: int, k: int) -> float:
    pmf = (math.comb(K,k) * math.comb(N-K,n-k)) / (math.comb(N,n))
    return pmf