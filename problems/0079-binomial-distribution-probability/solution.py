import math

def binomial_probability(n: int, k: int, p: float) -> float:
    nCk = math.comb(n,k)
    p_k = p**k
    return nCk * p_k * ((1-p)**(n-k))