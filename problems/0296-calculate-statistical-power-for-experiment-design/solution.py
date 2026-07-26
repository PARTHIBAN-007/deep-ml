import math
from statistics import NormalDist

def calculate_power(effect_size: float, sample_size_per_group: int, alpha: float = 0.05, two_tailed: bool = True) -> float:
    def cdf(x):
        return 0.5 * ( 1 + math.erf(x/(2**0.5)))
    d = effect_size
    ncp = d * (sample_size_per_group/2)**0.5
    if two_tailed:
        power = 1 - cdf(NormalDist().inv_cdf(1 - alpha / 2) - ncp) + cdf( -NormalDist().inv_cdf(1 - alpha / 2)-ncp)
    else:
        power = 1 - cdf(NormalDist().inv_cdf(1 - alpha) - ncp)
    return round(power,4)