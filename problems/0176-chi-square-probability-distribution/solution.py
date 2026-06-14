import math

def chi_square_probability(x, k):
    if x<0:
        return 0.0
    denominator = (2**(k/2)) * math.gamma(k/2)
    numerator = (x**((k/2)-1)) * math.exp(-x/2)
    probability = numerator / denominator
    return round(probability, 3)