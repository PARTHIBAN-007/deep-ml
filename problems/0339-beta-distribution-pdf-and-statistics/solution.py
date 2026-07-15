import math


def beta_distribution_stats(x: float, alpha: float, beta_param: float) -> dict:
    mean = alpha / (alpha + beta_param)
    variance = (alpha * beta_param) / (((alpha + beta_param)**2)*(alpha + beta_param + 1))
    if 0<=x<=1:
        beta_f = (math.gamma(alpha) * math.gamma(beta_param)) / math.gamma(alpha + beta_param)
        pdf = ((x ** (alpha - 1)) * ((1 - x) ** (beta_param - 1))) / beta_f
    else:
        pdf = 0.0
    
    return {
        "pdf": pdf,
        "mean": mean,
        "variance": variance
    }