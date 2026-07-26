import numpy as np
import math
from statistics import NormalDist

# scipy is not available in this environment. The two helpers below compute
# the regularized incomplete beta function via the Numerical Recipes
# continued-fraction algorithm — you'll need them to compute the p-value
# from the t-distribution.
def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function."""
    max_iter, eps = 200, 3e-12
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps: break
    return h

def _betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b

def t_cdf(t, df):
    """
    CDF of Student's t-distribution.
    """
    x = df / (df + t * t)
    a = df / 2.0
    b = 0.5

    ib = _betainc(a, b, x)

    if t >= 0:
        return 1.0 - 0.5 * ib
    else:
        return 0.5 * ib

def two_sample_t_test(sample1: list[float], sample2: list[float],
                      alpha: float = 0.05) -> dict:
    n1 , n2 = len(sample1) , len(sample2)
    sample1 , sample2 = np.array(sample1), np.array(sample2)
    mean1, mean2 = np.mean(sample1) , np.mean(sample2)
    var1, var2 = np.var(sample1,ddof = 1) , np.var(sample2,ddof = 1) 
    standard_error = ((var1/n1) + (var2/n2))**0.5
    t_stat = (mean1 - mean2) / standard_error
    df = (((var1/n1) + (var2/n2))**2) / (((var1/n1)**2)/(n1-1) + ((var2/n2)**2)/(n2-1))
    p_val =  2 * (1 - t_cdf(abs(t_stat), df))
    reject_null = True if p_val<alpha else False
    s_pooled = (((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))**0.5
    cohens_d = (mean1 - mean2)/s_pooled
    return {
        "t_statistic": t_stat,
        "p_value": p_val,
        "degrees_of_freedom": df,
        "reject_null": reject_null,
        "cohens_d": cohens_d
    }

