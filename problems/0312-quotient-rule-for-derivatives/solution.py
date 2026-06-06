import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    u = 0
    v = 0
    for n,p in enumerate(reversed(g_coeffs)):
        u += p *(x**n)
    for n,p in enumerate(reversed(h_coeffs)):
        v += p *(x**n)
    u_ = 0
    v_ = 0
    for n,p in enumerate(reversed(g_coeffs)):
        try:
            exp = x ** (n-1)
        except:
            exp = x
        u_ += n*p*exp
    for n,p in enumerate(reversed(h_coeffs)):
        try:
            exp = x ** (n-1)
        except:
            exp = x
        v_ += n*p*exp
    res = (u_*v - u*v_)/(v**2)
    return res
    
    