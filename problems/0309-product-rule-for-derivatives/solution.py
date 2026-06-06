import numpy as np
from collections import defaultdict
def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:
    f = []
    g = []
    for i,f_co in enumerate(f_coeffs):
        f.append((f_co,i))
    for i,g_co in enumerate(g_coeffs):
        g.append((g_co,i))
    mpp = defaultdict(list)
    for num1,x in f:
        for num2,y in g:
            co = x+y
            num = num1 * num2
            mpp[co].append(num)
    nums = []
    for k,v in mpp.items():
        nums.append((k,sum(v)))
    nums.sort(key = lambda x: x[0])
    res = []
    for x,num in nums:
        if x!=0:
            res.append(round(x*num,4)) 
    return res if len(res)!=0 else [0.0]

            
