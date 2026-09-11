import numpy as np

def bhattacharyya_distance(p: list[float], q: list[float]) -> float:
    n,m = len(p) , len(q)
    if n!=m or n==0 or m==0:
        return 0.0
    dist = 0
    for num1,num2 in zip(p,q):  
        dist += (num1 * num2)**0.5
    return -np.log(dist)