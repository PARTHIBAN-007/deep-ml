import numpy as np

def cross_product(a, b):
    v1 = (a[1]*b[2])-(a[2]*b[1])
    v2 = (a[2]*b[0])-(a[0]*b[2])
    v3 = (a[0]*b[1])-(a[1]*b[0])
    return [v1,v2,v3]
