import numpy as np
from collections import defaultdict
def k_nearest_neighbors(points, query_point, k):
    mpp = defaultdict(list)
    for point in points:
        distance = 0
        for i in range(len(point)):
            distance += (point[i]-query_point[i])**2
        distance = distance**0.5
        mpp[distance].append(point)
    mpp = dict(sorted(mpp.items()))
    res = []
    for key,val in mpp.items():
        for point in val:
            if k<=0:
                break
            else:
                k-=1
                res.append(point)
    return res




