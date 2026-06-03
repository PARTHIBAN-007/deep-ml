from collections import defaultdict
def empirical_pmf(samples):
    mpp = defaultdict(int)
    for num in samples:
        mpp[num] += 1
    n = len(samples)
    res = []
    for key,val in mpp.items():
        res.append((key,val/n))
    return res
    