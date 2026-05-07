def min_max(x: list[float]) -> list[float]:
    maxi = max(x)
    mini = min(x)
    res = []
    for num in x:
        scale = (num-mini)/(maxi-mini)
        res.append(scale)
    return res