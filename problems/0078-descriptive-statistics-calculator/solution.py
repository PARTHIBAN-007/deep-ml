import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    mean = np.mean(data)
    median = np.median(data)
    val, cnt = np.unique(data,return_counts = True)
    mode = val[np.argmax(cnt)]
    variance = np.var(data)
    std = np.std(data)
    percentile = np.quantile(data,[0.25,0.5,0.75])
    iqr = percentile[2] - percentile[0]
    return {
        "mean":mean,
        "median": median,
        "mode": mode,
        "variance": variance,
        "standard_deviation": std,
        "25th_percentile": percentile[0],
        "50th_percentile": median,
        "75th_percentile": percentile[2],
        "interquartile_range": iqr
    }
