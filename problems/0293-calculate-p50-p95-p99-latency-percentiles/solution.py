import numpy as np

def calculate_latency_percentiles(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {
        "P50": 0.0,
        "P95":0.0,
        "P99": 0.0
    }

    percentile = np.nanquantile(latencies,[0.50,0.95,0.99])
    return {
        "P50": np.round(percentile[0],4),
        "P95": np.round(percentile[1],4),
        "P99": np.round(percentile[2],4)
    }