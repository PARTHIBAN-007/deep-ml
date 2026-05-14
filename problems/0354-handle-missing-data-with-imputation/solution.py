import numpy as np

def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    res = data.astype(float).copy()
    rows, cols = res.shape
    
    for c in range(cols):
        column = res[:, c]
        mask = ~np.isnan(column)
        valid_values = column[mask]
        
        if valid_values.size == 0:
            continue
            
        if strategy == 'mean':
            fill_value = np.mean(valid_values)
            
        elif strategy == 'median':
            fill_value = np.median(valid_values)
            
        elif strategy == 'mode':
            values, counts = np.unique(valid_values, return_counts=True)
            max_freq = np.max(counts)
            modes = values[counts == max_freq]
            fill_value = np.min(modes)
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        res[~mask, c] = fill_value
            
    return res