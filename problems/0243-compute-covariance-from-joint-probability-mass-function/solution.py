import numpy as np

def covariance_from_joint_pmf(x_values: list, y_values: list, joint_pmf: np.ndarray) -> float:
    joint_pmf = np.array(joint_pmf)
    x_arr = np.array(x_values)
    y_arr = np.array(y_values)
    
    pmf_x = np.sum(joint_pmf, axis=1)
    pmf_y = np.sum(joint_pmf, axis=0)
    
    expected_x = np.sum(x_arr * pmf_x)
    expected_y = np.sum(y_arr * pmf_y)
    
    xy_grid = np.outer(x_arr, y_arr)
    expected_xy = np.sum(xy_grid * joint_pmf)
    
    covariance = expected_xy - (expected_x * expected_y)
    
    return float(covariance)