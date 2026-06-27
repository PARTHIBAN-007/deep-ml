import numpy as np

def adadelta_optimizer(parameter, grad, u, v, rho=0.95, epsilon=1e-6):
    parameter = np.asarray(parameter, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    updated_u = rho * u + (1 - rho) * (grad ** 2)
    delta_parameter = - (np.sqrt(v + epsilon) / np.sqrt(updated_u + epsilon)) * grad
    updated_v = rho * v + (1 - rho) * (delta_parameter ** 2)
    updated_parameter = parameter + delta_parameter

    return (
        np.round(updated_parameter, 5), 
        np.round(updated_u, 5), 
        np.round(updated_v, 5)
    )