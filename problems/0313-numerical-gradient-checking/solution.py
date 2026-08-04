import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    numerical_grad = np.zeros_like(x)
    x_flat = x.ravel()
    num_grad_flat = numerical_grad.ravel()

    for i in range(x_flat.size):
        old_val = x_flat[i]

        x_flat[i] = old_val + epsilon
        f_plus = f(x)

        x_flat[i] = old_val - epsilon
        f_minus = f(x)

        x_flat[i] = old_val

        num_grad_flat[i] = (f_plus - f_minus) / (2 * epsilon)

    numerator = np.linalg.norm(numerical_grad - analytical_grad)
    denominator = np.linalg.norm(numerical_grad) + np.linalg.norm(analytical_grad)

    if denominator == 0:
        relative_error = 0.0
    else:
        relative_error = numerator / denominator

    return numerical_grad, relative_error