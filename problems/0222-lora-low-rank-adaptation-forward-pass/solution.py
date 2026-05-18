import numpy as np

def lora_forward(
    x: list[list[float]],
    W: list[list[float]],
    A: list[list[float]],
    B: list[list[float]],
    alpha: float = 1.0
) -> list[list[float]]:

    x_np = np.array(x)
    W_np = np.array(W)
    A_np = np.array(A)
    B_np = np.array(B)

    base_output = np.dot(x_np, W_np)
    rank = B_np.shape[1]
    lora_adaptation = np.dot(np.dot(x_np, B_np), A_np)
    scaling = alpha / rank

    result = base_output + (scaling * lora_adaptation)

    return result.tolist()