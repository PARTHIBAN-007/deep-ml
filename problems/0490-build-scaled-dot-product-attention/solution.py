import numpy as np

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple:
    def softmax(x):
        shifted_x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    attention_score = Q @ K.T
    d_k = Q.shape[-1]
    scaled_attention = attention_score/np.sqrt(d_k)
    if mask is not None:
        scaled_attention = np.where(mask == 0, -1e9, scaled_attention)
    attention_weights = softmax(scaled_attention)
    context_weights = attention_weights @ V
    return context_weights, attention_weights