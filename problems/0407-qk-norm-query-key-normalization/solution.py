import numpy as np

def qk_norm_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, temperature: float = 1.0) -> tuple:
    eps = 1e-8
    q_norms = np.linalg.norm(Q, axis=-1, keepdims=True)
    k_norms = np.linalg.norm(K, axis=-1, keepdims=True)
    
    Q_norm = Q / np.maximum(q_norms, eps)
    K_norm = K / np.maximum(k_norms, eps)
    
    scores = (Q_norm @ K_norm.T) / temperature
    
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    attention_output = attention_weights @ V
    
    return attention_output, attention_weights