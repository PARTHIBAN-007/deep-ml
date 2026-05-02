import torch
import torch.nn as nn
import torch.nn.functional as F
def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = torch.matmul(X, W_q)
    K = torch.matmul(X, W_k)
    V = torch.matmul(X, W_v)
    return Q, K, V

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    d_k = K.shape[-1]
    attn_scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    attn_weights = F.softmax(attn_scores, dim=-1)
    context_vector = attn_weights @ V
    return context_vector
