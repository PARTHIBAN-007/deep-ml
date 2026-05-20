import torch
import torch.nn.functional as F
from typing import Tuple

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return (Q, K, V)

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    d_k = K.shape[-1]
    attention_scores = (Q @ K.t()) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    attention_weights = F.softmax(attention_scores, dim=-1)
    context_vector = attention_weights @ V
    return context_vector

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:
    seq_len, d_model = Q.shape
    head_dim = d_model // n_heads
    
    Q = Q.view(seq_len, n_heads, head_dim).transpose(0, 1)
    K = K.view(seq_len, n_heads, head_dim).transpose(0, 1)
    V = V.view(seq_len, n_heads, head_dim).transpose(0, 1)
    
    head_outputs = []
    for i in range(n_heads):
        output = self_attention(Q[i], K[i], V[i])
        head_outputs.append(output)
    
    combined = torch.stack(head_outputs, dim=0)
    combined = combined.transpose(0, 1)
    output = combined.reshape(seq_len, d_model)
    return output