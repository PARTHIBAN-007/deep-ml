import torch

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    return torch.matmul(X, W_q), torch.matmul(X, W_k), torch.matmul(X, W_v)

def masked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    d_k = Q.shape[-1]
    attention_scores = (Q @ K.t())/torch.sqrt(torch.tensor(d_k,dtype = torch.float32))
    masked_scores = attention_scores + mask
    attention_weights = torch.softmax(masked_scores, dim=-1)
    context_vector = attention_weights @ V
    return context_vector