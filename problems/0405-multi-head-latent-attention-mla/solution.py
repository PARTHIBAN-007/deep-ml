import torch

def multi_head_latent_attention(
    X: torch.Tensor,
    W_dkv: torch.Tensor,
    W_uk: torch.Tensor,
    W_uv: torch.Tensor,
    W_dq: torch.Tensor,
    W_uq: torch.Tensor,
    W_o: torch.Tensor,
    n_heads: int
) -> tuple:
    seq_len , d_model = X.shape
    head_dim = d_model // n_heads
    c_q = X @ W_dq
    c_kv = X @ W_dkv
    Q = c_q @ W_uq
    K = c_kv  @ W_uk
    V = c_kv @ W_uv

    Q = Q.view(seq_len,n_heads,head_dim).transpose(0,1)
    K = K.view(seq_len,n_heads,head_dim).transpose(0,1)
    V = V.view(seq_len,n_heads,head_dim).transpose(0,1)

    attn_scores = Q @ K.transpose(1,2)
    attn_weights = torch.softmax(attn_scores/(head_dim**0.5),dim=-1)
    context_vector = attn_weights @ V
    context_vector = context_vector.transpose(0,1).contiguous().view(seq_len,d_model)
    final_context_vector = context_vector @ W_o
    return (final_context_vector,c_kv)