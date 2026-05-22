import torch

def multiquery_attention(X: torch.Tensor, W_queries: list, W_key: torch.Tensor, W_value: torch.Tensor, W_out: torch.Tensor) -> torch.Tensor:
    n_head = len(W_queries)
    seq_len = X.shape[0]
    d_k = W_key.shape[1]

    Q = torch.stack([X @ query for query in W_queries])
    K = X @ W_key
    V = X @ W_value

    K = K.unsqueeze(0).expand(n_head,-1,-1)
    V = V.unsqueeze(0).expand(n_head,-1,-1)

    attn_scores = Q @ K.transpose(1,2)
    attn_weights = torch.softmax(attn_scores/(d_k**0.5),dim=-1)
    context_vector = attn_weights @ V

    d_v = V.shape[-1]
    context_vector = context_vector.permute(1,0,2).contiguous().view(seq_len,n_head* d_v)
    output = context_vector @ W_out
    return torch.round(output  , decimals=4)
