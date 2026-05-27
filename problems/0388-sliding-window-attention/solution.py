import torch

def sliding_window_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, window_size: int) -> torch.Tensor:
    seq_len , d_k = Q.shape
    attn_scores = Q @ K.T
    top_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=window_size + 1)
    bottom_mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=-window_size - 1)
    mask = (top_mask + bottom_mask).bool()
    attn_scores.masked_fill_(mask,float('-inf'))
    attn_weights = torch.softmax(attn_scores / (d_k**0.5),dim=-1)
    context_vector = attn_weights @ V
    return context_vector