import torch
import torch.nn.functional as F

def grouped_query_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, num_heads: int, num_kv_heads: int) -> torch.Tensor:
    batch_size,seq_len,d_model = Q.shape
    head_dim = d_model//num_heads
    Q = Q.view(batch_size,seq_len,num_heads,head_dim).transpose(1,2)
    K = K.view(batch_size,seq_len,num_kv_heads,head_dim).transpose(1,2)
    V = V.view(batch_size,seq_len,num_kv_heads,head_dim).transpose(1,2)
    queries_per_kv = num_heads//num_kv_heads
    
    if num_heads!=num_kv_heads:
        K = K.unsqueeze(2).expand(-1,-1,queries_per_kv,-1,-1)
        K = K.reshape(batch_size,queries_per_kv*num_kv_heads,seq_len,head_dim)

        V = V.unsqueeze(2).expand(-1,-1,queries_per_kv,-1,-1)
        V = V.reshape(batch_size,queries_per_kv*num_kv_heads,seq_len,head_dim)

    attn_scores = Q @ K.transpose(2,3)
    attn_weights = torch.softmax(attn_scores/(head_dim**0.5),dim=-1)
    context_vector = attn_weights @ V
    context_vector = context_vector.transpose(1,2)
    return context_vector.contiguous().view(batch_size,seq_len,d_model)