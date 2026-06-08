import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.w_q = nn.Linear(d_model,d_model,bias = False)
        self.w_k = nn.Linear(d_model,d_model,bias = False)
        self.w_v = nn.Linear(d_model,d_model,bias = False)
        self.W_o = nn.Linear(d_model,d_model,bias = False)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

    def forward(self, x, mask=None):
        batch, seq_len ,d_model = x.shape
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        Q = Q.view(batch,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        K = K.view(batch,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        V = V.view(batch,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        attn_scores = Q @ K.transpose(2,3)
        if mask is not None:
            attn_scores = attn_scores + mask
        attn_weights = torch.softmax(attn_scores/(self.head_dim**0.5), dim=-1)
        context_vector = attn_weights @ V
        context_vector = context_vector.transpose(1,2).contiguous().view(batch,seq_len,d_model)
        final_context_vector = self.W_o(context_vector)
        return final_context_vector