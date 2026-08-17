import torch

def softmax(t, dim):
    max_vals, _ = torch.max(t, dim=dim, keepdim=True)
    shifted_t = t - max_vals
    exp_t = torch.exp(shifted_t)
    sum_exp = torch.sum(exp_t, dim=dim, keepdim=True)    
    return exp_t / sum_exp