import torch
import torch.nn as nn
import torch.nn.functional as F

def noisy_topk_gating(
    X: torch.Tensor,
    W_g: torch.Tensor,
    W_noise: torch.Tensor,
    N: torch.Tensor,
    k: int
) -> torch.Tensor:
    h_base = X @ W_g
    h_noise = X @ W_noise
    noise_scale = torch.nn.functional.softplus(h_noise)
    noisy_logits = h_base + N * noise_scale
    topk_values, topk_indices = torch.topk(noisy_logits, k, dim=-1)
    mask_logits = torch.full_like(noisy_logits, float('-inf'))
    mask_logits.scatter_(dim=-1, index=topk_indices, src=topk_values)
    gating_probs = F.softmax(mask_logits, dim=-1)
    return gating_probs