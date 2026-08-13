import torch

def bce_with_logits(logits, targets):
    loss = torch.clamp(logits, min=0) - logits * targets + torch.log(1 + torch.exp(-torch.abs(logits)))
    mean_loss = loss.mean().item()
    return round(mean_loss, 4)