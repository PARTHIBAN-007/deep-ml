import torch

def mse(pred, target):
    return torch.mean((pred-target)**2).item()