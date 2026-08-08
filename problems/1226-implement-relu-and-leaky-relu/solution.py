import torch

def relu(t):
    return torch.clamp(t,min = 0)

def leaky_relu(t, slope=0.01):
    return torch.where(t > 0, t, slope * t)