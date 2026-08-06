import torch

def reshape_transpose(t):
    tensor =  torch.reshape(t,(2,3))
    return tensor.T