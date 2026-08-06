import torch

def make_tensor():
    nums = [[1,2,3],[4,5,6]]
    return torch.tensor(nums,dtype = torch.float32)