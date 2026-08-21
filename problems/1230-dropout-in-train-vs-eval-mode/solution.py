import torch
import torch.nn as nn


def dropout_demo():
    torch.manual_seed(0)
    
    x = torch.ones(10)
    drop = nn.Dropout(p=0.5)
    
    drop.eval()
    eval_output = drop(x)
    
    drop.train()
    train_output = drop(x)
    
    train_nonzero_count = int((train_output != 0).sum().item())
    
    return (eval_output, train_nonzero_count)