import torch

def grad_wss(w_list, x_list):
    x = torch.tensor(x_list)
    w = torch.tensor(w_list)
    res = []
    for w,x in zip(w_list,x_list):
        res.append(w*(x**2))
    return res

