import torch
import torch.nn as nn


def apply_conv2d():
    torch.manual_seed(0)
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, bias=False)
    weight = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    input_tensor = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])
    with torch.no_grad():
        conv.weight.copy_(weight)
        output = conv(input_tensor)
    return output