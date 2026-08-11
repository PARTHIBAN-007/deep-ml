import torch
import torch.nn as nn

def single_neuron_forward(x):
    neuron = nn.Linear(3, 1)
    with torch.no_grad():
        neuron.weight.copy_(torch.tensor([[0.5, -0.2, 0.3]]))
        neuron.bias.copy_(torch.tensor([0.1]))
    output = neuron(x)
    return output.item()
    
