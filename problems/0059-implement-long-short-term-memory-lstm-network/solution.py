import torch
import torch.nn as nn
class LSTM:
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wf = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wi = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wc = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wo = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)

        self.bf = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bi = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bc = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bo = torch.zeros(hidden_size, 1, dtype=torch.float64)

    def forward(self, x: torch.Tensor, initial_hidden_state: torch.Tensor, initial_cell_state: torch.Tensor):
        h_t = initial_hidden_state
        c_t = initial_cell_state
        outputs = []
        for xx in x:
            xx = xx.unsqueeze(-1)
            hidden = torch.cat((h_t,xx),dim=0)
            f_t = torch.sigmoid(self.Wf @ hidden + self.bf)
            i_t = torch.sigmoid((self.Wi @  hidden + self.bi))
            c_tidle = torch.tanh(self.Wc @  hidden + self.bc)
            c_t = f_t * c_t + i_t * c_tidle
            o_t = torch.sigmoid(self.Wo @ hidden + self.bo)
            h_t = o_t * torch.tanh(c_t)  
            outputs.append(h_t)
        return torch.stack(outputs),h_t,c_t
