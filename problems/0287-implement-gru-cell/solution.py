import torch
import torch.nn 
def gru_cell(x: torch.Tensor, h_prev: torch.Tensor,
             W_z: torch.Tensor, U_z: torch.Tensor, b_z: torch.Tensor,
             W_r: torch.Tensor, U_r: torch.Tensor, b_r: torch.Tensor,
             W_h: torch.Tensor, U_h: torch.Tensor, b_h: torch.Tensor) -> torch.Tensor:
    z_t = torch.nn.functional.sigmoid(W_z @ x + U_z @ h_prev + b_z)
    r_t = torch.nn.functional.sigmoid(W_r @ x + U_r @ h_prev + b_r)
    h_t = torch.nn.functional.tanh(W_h @  x  +  U_h @ (r_t * h_prev) + b_h)
    h_t = (1 - z_t) * h_prev + z_t * h_t
    return h_t