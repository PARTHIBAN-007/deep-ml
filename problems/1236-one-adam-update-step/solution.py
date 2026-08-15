import torch

def adam_step(w, grad, m, v, t, lr, beta1, beta2, eps):
    m_new = beta1 * m + (1- beta1) * grad
    v_new = beta2*v + (1-beta2)* (grad**2)
    m_hat = m_new / (1-beta1**t)
    v_hat = v_new / (1 - beta2**t)
    w_new = w -lr * m_hat / (torch.sqrt(v_hat) + eps)
    return (w_new,m_new,v_new)