import torch

def train_step(model, x, y, optimizer, loss_fn):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y,y_pred)
    loss.backward()
    optimizer.step()
    return loss.item()

