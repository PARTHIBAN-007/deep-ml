import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = []
    n_train = X_train.size(0)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        total_train_loss = 0.0

        for i in range(0, n_train, batch_size):
            indices = perm[i:i + batch_size]
            X_batch = X_train[indices]
            y_batch = y_train[indices]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * X_batch.size(0)

        train_loss = total_train_loss / n_train

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            preds = val_outputs.argmax(dim=1)
            val_accuracy = (preds == y_val).float().mean().item()

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })

    return history