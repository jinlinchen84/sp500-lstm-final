import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=25, dropout=0.16, num_classes=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


def train_lstm(X_train, y_train, device, hidden_size=25, dropout=0.16,
               max_epochs=1000, patience=10, val_ratio=0.2, batch_size=512):
    n = len(X_train)
    n_val = int(n * val_ratio)
    n_tr = n - n_val

    X_tr, X_val = X_train[:n_tr], X_train[n_tr:]
    y_tr, y_val = y_train[:n_tr], y_train[n_tr:]

    tr_loader = DataLoader(SequenceDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = LSTMModel(hidden_size=hidden_size, dropout=dropout).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_weights = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for X_batch, y_batch in tr_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out = model(X_batch)
                val_loss += criterion(out, y_batch).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    model.load_state_dict(best_weights)
    return model


def predict_proba_lstm(model, X, device, batch_size=512):
    y_dummy = np.zeros(len(X), dtype=np.int64)
    dataset = SequenceDataset(X, y_dummy)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs = []
    model.eval()
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            out = model(X_batch)
            prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            probs.append(prob)
    return np.concatenate(probs)
