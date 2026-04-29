import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def build_benchmark_features(df, train_df=None, seq_len=240):
    """
    Build multi-period cumulative return features for benchmark models.
    Following Krauss et al. (2017): m in {1,...,20, 40, 60,...,240}
    If train_df is provided, use it to extend history for trade period.
    """
    if train_df is not None:
        df_combined = pd.concat([train_df, df], ignore_index=True)
    else:
        df_combined = df.copy()

    df_combined['date'] = pd.to_datetime(df_combined['date'])
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    trade_dates = set(df['date'].unique())

    periods = list(range(1, 21)) + list(range(40, 241, 20))

    features = []
    labels = []
    meta = []

    for permno, group in df_combined.groupby('permno'):
        group = group.sort_values('date').reset_index(drop=True)
        ret = group['ret'].values
        label = group['label_t1'].values
        dates = group['date'].values

        for i in range(max(periods), len(group)):
            if pd.Timestamp(dates[i]) not in trade_dates:
                continue
            if pd.isna(label[i]):
                continue
            row = []
            valid = True
            for m in periods:
                cum_ret = np.prod(1 + ret[i-m:i]) - 1
                if np.isnan(cum_ret):
                    valid = False
                    break
                row.append(cum_ret)
            if valid:
                features.append(row)
                labels.append(label[i])
                meta.append({'date': dates[i], 'permno': permno})

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    meta_df = pd.DataFrame(meta)
    return X, y, meta_df


def train_logistic(X_train, y_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=100, solver='lbfgs', C=1.0)
    clf.fit(X_train_scaled, y_train)
    return clf.predict_proba(X_test_scaled)[:, 1]


def train_random_forest(X_train, y_train, X_test):
    clf = RandomForestClassifier(n_estimators=1000, max_depth=20, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


class DNNModel(nn.Module):
    def __init__(self, input_size=31, num_classes=2):
        super(DNNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 31),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(31, 10),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(5, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class _TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_dnn(X_train, y_train, device, max_epochs=100, patience=10,
              val_ratio=0.2, batch_size=512):
    n = len(X_train)
    n_val = int(n * val_ratio)
    n_tr = n - n_val

    X_tr, X_val = X_train[:n_tr], X_train[n_tr:]
    y_tr, y_val = y_train[:n_tr], y_train[n_tr:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    tr_loader = DataLoader(_TabularDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_TabularDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = DNNModel(input_size=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_weights = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for X_batch, y_batch in tr_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item()
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
    return model, scaler


def predict_proba_dnn(model, scaler, X, device, batch_size=512):
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            out = model(batch)
            prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            probs.append(prob)
    return np.concatenate(probs)
