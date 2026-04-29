import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .models import SequenceDataset


def build_sequences(df, seq_len=240):
    """
    Build 240-step return sequences for LSTM input.
    Only use rows where available_for_feature_generation=True.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['permno', 'date']).reset_index(drop=True)

    usable = df[df['available_for_feature_generation'] == True].copy()

    sequences = []
    labels = []
    meta = []

    for permno, group in usable.groupby('permno'):
        group = group.sort_values('date').reset_index(drop=True)
        ret_z = group['ret_z'].values
        label = group['label_t1'].values
        dates = group['date'].values

        for i in range(seq_len - 1, len(group)):
            seq = ret_z[i - seq_len + 1: i + 1]
            lbl = label[i]
            if len(seq) == seq_len and not np.isnan(seq).any() and not np.isnan(lbl):
                sequences.append(seq)
                labels.append(int(lbl))
                meta.append({'date': dates[i], 'permno': permno})

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    meta_df = pd.DataFrame(meta)

    return X, y, meta_df


def predict_all_trade_days(model, train_df, trade_df, device, seq_len=240, batch_size=512):
    """
    Generate predictions for every stock on every trading day.
    Uses train data to extend history for full 240-step sequences.
    """
    train_df = train_df.copy()
    trade_df = trade_df.copy()

    train_df['date'] = pd.to_datetime(train_df['date'])
    trade_df['date'] = pd.to_datetime(trade_df['date'])

    trade_dates = set(pd.to_datetime(trade_df['date'].unique()))

    combined = pd.concat([train_df, trade_df], ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])
    combined = combined.sort_values(['permno', 'date']).reset_index(drop=True)

    results = []

    for permno, group in combined.groupby('permno'):
        group = group.sort_values('date').reset_index(drop=True)
        ret_z = group['ret_z'].values
        dates = group['date'].values
        labels = group['label_t1'].values

        for i in range(seq_len - 1, len(group)):
            d = pd.Timestamp(dates[i])
            if d not in trade_dates:
                continue
            seq = ret_z[i - seq_len + 1: i + 1]
            if len(seq) == seq_len and not np.isnan(seq).any():
                lbl = labels[i]
                results.append({
                    'date': d,
                    'permno': permno,
                    'seq': seq,
                    'true_label': float(lbl) if lbl is not None else -1
                })

    if not results:
        return pd.DataFrame()

    X = np.array([r['seq'] for r in results], dtype=np.float32)
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

    meta = pd.DataFrame([{
        'date': r['date'],
        'permno': r['permno'],
        'true_label': r['true_label']
    } for r in results])
    meta['lstm_prob'] = np.concatenate(probs)

    return meta
