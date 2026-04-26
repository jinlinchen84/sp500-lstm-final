from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

EXT_DIR = ROOT / "extension"
TABLE_DIR = EXT_DIR / "tables"
FIGURE_DIR = EXT_DIR / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

def add_enhanced_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)

    g = df.groupby("permno")

    df["past_5d_ret"] = g["ret"].transform(
        lambda x: (1 + x).rolling(5, min_periods=5).apply(np.prod, raw=True) - 1
    )

    df["vol_20d"] = g["ret"].transform(
        lambda x: x.rolling(20, min_periods=20).std()
    )

    # train set
    feature_cols_raw = ["past_5d_ret", "vol_20d"]

    for col in feature_cols_raw:
        mean = df.loc[df["is_train"] == True, col].mean()
        std = df.loc[df["is_train"] == True, col].std()

        df[col + "_z"] = (df[col] - mean) / std

    return df

def build_feature_sequences(df, feature_cols, seq_len=240):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)

    usable = df[df["available_for_feature_generation"] == True].copy()

    X, y, meta = [], [], []

    for permno, group in usable.groupby("permno"):
        group = group.sort_values("date").reset_index(drop=True)

        values = group[feature_cols].values
        labels = group["label_t1"].values
        dates = group["date"].values

        for i in range(seq_len - 1, len(group)):
            seq = values[i - seq_len + 1 : i + 1]
            label = labels[i]

            if (
                seq.shape == (seq_len, len(feature_cols))
                and not np.isnan(seq).any()
                and not np.isnan(label)
            ):
                X.append(seq)
                y.append(int(label))
                meta.append({"date": dates[i], "permno": permno})

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), pd.DataFrame(meta)

class FeatureLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=25, dropout=0.16, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out
    
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_feature_lstm(X_train, y_train, device, input_size=3,
                       hidden_size=25, dropout=0.16,
                       max_epochs=100, patience=10,
                       val_ratio=0.2, batch_size=512):

    n = len(X_train)
    n_val = int(n * val_ratio)
    n_tr = n - n_val

    X_tr, X_val = X_train[:n_tr], X_train[n_tr:]
    y_tr, y_val = y_train[:n_tr], y_train[n_tr:]

    train_loader = DataLoader(
        SequenceDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        SequenceDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False
    )

    model = FeatureLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_weights = None
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    model.load_state_dict(best_weights)
    return model

def predict_feature_lstm(model, X, device, batch_size=512):
    model.eval()
    loader = DataLoader(
        SequenceDataset(X, np.zeros(len(X), dtype=np.int64)),
        batch_size=batch_size,
        shuffle=False
    )

    probs = []

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            out = model(xb)
            prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            probs.append(prob)

    return np.concatenate(probs)

# =========================
# Feature-LSTM Extension Runner
# =========================

MAX_PERIODS = 23      # For a quick test, set this to 1 first
MAX_EPOCHS = 50      # Increase to 100 if you want a stronger final run
SEQ_LEN = 240
K = 10
COST_BPS = 5

FEATURE_COLS = [
    "ret_z",
    "past_5d_ret_z",
    "vol_20d_z",
]


def build_feature_sequences_for_dates(df, feature_cols, target_dates=None, seq_len=240):
    """
    Build feature sequences.

    If target_dates is provided, only return sequences whose ending date is in target_dates.
    This is important for trade-period prediction: we need train history + trade days
    but only output predictions for trade dates.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)

    if target_dates is not None:
        target_dates = set(pd.to_datetime(target_dates))

    usable = df[df["available_for_feature_generation"] == True].copy()

    X, y, meta = [], [], []

    for permno, group in usable.groupby("permno"):
        group = group.sort_values("date").reset_index(drop=True)

        values = group[feature_cols].values
        labels = group["label_t1"].values
        dates = pd.to_datetime(group["date"]).values

        for i in range(seq_len - 1, len(group)):
            d = pd.Timestamp(dates[i])

            if target_dates is not None and d not in target_dates:
                continue

            seq = values[i - seq_len + 1 : i + 1]
            label = labels[i]

            if (
                seq.shape == (seq_len, len(feature_cols))
                and not np.isnan(seq).any()
                and not np.isnan(label)
            ):
                X.append(seq)
                y.append(int(label))
                meta.append({
                    "date": d,
                    "permno": str(permno),
                    "true_label": int(label),
                })

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int64),
        pd.DataFrame(meta),
    )


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_feature_lstm_predictions():
    """
    Train Feature-LSTM period by period and generate daily predictions.
    """
    device = get_device()
    print(f"Using device: {device}")

    all_predictions = []
    accuracy_rows = []

    for period in range(1, MAX_PERIODS + 1):
        print(f"\n=== Feature-LSTM Period {period}/{MAX_PERIODS} ===")

        train_path = ROOT / f"output/train_samples_period_{period:02d}.parquet"
        trade_path = ROOT / f"output/trade_samples_period_{period:02d}.parquet"

        if not train_path.exists():
            raise FileNotFoundError(f"Missing file: {train_path}")
        if not trade_path.exists():
            raise FileNotFoundError(f"Missing file: {trade_path}")

        train_df = pd.read_parquet(train_path)
        trade_df = pd.read_parquet(trade_path)

        train_df["is_train"] = True
        train_df["is_trade"] = False

        trade_df["is_train"] = False
        trade_df["is_trade"] = True

        train_df["period"] = period
        trade_df["period"] = period

        # Combine train + trade so trade sequences can use train history
        combined = pd.concat([train_df, trade_df], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])
        combined["permno"] = combined["permno"].astype(str)

        combined = add_enhanced_features(combined)

        train_dates = pd.to_datetime(train_df["date"].unique())
        trade_dates = pd.to_datetime(trade_df["date"].unique())

        X_train, y_train, _ = build_feature_sequences_for_dates(
            combined,
            feature_cols=FEATURE_COLS,
            target_dates=train_dates,
            seq_len=SEQ_LEN,
        )

        X_trade, y_trade, meta_trade = build_feature_sequences_for_dates(
            combined,
            feature_cols=FEATURE_COLS,
            target_dates=trade_dates,
            seq_len=SEQ_LEN,
        )

        print("X_train shape:", X_train.shape)
        print("X_trade shape:", X_trade.shape)

        if len(X_train) == 0 or len(X_trade) == 0:
            print(f"Skipping period {period}: not enough sequences.")
            continue

        model = train_feature_lstm(
            X_train,
            y_train,
            device=device,
            input_size=len(FEATURE_COLS),
            max_epochs=MAX_EPOCHS,
            patience=10,
        )

        probs = predict_feature_lstm(model, X_trade, device=device)

        meta_trade["period"] = period
        meta_trade["feature_lstm_prob"] = probs

        acc = ((meta_trade["feature_lstm_prob"] >= 0.5).astype(int) == meta_trade["true_label"]).mean()

        print(f"Period {period} Feature-LSTM accuracy: {acc:.4f}")

        accuracy_rows.append({
            "period": period,
            "feature_lstm_accuracy": acc,
            "n_predictions": len(meta_trade),
        })

        all_predictions.append(meta_trade)

    if not all_predictions:
        raise RuntimeError("No predictions generated. Check input data and sequence construction.")

    pred_df = pd.concat(all_predictions, ignore_index=True)
    acc_df = pd.DataFrame(accuracy_rows)

    pred_path = TABLE_DIR / "feature_lstm_predictions.parquet"
    acc_path = TABLE_DIR / "feature_lstm_accuracy_by_period.csv"

    pred_df.to_parquet(pred_path, index=False)
    acc_df.to_csv(acc_path, index=False)

    print(f"\nSaved predictions to: {pred_path}")
    print(f"Saved accuracy table to: {acc_path}")

    return pred_df, acc_df


def load_trade_returns():
    all_trade = []

    for period in range(1, MAX_PERIODS + 1):
        trade_path = ROOT / f"output/trade_samples_period_{period:02d}.parquet"
        trade_df = pd.read_parquet(trade_path)

        trade_df["date"] = pd.to_datetime(trade_df["date"])
        trade_df["period"] = period
        trade_df["permno"] = trade_df["permno"].astype(str)

        all_trade.append(
            trade_df[["date", "permno", "period", "fwd_ret_1d"]].copy()
        )

    return pd.concat(all_trade, ignore_index=True)


def load_baseline_lstm_predictions():
    baseline_path = ROOT / "predictions_daily/lstm_all_periods.parquet"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline LSTM predictions: {baseline_path}")

    baseline = pd.read_parquet(baseline_path)

    baseline["date"] = pd.to_datetime(baseline["date"])
    baseline["permno"] = baseline["permno"].astype(str)

    baseline = baseline[baseline["period"] <= MAX_PERIODS].copy()

    return baseline[["date", "permno", "period", "lstm_prob"]]


def run_long_short_backtest(df, signal_col, k=10, cost_bps=5, ascending=False):
    """
    Generic long-short backtest.

    ascending=False:
        Higher signal = long, lower signal = short.
        Use this for LSTM probability.

    ascending=True:
        Lower signal = long, higher signal = short.
    """
    results = []
    prev_long = set()
    prev_short = set()

    for date, group in df.groupby("date"):
        group = group.dropna(subset=[signal_col, "fwd_ret_1d"]).copy()

        if len(group) < 2 * k:
            continue

        group = group.sort_values(signal_col, ascending=ascending).reset_index(drop=True)

        long = group.head(k)
        short = group.tail(k)

        long_stocks = set(long["permno"])
        short_stocks = set(short["permno"])

        long_ret = long["fwd_ret_1d"].mean()
        short_ret = short["fwd_ret_1d"].mean()
        gross_ret = long_ret - short_ret

        if prev_long:
            long_turnover = len(long_stocks - prev_long) / k
            short_turnover = len(short_stocks - prev_short) / k
            turnover = (long_turnover + short_turnover) / 2
        else:
            turnover = 1.0

        cost = turnover * cost_bps / 10000
        net_ret = gross_ret - cost

        results.append({
            "date": date,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "gross_ret": gross_ret,
            "turnover": turnover,
            "cost": cost,
            "net_ret": net_ret,
        })

        prev_long = long_stocks
        prev_short = short_stocks

    return pd.DataFrame(results)


def run_backtest_all_periods(df, signal_col, k=10, cost_bps=5, ascending=False):
    all_results = []

    for period in sorted(df["period"].dropna().unique()):
        period_df = df[df["period"] == period].copy()

        bt = run_long_short_backtest(
            period_df,
            signal_col=signal_col,
            k=k,
            cost_bps=cost_bps,
            ascending=ascending,
        )

        bt["period"] = period
        all_results.append(bt)

    return pd.concat(all_results, ignore_index=True)


def compute_metrics(bt):
    ret = bt["net_ret"].dropna()

    cumulative = (1 + ret).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1

    return {
        "Mean Gross Return (%)": bt["gross_ret"].mean() * 100,
        "Mean Net Return (%)": ret.mean() * 100,
        "Volatility (%)": ret.std() * 100,
        "Sharpe Ratio": ret.mean() / ret.std() * np.sqrt(252),
        "Max Drawdown (%)": drawdown.min() * 100,
        "Win Rate (%)": (ret > 0).mean() * 100,
        "Average Turnover": bt["turnover"].mean(),
    }


def compare_feature_lstm_to_baseline(feature_pred_df):
    trade_rets = load_trade_returns()

    feature_df = feature_pred_df.merge(
        trade_rets,
        on=["date", "permno", "period"],
        how="left",
    )

    baseline_df = load_baseline_lstm_predictions().merge(
        trade_rets,
        on=["date", "permno", "period"],
        how="left",
    )

    feature_bt = run_backtest_all_periods(
        feature_df,
        signal_col="feature_lstm_prob",
        k=K,
        cost_bps=COST_BPS,
        ascending=False,
    )

    baseline_bt = run_backtest_all_periods(
        baseline_df,
        signal_col="lstm_prob",
        k=K,
        cost_bps=COST_BPS,
        ascending=False,
    )

    feature_bt["strategy"] = "Feature-LSTM"
    baseline_bt["strategy"] = "Baseline LSTM"

    feature_bt.to_csv(TABLE_DIR / "feature_lstm_daily_returns.csv", index=False)
    baseline_bt.to_csv(TABLE_DIR / "baseline_lstm_for_feature_comparison_daily_returns.csv", index=False)

    rows = []

    for name, bt in [
        ("Baseline LSTM", baseline_bt),
        ("Feature-LSTM", feature_bt),
    ]:
        metrics = compute_metrics(bt)
        metrics["Model"] = name
        rows.append(metrics)

    summary = pd.DataFrame(rows)

    summary = summary[
        [
            "Model",
            "Mean Gross Return (%)",
            "Mean Net Return (%)",
            "Volatility (%)",
            "Sharpe Ratio",
            "Max Drawdown (%)",
            "Win Rate (%)",
            "Average Turnover",
        ]
    ]

    summary_path = TABLE_DIR / "feature_lstm_vs_baseline_performance.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved performance comparison to: {summary_path}")

    return baseline_bt, feature_bt, summary


def plot_feature_lstm_vs_baseline(baseline_bt, feature_bt):
    fig, ax = plt.subplots(figsize=(10, 5))

    baseline_cum = baseline_bt["net_ret"].cumsum() * 100
    feature_cum = feature_bt["net_ret"].cumsum() * 100

    ax.plot(baseline_bt["date"], baseline_cum, label="Baseline LSTM", linewidth=2)
    ax.plot(feature_bt["date"], feature_cum, label="Feature-LSTM", linewidth=2)

    ax.axhline(0, linestyle="--", alpha=0.5)
    ax.set_title("Baseline LSTM vs Feature-Enhanced LSTM")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Net Return (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig_path = FIGURE_DIR / "feature_lstm_vs_baseline.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"Saved figure to: {fig_path}")


def main():
    print("Feature-LSTM extension script started.")
    print(f"Project root: {ROOT}")
    print(f"Tables will be saved to: {TABLE_DIR}")
    print(f"Figures will be saved to: {FIGURE_DIR}")
    print(f"MAX_PERIODS = {MAX_PERIODS}")
    print(f"MAX_EPOCHS = {MAX_EPOCHS}")
    print(f"Features = {FEATURE_COLS}")

    feature_pred_df, acc_df = run_feature_lstm_predictions()

    print("\nFeature-LSTM accuracy summary:")
    print(acc_df.round(4))
    print(f"Mean Feature-LSTM accuracy: {acc_df['feature_lstm_accuracy'].mean():.4f}")

    baseline_bt, feature_bt, summary = compare_feature_lstm_to_baseline(feature_pred_df)

    print("\nFeature-LSTM vs Baseline LSTM performance:")
    print(summary.round(4))

    plot_feature_lstm_vs_baseline(baseline_bt, feature_bt)

    print("\nDone. Feature-LSTM extension outputs saved to extension/.")


if __name__ == "__main__":
    main()