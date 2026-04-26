from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "extension"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_trade_data():
    all_trade = []

    for period in range(1, 24):
        path = ROOT / f"output/trade_samples_period_{period:02d}.parquet"
        trade = pd.read_parquet(path)

        trade["date"] = pd.to_datetime(trade["date"])
        trade["period"] = period
        trade["permno"] = trade["permno"].astype(str)

        keep_cols = [
            "date",
            "permno",
            "period",
            "ret",
            "fwd_ret_1d",
        ]

        trade = trade[keep_cols].copy()
        all_trade.append(trade)

    trade_all = pd.concat(all_trade, ignore_index=True)
    trade_all = trade_all.sort_values(["period", "permno", "date"]).reset_index(drop=True)

    return trade_all


def add_reversal_signal(trade_all):
    """
    Signal: past 5-day cumulative return using information up to signal date t.
    Low past_5d_ret = recent loser -> long candidate.
    High past_5d_ret = recent winner -> short candidate.
    """
    df = trade_all.copy()
    df = df.sort_values(["period", "permno", "date"]).reset_index(drop=True)

    grouped = df.groupby(["period", "permno"])

    df["past_5d_ret"] = grouped["ret"].transform(
        lambda x: (1 + x).rolling(5, min_periods=5).apply(np.prod, raw=True) - 1
    )

    return df


def load_lstm_predictions():
    lstm = pd.read_parquet(ROOT / "predictions_daily/lstm_all_periods.parquet")

    lstm["date"] = pd.to_datetime(lstm["date"])
    lstm["permno"] = lstm["permno"].astype(str)

    keep_cols = ["date", "permno", "period", "lstm_prob"]

    return lstm[keep_cols].copy()


def prepare_extension_data():
    trade_all = load_trade_data()
    trade_all = add_reversal_signal(trade_all)

    lstm = load_lstm_predictions()

    # Align reversal strategy universe with LSTM prediction universe
    df = lstm.merge(
        trade_all,
        on=["date", "permno", "period"],
        how="left"
    )

    return df


def run_long_short_strategy(
    df,
    signal_col,
    k=10,
    cost_bps=5,
    ascending=False,
):
    """
    Generic daily long-short backtest.

    ascending=False:
        Higher signal = long, lower signal = short.
        Use this for LSTM probability.

    ascending=True:
        Lower signal = long, higher signal = short.
        Use this for short-term reversal, where low past 5d return means recent loser.
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


def run_strategy_all_periods(
    df,
    signal_col,
    k=10,
    cost_bps=5,
    ascending=False,
):
    all_results = []

    for period in sorted(df["period"].dropna().unique()):
        period_df = df[df["period"] == period].copy()

        bt = run_long_short_strategy(
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


def compare_lstm_vs_reversal(df):
    # Baseline LSTM strategy: high lstm_prob -> long
    lstm_bt = run_strategy_all_periods(
        df,
        signal_col="lstm_prob",
        k=10,
        cost_bps=5,
        ascending=False,
    )

    # Reversal extension: low past_5d_ret -> long, high past_5d_ret -> short
    reversal_bt = run_strategy_all_periods(
        df,
        signal_col="past_5d_ret",
        k=10,
        cost_bps=5,
        ascending=True,
    )

    lstm_bt["strategy"] = "LSTM"
    reversal_bt["strategy"] = "Short-Term Reversal"

    summary_rows = []

    for name, bt in [
        ("LSTM", lstm_bt),
        ("Short-Term Reversal", reversal_bt),
    ]:
        metrics = compute_metrics(bt)
        metrics["Strategy"] = name
        summary_rows.append(metrics)

    summary = pd.DataFrame(summary_rows)

    summary = summary[
        [
            "Strategy",
            "Mean Gross Return (%)",
            "Mean Net Return (%)",
            "Volatility (%)",
            "Sharpe Ratio",
            "Max Drawdown (%)",
            "Win Rate (%)",
            "Average Turnover",
        ]
    ]

    summary.to_csv(
        OUTPUT_DIR / "reversal_extension_performance.csv",
        index=False,
    )

    lstm_bt.to_csv(
        OUTPUT_DIR / "lstm_extension_daily_returns.csv",
        index=False,
    )

    reversal_bt.to_csv(
        OUTPUT_DIR / "reversal_extension_daily_returns.csv",
        index=False,
    )

    return lstm_bt, reversal_bt, summary


def regression_r2(x, y):
    clean = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(clean) < 5:
        return np.nan, np.nan, np.nan

    slope, intercept = np.polyfit(clean["x"], clean["y"], 1)
    pred = intercept + slope * clean["x"]

    ss_res = ((clean["y"] - pred) ** 2).sum()
    ss_tot = ((clean["y"] - clean["y"].mean()) ** 2).sum()

    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    return intercept, slope, r2


def explain_lstm_with_reversal(lstm_bt, reversal_bt):
    merged = lstm_bt.merge(
        reversal_bt,
        on=["date", "period"],
        suffixes=("_lstm", "_reversal"),
        how="inner",
    )

    # Convert short leg realized returns into short profits
    merged["short_profit_lstm"] = -merged["short_ret_lstm"]
    merged["short_profit_reversal"] = -merged["short_ret_reversal"]

    total_intercept, total_slope, total_r2 = regression_r2(
        merged["gross_ret_reversal"],
        merged["gross_ret_lstm"],
    )

    long_intercept, long_slope, long_r2 = regression_r2(
        merged["long_ret_reversal"],
        merged["long_ret_lstm"],
    )

    short_intercept, short_slope, short_r2 = regression_r2(
        merged["short_profit_reversal"],
        merged["short_profit_lstm"],
    )

    explain_df = pd.DataFrame({
        "Regression": [
            "LSTM total return on reversal total return",
            "LSTM long leg on reversal long leg",
            "LSTM short profit on reversal short profit",
        ],
        "Intercept": [
            total_intercept,
            long_intercept,
            short_intercept,
        ],
        "Slope": [
            total_slope,
            long_slope,
            short_slope,
        ],
        "R_squared": [
            total_r2,
            long_r2,
            short_r2,
        ],
    })

    explain_df.to_csv(
        OUTPUT_DIR / "reversal_explains_lstm.csv",
        index=False,
    )

    return explain_df


def plot_lstm_vs_reversal(lstm_bt, reversal_bt):
    fig, ax = plt.subplots(figsize=(10, 5))

    lstm_cum = lstm_bt["net_ret"].cumsum() * 100
    reversal_cum = reversal_bt["net_ret"].cumsum() * 100

    ax.plot(lstm_bt["date"], lstm_cum, label="LSTM", linewidth=2)
    ax.plot(reversal_bt["date"], reversal_cum, label="Short-Term Reversal", linewidth=2)

    ax.axhline(0, linestyle="--", alpha=0.5)
    ax.set_title("LSTM vs Short-Term Reversal Strategy")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Net Return (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lstm_vs_reversal_extension.png", dpi=150)
    plt.close()


def main():
    df = prepare_extension_data()

    print("Extension data shape:", df.shape)
    print("Missing past_5d_ret:", df["past_5d_ret"].isna().sum())
    print("Missing fwd_ret_1d:", df["fwd_ret_1d"].isna().sum())

    lstm_bt, reversal_bt, summary = compare_lstm_vs_reversal(df)

    print("\nExtension performance:")
    print(summary.round(4))

    explain_df = explain_lstm_with_reversal(lstm_bt, reversal_bt)

    print("\nHow much reversal explains LSTM:")
    print(explain_df.round(4))

    plot_lstm_vs_reversal(lstm_bt, reversal_bt)

    print("\nDone. Outputs saved to predictions/.")


if __name__ == "__main__":
    main()