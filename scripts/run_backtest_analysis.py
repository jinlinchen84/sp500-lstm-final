from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(".")
OUTPUT_DIR = ROOT / "predictions"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_backtest_input():
    lstm_daily = pd.read_parquet("predictions_daily/lstm_all_periods.parquet")
    bench_daily = pd.read_parquet("predictions_daily/benchmark_all_periods.parquet")

    lstm_daily["date"] = pd.to_datetime(lstm_daily["date"])
    bench_daily["date"] = pd.to_datetime(bench_daily["date"])

    lstm_daily["permno"] = lstm_daily["permno"].astype(str)
    bench_daily["permno"] = bench_daily["permno"].astype(str)

    df = lstm_daily.merge(
        bench_daily,
        on=["date", "permno", "period"],
        how="left"
    )

    all_trade = []
    for period in range(1, 24):
        path = f"output/trade_samples_period_{period:02d}.parquet"
        trade = pd.read_parquet(path)
        trade["date"] = pd.to_datetime(trade["date"])
        trade["period"] = period
        trade["permno"] = trade["permno"].astype(str)
        all_trade.append(trade[["date", "permno", "period", "fwd_ret_1d"]])

    trade_rets = pd.concat(all_trade, ignore_index=True)

    df = df.merge(
        trade_rets,
        on=["date", "permno", "period"],
        how="left"
    )

    return df


def run_backtest(df, prob_col="lstm_prob", k=10, cost_bps=5):
    results = []
    prev_long = set()
    prev_short = set()

    for date, group in df.groupby("date"):
        group = group.dropna(subset=[prob_col, "fwd_ret_1d"]).copy()

        if len(group) < 2 * k:
            continue

        group = group.sort_values(prob_col, ascending=False)

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
            "net_ret": net_ret
        })

        prev_long = long_stocks
        prev_short = short_stocks

    return pd.DataFrame(results)


def run_backtest_all_periods(df, prob_col="lstm_prob", k=10, cost_bps=5):
    out = []

    for period in sorted(df["period"].unique()):
        part = df[df["period"] == period].copy()
        bt = run_backtest(part, prob_col=prob_col, k=k, cost_bps=cost_bps)
        bt["period"] = period
        out.append(bt)

    return pd.concat(out, ignore_index=True)


def compute_metrics(bt):
    ret = bt["net_ret"]

    cumulative = (1 + ret).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1

    return {
        "Mean Net Return (%)": ret.mean() * 100,
        "Volatility (%)": ret.std() * 100,
        "Sharpe Ratio": ret.mean() / ret.std() * np.sqrt(252),
        "Max Drawdown (%)": drawdown.min() * 100,
        "Win Rate (%)": (ret > 0).mean() * 100,
        "Average Turnover": bt["turnover"].mean()
    }


def make_performance_table(df):
    models = {
        "LSTM": "lstm_prob",
        "RAF": "rf_prob",
        "LOG": "log_prob",
        "DNN": "dnn_prob",
    }

    rows = []

    for model, prob_col in models.items():
        bt = run_backtest_all_periods(df, prob_col=prob_col, k=10)
        m = compute_metrics(bt)
        m["Model"] = model
        m["Mean Gross Return (%)"] = bt["gross_ret"].mean() * 100
        rows.append(m)

        bt.to_csv(OUTPUT_DIR / f"backtest_{model.lower()}_k10.csv", index=False)

    table = pd.DataFrame(rows)
    table = table[
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

    table.to_csv(OUTPUT_DIR / "performance_summary_from_script.csv", index=False)
    return table


def top_flop_bucket_analysis(df, prob_col="lstm_prob"):
    clean = df.dropna(subset=[prob_col, "fwd_ret_1d"]).copy()

    clean["bucket"] = clean.groupby("date")[prob_col].transform(
        lambda x: pd.qcut(x.rank(method="first"), 10, labels=False)
    )

    bucket_ret = (
        clean.groupby("bucket")["fwd_ret_1d"]
        .mean()
        .reset_index()
    )

    bucket_ret["mean_return_pct"] = bucket_ret["fwd_ret_1d"] * 100
    bucket_ret.to_csv(OUTPUT_DIR / "lstm_bucket_return_analysis.csv", index=False)

    return bucket_ret

def compute_top_bottom_spread(bucket_ret):
    bottom_bucket = bucket_ret["bucket"].min()
    top_bucket = bucket_ret["bucket"].max()

    bottom_ret = bucket_ret.loc[
        bucket_ret["bucket"] == bottom_bucket, "fwd_ret_1d"
    ].iloc[0]

    top_ret = bucket_ret.loc[
        bucket_ret["bucket"] == top_bucket, "fwd_ret_1d"
    ].iloc[0]

    spread = top_ret - bottom_ret

    spread_df = pd.DataFrame({
        "bottom_bucket": [bottom_bucket],
        "top_bucket": [top_bucket],
        "bottom_mean_return": [bottom_ret],
        "top_mean_return": [top_ret],
        "top_bottom_spread": [spread],
        "bottom_mean_return_pct": [bottom_ret * 100],
        "top_mean_return_pct": [top_ret * 100],
        "top_bottom_spread_pct": [spread * 100],
    })

    spread_df.to_csv(OUTPUT_DIR / "lstm_top_bottom_spread.csv", index=False)

    return spread_df

def build_trade_history():
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

    hist = pd.concat(all_trade, ignore_index=True)
    hist = hist.sort_values(["period", "permno", "date"]).reset_index(drop=True)

    return hist


def add_past_return_features(hist):
    hist = hist.copy()
    hist = hist.sort_values(["period", "permno", "date"]).reset_index(drop=True)

    grouped = hist.groupby(["period", "permno"])

    hist["past_1d_ret"] = grouped["ret"].shift(0)

    hist["past_5d_ret"] = grouped["ret"].transform(
        lambda x: (1 + x).rolling(5).apply(np.prod, raw=True) - 1
    )

    hist["past_20d_ret"] = grouped["ret"].transform(
        lambda x: (1 + x).rolling(20).apply(np.prod, raw=True) - 1
    )

    hist["past_20d_vol"] = grouped["ret"].transform(
        lambda x: x.rolling(20).std()
    )

    feature_cols = [
        "date",
        "permno",
        "period",
        "past_1d_ret",
        "past_5d_ret",
        "past_20d_ret",
        "past_20d_vol",
    ]

    return hist[feature_cols]


def top_flop_pattern_analysis(df, k=10, prob_col="lstm_prob"):
    clean = df.dropna(subset=[prob_col, "fwd_ret_1d"]).copy()

    hist = build_trade_history()
    past_features = add_past_return_features(hist)

    clean = clean.merge(
        past_features,
        on=["date", "permno", "period"],
        how="left"
    )

    rows = []

    for date, group in clean.groupby("date"):
        group = group.dropna(subset=[prob_col, "fwd_ret_1d"]).copy()

        if len(group) < 2 * k:
            continue

        group = group.sort_values(prob_col, ascending=False)

        top = group.head(k).copy()
        flop = group.tail(k).copy()

        top["group"] = "Top 10"
        flop["group"] = "Flop 10"

        rows.append(pd.concat([top, flop], ignore_index=True))

    selected = pd.concat(rows, ignore_index=True)

    summary = (
        selected
        .groupby("group")
        [
            [
                "past_1d_ret",
                "past_5d_ret",
                "past_20d_ret",
                "past_20d_vol",
                "fwd_ret_1d",
            ]
        ]
        .mean()
        .reset_index()
    )

    pct_cols = [
        "past_1d_ret",
        "past_5d_ret",
        "past_20d_ret",
        "past_20d_vol",
        "fwd_ret_1d",
    ]

    for col in pct_cols:
        summary[col + "_pct"] = summary[col] * 100

    summary.to_csv(
        OUTPUT_DIR / "lstm_top_flop_pattern_table.csv",
        index=False
    )

    selected.to_csv(
        OUTPUT_DIR / "lstm_top_flop_selected_stocks.csv",
        index=False
    )

    return summary

def plot_bucket_returns(bucket_ret):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(bucket_ret["bucket"], bucket_ret["mean_return_pct"])
    ax.set_xlabel("Prediction Probability Decile")
    ax.set_ylabel("Next-Day Mean Return (%)")
    ax.set_title("LSTM Ranking Power: Return by Prediction Decile")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lstm_bucket_return_analysis.png", dpi=150)
    plt.close()


def main():
    df = load_backtest_input()

    print("Backtest input shape:", df.shape)
    print("Missing fwd_ret_1d:", df["fwd_ret_1d"].isna().sum())

    perf = make_performance_table(df)
    print(perf.round(4))

    bucket_ret = top_flop_bucket_analysis(df)
    print("\nBucket return analysis:")
    print(bucket_ret)

    spread_df = compute_top_bottom_spread(bucket_ret)
    print("\nTop-bottom spread:")
    print(spread_df)

    pattern_table = top_flop_pattern_analysis(df, k=10, prob_col="lstm_prob")
    print("\nTop/Flop pattern analysis:")
    print(pattern_table)

    plot_bucket_returns(bucket_ret)

    print("Done. Outputs saved to predictions/")


if __name__ == "__main__":
    main()