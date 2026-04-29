"""
LSTM-Based Stock Market Prediction Pipeline
Reproduction of Fischer & Krauss (2017)

Usage:
    # First, download raw data (one-time):
    # export WRDS_PASSWORD='your_password'
    # PYTHONPATH=src python3 scripts/download_wrds_crsp_sp500.py --wrds-username your_username --output-dir data/raw

    PYTHONPATH=src python3 main.py --mode all          # Run full pipeline
    PYTHONPATH=src python3 main.py --mode data         # Run data pipeline only (Part A)
    PYTHONPATH=src python3 main.py --mode model        # Run model training only (Part B)
    PYTHONPATH=src python3 main.py --mode backtest     # Run backtest only (Part B)
    PYTHONPATH=src python3 main.py --mode extension    # Run extensions only (Part C)
"""

import argparse
import glob
import os
import subprocess
import torch
import pandas as pd
import numpy as np

from model_pipeline import (
    build_sequences, predict_all_trade_days,
    train_lstm, predict_proba_lstm,
    build_benchmark_features, train_logistic, train_random_forest,
    train_dnn, predict_proba_dnn,
    run_backtest_all_periods, compute_metrics
)

# ── Config ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = "output"
PRED_DIR = "predictions_daily"
K = 10
COST_BPS = 5


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Step 0: Data Pipeline (Part A) ───────────────────────────────────────────
def run_data_pipeline():
    print("Running data pipeline (Part A)...")
    subprocess.run(
        ["python3", "scripts/run_pipeline.py", "--config", "config/config.yaml"],
        env={**os.environ, "PYTHONPATH": "src"},
        check=True
    )
    print("Data pipeline done.")


# ── Step 1: Model Training and Prediction (Part B) ───────────────────────────
def run_model_pipeline(device):
    os.makedirs(PRED_DIR, exist_ok=True)

    train_files = sorted(glob.glob(f"{OUTPUT_DIR}/train_samples_period_*.parquet"))
    trade_files = sorted(glob.glob(f"{OUTPUT_DIR}/trade_samples_period_*.parquet"))

    if not train_files:
        raise FileNotFoundError("No train samples found. Run data pipeline first.")

    all_lstm = []
    all_benchmark = []

    for i, (tr_file, td_file) in enumerate(zip(train_files, trade_files)):
        period = i + 1
        print(f"\n=== Period {period}/23 ===")

        train_df = pd.read_parquet(tr_file)
        trade_df = pd.read_parquet(td_file)

        # LSTM
        X_tr, y_tr, _ = build_sequences(train_df)
        model = train_lstm(X_tr, y_tr, device)
        daily_preds = predict_all_trade_days(model, train_df, trade_df, device)
        daily_preds['period'] = period
        all_lstm.append(daily_preds)
        daily_preds.to_parquet(f"{PRED_DIR}/lstm_period_{period:02d}.parquet")
        print(f"LSTM: {daily_preds['date'].nunique()} unique dates")

        # Benchmarks
        X_btr, y_btr, _ = build_benchmark_features(train_df)
        X_btd, y_btd, meta_btd = build_benchmark_features(trade_df, train_df=train_df)

        meta_btd['log_prob'] = train_logistic(X_btr, y_btr, X_btd)
        meta_btd['rf_prob'] = train_random_forest(X_btr, y_btr, X_btd)
        dnn_model, dnn_scaler = train_dnn(X_btr, y_btr, device)
        meta_btd['dnn_prob'] = predict_proba_dnn(dnn_model, dnn_scaler, X_btd, device)
        meta_btd['period'] = period
        all_benchmark.append(meta_btd[['date','permno','period','log_prob','rf_prob','dnn_prob']])
        print(f"Benchmarks: {meta_btd['date'].nunique()} unique dates")

    lstm_daily = pd.concat(all_lstm, ignore_index=True)
    lstm_daily.to_parquet(f"{PRED_DIR}/lstm_all_periods.parquet")

    benchmark_daily = pd.concat(all_benchmark, ignore_index=True)
    benchmark_daily.to_parquet(f"{PRED_DIR}/benchmark_all_periods.parquet")

    print(f"\nModel pipeline done.")
    print(f"LSTM: {len(lstm_daily)} rows, Benchmarks: {len(benchmark_daily)} rows")


# ── Step 2: Backtest (Part B) ─────────────────────────────────────────────────
def run_backtest_pipeline():
    lstm_daily = pd.read_parquet(f"{PRED_DIR}/lstm_all_periods.parquet")
    lstm_daily['date'] = pd.to_datetime(lstm_daily['date'])

    benchmark_daily = pd.read_parquet(f"{PRED_DIR}/benchmark_all_periods.parquet")
    benchmark_daily['date'] = pd.to_datetime(benchmark_daily['date'])
    benchmark_daily['permno'] = benchmark_daily['permno'].astype(str)
    lstm_daily['permno'] = lstm_daily['permno'].astype(str)

    bt_final = lstm_daily.merge(benchmark_daily, on=['date','permno','period'], how='left')

    bt_final = bt_final.sort_values(['period','permno','date'])
    bt_final[['log_prob','rf_prob','dnn_prob']] = bt_final.groupby(
        ['period','permno'])[['log_prob','rf_prob','dnn_prob']].ffill()
    bt_final[['log_prob','rf_prob','dnn_prob']] = bt_final.groupby(
        ['period','permno'])[['log_prob','rf_prob','dnn_prob']].bfill()

    trade_files = sorted(glob.glob(f"{OUTPUT_DIR}/trade_samples_period_*.parquet"))
    all_trade = []
    for i, td_file in enumerate(trade_files):
        trade_df = pd.read_parquet(td_file)
        trade_df['date'] = pd.to_datetime(trade_df['date'])
        trade_df['period'] = i + 1
        trade_df['permno'] = trade_df['permno'].astype(str)
        all_trade.append(trade_df[['date','permno','period','fwd_ret_1d']])

    trade_rets = pd.concat(all_trade, ignore_index=True)
    bt_final = bt_final.merge(trade_rets, on=['date','permno','period'], how='left')

    print("\nRunning backtest...")
    lstm_bt = run_backtest_all_periods(bt_final, prob_col='lstm_prob', k=K)
    log_bt  = run_backtest_all_periods(bt_final, prob_col='log_prob',  k=K)
    rf_bt   = run_backtest_all_periods(bt_final, prob_col='rf_prob',   k=K)
    dnn_bt  = run_backtest_all_periods(bt_final, prob_col='dnn_prob',  k=K)

    print(f"\nPerformance Summary (k={K}, {COST_BPS} bps cost):")
    compute_metrics(lstm_bt, "LSTM")
    compute_metrics(log_bt,  "LOG")
    compute_metrics(rf_bt,   "RAF")
    compute_metrics(dnn_bt,  "DNN")

    os.makedirs("predictions", exist_ok=True)
    summary = pd.DataFrame({
        'Model': ['LSTM', 'LOG', 'RAF', 'DNN'],
        'Mean Net Return (%)': [bt['net_ret'].mean()*100 for bt in [lstm_bt, log_bt, rf_bt, dnn_bt]],
        'Sharpe': [(bt['net_ret'].mean()/bt['net_ret'].std())*np.sqrt(252)
                   for bt in [lstm_bt, log_bt, rf_bt, dnn_bt]],
    })
    summary.to_csv("predictions/performance_summary.csv", index=False)
    print("\nResults saved to predictions/performance_summary.csv")


# ── Step 3: Extensions (Part C) ───────────────────────────────────────────────
def run_extensions():
    print("\nRunning Extension 1: Reversal Strategy...")
    subprocess.run(
        ["python3", "scripts/run_reversal_extension.py"],
        env={**os.environ, "PYTHONPATH": "src"},
        check=True
    )
    print("Extension 1 done.")

    print("\nRunning Extension 2: Feature-Enhanced LSTM...")
    subprocess.run(
        ["python3", "scripts/run_feature_lstm_extension.py"],
        env={**os.environ, "PYTHONPATH": "src"},
        check=True
    )
    print("Extension 2 done.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Stock Prediction Pipeline")
    parser.add_argument('--mode',
                        choices=['all', 'data', 'model', 'backtest', 'extension'],
                        default='all',
                        help='Pipeline mode to run')
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    if args.mode in ['all', 'data']:
        run_data_pipeline()

    if args.mode in ['all', 'model']:
        run_model_pipeline(device)

    if args.mode in ['all', 'backtest']:
        run_backtest_pipeline()

    if args.mode in ['all', 'extension']:
        run_extensions()
