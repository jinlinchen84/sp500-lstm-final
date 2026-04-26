# Reproducing and Extending LSTM-Based Stock Market Prediction

This project reproduces and extends the LSTM-based cross-sectional stock prediction framework from Fischer and Krauss (2017). The original paper applies LSTM networks to S&P 500 constituent stocks and converts predicted probabilities into a daily long-short trading strategy.

Our project includes three main parts:

1. Data preparation and sample construction
2. Model reproduction
3. Backtesting, interpretation, and extensions

---

## 1. Project Overview

The goal of this project is to test whether LSTM networks can generate economically meaningful stock ranking signals.

The model predicts whether each S&P 500 constituent stock will outperform the cross-sectional median return on the next trading day. These prediction probabilities are then used to construct long-short portfolios:

- Long the top-ranked stocks
- Short the bottom-ranked stocks
- Daily rebalance
- Equal weighting
- 5 bps transaction cost per half-turn

---

## 2. Data

The project uses CRSP data from WRDS.

Main input data:

- Historical S&P 500 constituents
- Daily stock returns
- Daily adjusted price data

The sample is split into 23 rolling study periods:

- 750 trading days for training
- 250 trading days for out-of-sample trading

The LSTM input is a 240-day sequence of standardized daily returns.

Large intermediate data files are not uploaded to GitHub due to file size limits. They can be regenerated from the data pipeline.

---

## 3. Model Reproduction

We reproduce four models:

| Model | Description |
|---|---|
| LSTM | 240-day return sequence, LSTM with 25 hidden units |
| LOG | Logistic regression with L2 regularization |
| RAF | Random forest with 1000 trees and max depth 20 |
| DNN | Feedforward neural network benchmark |

The baseline LSTM follows the original paper’s architecture:

- Input: 1 feature × 240 timesteps
- LSTM hidden units: 25
- Dropout: 0.16
- Output: 2 neurons with softmax
- Loss: cross-entropy
- Optimizer: RMSprop

---

## 4. Backtesting Framework

The backtest converts model prediction probabilities into long-short portfolios.

Strategy design:

- Long top k stocks by predicted probability
- Short bottom k stocks by predicted probability
- Main setting: k = 10
- Daily rebalance
- Equal weight
- Transaction cost: 5 bps per half-turn

Main performance metrics:

- Mean gross return
- Mean net return
- Sharpe ratio
- Max drawdown
- Win rate
- Turnover

---

## 5. Main Backtest Results

For the k = 10 long-short portfolio after transaction costs:

| Model | Mean Net Return | Sharpe Ratio | Max Drawdown | Win Rate |
|---|---:|---:|---:|---:|
| LSTM | 0.2933% | 2.2395 | -43.76% | 56.66% |
| RAF | 0.2491% | 2.0872 | -45.92% | 55.11% |
| LOG | 0.1338% | 0.9003 | -80.63% | 51.67% |
| DNN | 0.0963% | 0.7945 | -89.05% | 52.23% |

The LSTM achieves the strongest risk-adjusted performance among the reproduced models.

---

## 6. Ranking Power Analysis

To understand whether the LSTM prediction probabilities contain economic information, we sort stocks into prediction deciles.

Key result:

- Bottom prediction decile mean next-day return: -0.0280%
- Top prediction decile mean next-day return: 0.1553%
- Top-bottom spread: 0.1834% per day

This shows that the LSTM has meaningful cross-sectional ranking power. Its value comes not only from classification accuracy, but also from ranking stocks by expected relative performance.

Related outputs:

- `prediction/lstm_bucket_return_analysis.png`
- `prediction/lstm_bucket_return_analysis.csv`
- `prediction/lstm_top_bottom_spread.csv`

---

## 7. Top/Flop Pattern Analysis

We analyze the stocks selected into the Top 10 and Flop 10 LSTM portfolios.

Key result:

| Group | Past 1D Return | Past 5D Return | Past 20D Return | Next 1D Return |
|---|---:|---:|---:|---:|
| Top 10 | -2.67% | -4.48% | -4.46% | 0.21% |
| Flop 10 | 2.71% | 6.68% | 8.03% | -0.12% |

This suggests that the LSTM tends to buy recent losers and short recent winners. The pattern is consistent with a short-term reversal effect.

Related outputs:

- `prediction/lstm_top_flop_pattern_table.csv`
- `prediction/lstm_top_flop_selected_stocks.csv`

---

## 8. Extension 1: Short-Term Reversal Strategy

Motivated by the top/flop pattern analysis, we construct a transparent short-term reversal strategy.

Strategy rule:

- Long the 10 stocks with the lowest past 5-day cumulative returns
- Short the 10 stocks with the highest past 5-day cumulative returns
- Daily rebalance
- Equal weight
- 5 bps transaction cost per half-turn

This extension tests whether the LSTM strategy’s profitability can be partially explained by a traditional short-term reversal effect.

Main finding:

The short-term reversal strategy generates positive cumulative returns and follows a broadly similar pattern to the LSTM strategy. However, the LSTM still outperforms the simple reversal rule, suggesting that it captures additional nonlinear temporal information beyond simple reversal.

Related outputs:

- `extension1/tables/reversal_extension_performance.csv`
- `extension1/tables/reversal_explains_lstm.csv`
- `extension1/figures/lstm_vs_reversal_extension.png`

---

## 9. Extension 2: Feature-Enhanced LSTM

The baseline LSTM uses only a 240-day sequence of standardized daily returns. The Feature-Enhanced LSTM adds two additional features:

- Standardized past 5-day cumulative return
- Standardized 20-day rolling volatility

The model architecture remains the same except that the input dimension increases from 1 to 3.

Full-period result:

| Model | Mean Net Return | Sharpe Ratio | Max Drawdown | Win Rate | Average Turnover |
|---|---:|---:|---:|---:|---:|
| Baseline LSTM | 0.2933% | 2.2395 | -43.76% | 56.66% | 0.6968 |
| Feature-LSTM | 0.3167% | 2.3564 | -44.94% | 57.15% | 0.6373 |

The Feature-Enhanced LSTM improves mean net return, Sharpe ratio, win rate, and turnover. However, it has slightly higher volatility and a slightly larger maximum drawdown. Overall, the result suggests that reversal and volatility information can complement the original return sequence.

Related outputs:

- `extension2/tables/feature_lstm_accuracy_by_period.csv`
- `extension2/tables/feature_lstm_vs_baseline_performance.csv`
- `extension2/figures/feature_lstm_vs_baseline.png`

---

## 10. How to Run

Create and activate an environment:

```bash
conda create -n sp500_lstm python=3.10 -y
conda activate sp500_lstm
pip install -r requirements.txt