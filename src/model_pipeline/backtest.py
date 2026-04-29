import numpy as np
import pandas as pd


def run_backtest(df, prob_col='lstm_prob', k=10, cost_bps=5):
    """
    Long-short portfolio backtest following Fischer & Krauss (2017).
    - Long top k stocks by prob_col
    - Short flop k stocks by prob_col
    - Equal weight, daily rebalance
    - Transaction cost: cost_bps per half-turn
    """
    results = []
    prev_long = set()
    prev_short = set()

    dates = sorted(df['date'].unique())

    for date in dates:
        group = df[df['date'] == date].copy()
        group = group.dropna(subset=[prob_col, 'fwd_ret_1d'])
        if len(group) < 2 * k:
            continue

        group = group.sort_values(prob_col, ascending=False).reset_index(drop=True)

        long_stocks = set(group.head(k)['permno'])
        short_stocks = set(group.tail(k)['permno'])

        long_ret = group.head(k)['fwd_ret_1d'].mean()
        short_ret = group.tail(k)['fwd_ret_1d'].mean()
        gross_ret = long_ret - short_ret

        long_turnover = len(long_stocks - prev_long) / k if prev_long else 1.0
        short_turnover = len(short_stocks - prev_short) / k if prev_short else 1.0
        turnover = (long_turnover + short_turnover) / 2

        cost = turnover * cost_bps / 10000
        net_ret = gross_ret - cost

        results.append({
            'date': date,
            'long_ret': long_ret,
            'short_ret': short_ret,
            'gross_ret': gross_ret,
            'turnover': turnover,
            'cost': cost,
            'net_ret': net_ret
        })

        prev_long = long_stocks
        prev_short = short_stocks

    return pd.DataFrame(results)


def run_backtest_all_periods(df, prob_col='lstm_prob', k=10, cost_bps=5):
    """Run backtest across all study periods."""
    all_results = []

    for period in sorted(df['period'].unique()):
        period_df = df[df['period'] == period].copy()
        period_df = period_df.dropna(subset=[prob_col])
        bt = run_backtest(period_df, prob_col=prob_col, k=k, cost_bps=cost_bps)
        bt['period'] = period
        all_results.append(bt)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


def compute_metrics(bt, name=None):
    """Compute performance metrics for a backtest result."""
    rets = pd.Series(bt['net_ret'].values)
    gross = bt['gross_ret'].values

    mean_gross = gross.mean() * 100
    mean_net = rets.mean() * 100
    std = rets.std() * 100
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252)
    cum_ret = (1 + rets).cumprod()
    max_dd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min() * 100
    win_rate = (rets > 0).mean() * 100
    annualized_ret = rets.mean() * 252 * 100
    calmar = annualized_ret / abs(max_dd)

    metrics = {
        'mean_gross_ret': mean_gross,
        'mean_net_ret': mean_net,
        'std': std,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'calmar': calmar
    }

    if name:
        print(f"{name}:")
        print(f"  Mean Gross Return: {mean_gross:.4f}%")
        print(f"  Mean Net Return:   {mean_net:.4f}%")
        print(f"  Std Dev:           {std:.4f}%")
        print(f"  Sharpe Ratio:      {sharpe:.4f}")
        print(f"  Max Drawdown:      {max_dd:.4f}%")
        print(f"  Win Rate:          {win_rate:.4f}%")
        print(f"  Calmar Ratio:      {calmar:.4f}")
        print()

    return metrics
