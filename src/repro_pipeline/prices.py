from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import EPSILON



def clean_prices(
    prices: pd.DataFrame,
    date_col: str,
    id_col: str,
    price_col: str,
    return_col: str,
    drop_non_positive_prices: bool = True,
) -> pd.DataFrame:
    df = prices.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[[date_col, id_col, price_col,return_col]].copy()
    df[id_col] = df[id_col].astype(str)
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df[return_col] = pd.to_numeric(df[return_col], errors="coerce")
    df = df.dropna(subset=[date_col, id_col, price_col, return_col])

    if drop_non_positive_prices:
        df = df[df[price_col] > 0].copy()

    df = df.drop_duplicates([date_col, id_col], keep="last")
    df = df.sort_values([id_col, date_col]).reset_index(drop=True)
    return df



def add_simple_returns(
    prices: pd.DataFrame,
    date_col: str,
    id_col: str,
    price_col: str,
) -> pd.DataFrame:
    df = prices.copy()
    df["ret_1d"] = df.groupby(id_col, sort=False)[price_col].pct_change()
    return df



def winsorize_by_date(
    df: pd.DataFrame,
    value_col: str,
    lower: float,
    upper: float,
    date_col: str = "date",
) -> pd.DataFrame:
    out = df.copy()

    def _clip(group: pd.DataFrame) -> pd.DataFrame:
        lo = group[value_col].quantile(lower)
        hi = group[value_col].quantile(upper)
        group[value_col] = group[value_col].clip(lo, hi)
        return group

    return out.groupby(date_col, group_keys=False).apply(_clip)



def standardize_with_train_only(
    df: pd.DataFrame,
    value_col: str,
    train_mask: pd.Series,
    out_col: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    out = df.copy()
    train_values = out.loc[train_mask, value_col].dropna()
    mean_ = float(train_values.mean())
    std_ = float(train_values.std(ddof=0))
    std_ = max(std_, EPSILON)
    out[out_col] = (out[value_col] - mean_) / std_
    return out, {"mean": mean_, "std": std_}



def build_trading_calendar(
    prices: pd.DataFrame,
    date_col: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(prices[date_col].unique())))
    if start_date is not None:
        dates = dates[dates >= pd.Timestamp(start_date)]
    if end_date is not None:
        dates = dates[dates <= pd.Timestamp(end_date)]
    return dates
