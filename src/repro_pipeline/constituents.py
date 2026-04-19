from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.tseries.offsets import MonthBegin


@dataclass(slots=True)
class ConstituentNote:
    text: str



def reconstruct_constituent_membership(
    month_end_df: pd.DataFrame,
    date_col: str,
    id_col: str,
    in_index_col: str,
) -> tuple[pd.DataFrame, ConstituentNote]:
    """
    Convert month-end constituent snapshots into an effective-from table.

    Paper logic: month-end constituent list is used to indicate membership in the
    subsequent month.
    """
    df = month_end_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[[date_col, id_col, in_index_col]].copy()
    df[in_index_col] = df[in_index_col].astype(int)
    df[id_col] = df[id_col].astype(str)
    df = df.sort_values([id_col, date_col]).reset_index(drop=True)

    df["effective_start"] = (df[date_col] + MonthBegin(1)).dt.normalize()
    df["effective_end"] = (df[date_col] + MonthBegin(2)).dt.normalize() - pd.Timedelta(days=1)

    note = ConstituentNote(
        text=(
            "# Constituent reconstruction note\n\n"
            "- Input is month-end S&P 500 membership snapshots.\n"
            "- Each month-end snapshot is shifted forward to represent membership in the subsequent month.\n"
            "- Output is an effective-date table with `effective_start` and `effective_end`.\n"
            "- This produces a monthly binary membership matrix indicating whether a stock belongs to the index in month t+1.\n"
            "- This monthly membership is later used to define the stock universe at the end of each training window for rolling study periods.\n"
        )
    )
    return df, note



def expand_membership_to_daily(
    effective_df: pd.DataFrame,
    daily_calendar: pd.DatetimeIndex,
    id_col: str,
    in_index_col: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, row in effective_df.iterrows():
        if int(row[in_index_col]) != 1:
            continue
        mask = (daily_calendar >= row["effective_start"]) & (daily_calendar <= row["effective_end"])
        if not mask.any():
            continue
        piece = pd.DataFrame(
            {
                "date": daily_calendar[mask],
                id_col: row[id_col],
                "is_constituent": 1,
            }
        )
        rows.append(piece)
    if not rows:
        return pd.DataFrame(columns=["date", id_col, "is_constituent"])
    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(["date", id_col]).sort_values(["date", id_col])
    return out.reset_index(drop=True)



def get_constituent_universe_on_date(
    daily_membership: pd.DataFrame,
    as_of_date: pd.Timestamp,
    id_col: str,
) -> list[str]:
    tickers = (
        daily_membership.loc[daily_membership["date"] == pd.Timestamp(as_of_date), id_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    return sorted(tickers)
