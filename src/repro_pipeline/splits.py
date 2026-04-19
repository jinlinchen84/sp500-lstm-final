from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class StudyPeriod:
    period_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    trade_start: pd.Timestamp
    trade_end: pd.Timestamp




def generate_study_periods(
    calendar: pd.DatetimeIndex,
    train_days: int = 750,
    trade_days: int = 250,
) -> list[StudyPeriod]:
    periods: list[StudyPeriod] = []
    start_idx = 0
    period_id = 1

    while True:
        train_start_idx = start_idx
        train_end_idx = train_start_idx + train_days - 1
        trade_start_idx = train_end_idx + 1
        trade_end_idx = trade_start_idx + trade_days - 1

        if trade_end_idx >= len(calendar):
            break

        periods.append(
            StudyPeriod(
                period_id=period_id,
                train_start=calendar[train_start_idx],
                train_end=calendar[train_end_idx],
                trade_start=calendar[trade_start_idx],
                trade_end=calendar[trade_end_idx],
            )
        )
        start_idx += trade_days
        period_id += 1

    return periods



def periods_to_frame(periods: list[StudyPeriod]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "period_id": p.period_id,
                "train_start": p.train_start,
                "train_end": p.train_end,
                "trade_start": p.trade_start,
                "trade_end": p.trade_end
            }
            for p in periods
        ]
    )
