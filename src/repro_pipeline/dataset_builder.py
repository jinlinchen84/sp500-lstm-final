from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AppConfig
from .constituents import (
    expand_membership_to_daily,
    get_constituent_universe_on_date,
    reconstruct_constituent_membership,
)
from .io_utils import ensure_dir, read_table, write_table
from .labels import add_forward_return_and_label
from .prices import (
    add_simple_returns,
    build_trading_calendar,
    clean_prices,
    standardize_with_train_only,
    winsorize_by_date,
)
from .splits import generate_study_periods, periods_to_frame
from .summary import summarize_period_samples, summaries_to_frame


@dataclass(slots=True)
class PipelineOutputs:
    master_dataset_path: Path
    study_periods_path: Path
    sample_count_summary_path: Path
    constituent_note_path: Path



def _add_sequence_flags(df: pd.DataFrame, id_col: str, sequence_length: int) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values([id_col, "date"]).reset_index(drop=True)
    out["obs_number"] = out.groupby(id_col, sort=False).cumcount() + 1
    out["has_full_sequence"] = out["obs_number"] >= sequence_length
    out["sequence_start_date"] = out.groupby(id_col, sort=False)["date"].shift(sequence_length - 1)
    return out



def _filter_period_panel(
    master: pd.DataFrame,
    universe: list[str],
    train_start: pd.Timestamp,
    trade_end: pd.Timestamp,
) -> pd.DataFrame:
    panel = master[
        master["permno"].isin(universe)
        & (master["date"] >= train_start)
        & (master["date"] <= trade_end)
    ].copy()
    return panel.sort_values(["permno", "date"]).reset_index(drop=True)



def _split_period_panel(
    panel: pd.DataFrame,
    train_end: pd.Timestamp,
    trade_start: pd.Timestamp,
    trade_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = panel[(panel["date"] <= train_end)].copy()
    trade_df = panel[(panel["date"] >= trade_start) & (panel["date"] <= trade_end)].copy()
    return train_df, trade_df



def _apply_train_only_standardization(period_panel: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    train_mask = period_panel["date"] <= train_end
    standardized, stats = standardize_with_train_only(
        period_panel,
        value_col="ret",
        train_mask=train_mask,
        out_col="ret_z",
    )
    standardized["train_mean_ret_1d"] = stats["mean"]
    standardized["train_std_ret_1d"] = stats["std"]
    return standardized



def build_master_dataset(config: AppConfig) -> pd.DataFrame:
    constituents_raw = read_table(config.paths.constituents_month_end).rename(columns={"month_end_date": "date"})

    prices_raw = read_table(config.paths.prices_daily)

    effective_constituents, note = reconstruct_constituent_membership(
        constituents_raw,
        date_col=config.columns.date,
        id_col=config.columns.id,
        in_index_col=config.columns.in_index,
    )

    prices = clean_prices(
        prices_raw,
        date_col=config.columns.date,
        id_col=config.columns.id,
        price_col=config.columns.price,
        return_col=config.columns.ret,
        drop_non_positive_prices=config.cleaning.drop_non_positive_prices,
    )


    if config.cleaning.winsorize_returns:
        prices = winsorize_by_date(
            prices,
            value_col=config.columns.ret,
            lower=config.cleaning.winsorize_lower,
            upper=config.cleaning.winsorize_upper,
            date_col=config.columns.date,
        )

    prices = add_forward_return_and_label(
        prices,
        date_col=config.columns.date,
        id_col=config.columns.id,
        return_col=config.columns.ret,
    )

    calendar = build_trading_calendar(
        prices,
        date_col=config.columns.date,
        start_date=config.pipeline.start_date,
        end_date=config.pipeline.end_date,
    )
    daily_membership = expand_membership_to_daily(
        effective_constituents,
        daily_calendar=calendar,
        id_col=config.columns.id,
        in_index_col=config.columns.in_index,
    )

    master = prices.merge(
        daily_membership,
        how="left",
        left_on=[config.columns.date, config.columns.id],
        right_on=["date", config.columns.id],
        suffixes=("", "_membership"),
    )
    master["is_constituent"] = master["is_constituent"].fillna(0).astype(int)
    master = master.sort_values([config.columns.id, config.columns.date]).reset_index(drop=True)

    output_dir = ensure_dir(config.paths.output_dir)
    write_table(master, output_dir / "master_dataset.parquet")
    (output_dir / "constituent_reconstruction_note.md").write_text(note.text, encoding="utf-8")
    return master



def build_period_datasets(config: AppConfig, master: pd.DataFrame) -> PipelineOutputs:
    output_dir = ensure_dir(config.paths.output_dir)
    calendar = build_trading_calendar(
        master,
        date_col=config.columns.date,
        start_date=config.pipeline.start_date,
        end_date=config.pipeline.end_date,
    )
    periods = generate_study_periods(
        calendar,
        train_days=config.pipeline.train_days,
        trade_days=config.pipeline.trade_days,
    )
    periods_df = periods_to_frame(periods)
    write_table(periods_df, output_dir / "study_periods.parquet")

    daily_membership = master.loc[master["is_constituent"] == 1, [config.columns.date, config.columns.id, "is_constituent"]].drop_duplicates()

    summaries: list[dict] = []
    all_period_frames: list[pd.DataFrame] = []

    for period in periods:
        universe = get_constituent_universe_on_date(daily_membership, period.train_end, config.columns.id)
        panel = _filter_period_panel(master, universe, period.train_start, period.trade_end)
        panel = _apply_train_only_standardization(panel, train_end=period.train_end)
        panel = _add_sequence_flags(panel, id_col=config.columns.id, sequence_length=config.pipeline.sequence_length)

        panel["period_id"] = period.period_id
        panel["is_train"] = panel["date"] <= period.train_end
        panel["is_trade"] = panel["date"].between(period.trade_start, period.trade_end)
        panel["available_for_feature_generation"] = panel["ret_z"].notna() & panel["has_full_sequence"]

        train_df, trade_df = _split_period_panel(panel, period.train_end, period.trade_start, period.trade_end)
        summaries.append(summarize_period_samples(train_df, trade_df, period.period_id))

        if config.pipeline.save_period_samples:
            write_table(train_df, output_dir / f"train_samples_period_{period.period_id:02d}.parquet")
            write_table(trade_df, output_dir / f"trade_samples_period_{period.period_id:02d}.parquet")

        all_period_frames.append(panel)

    summary_df = summaries_to_frame(summaries)
    write_table(summary_df, output_dir / "sample_count_summary.csv")

    full_period_panel = pd.concat(all_period_frames, ignore_index=True) if all_period_frames else pd.DataFrame()
    if not full_period_panel.empty:
        write_table(full_period_panel, output_dir / "master_dataset_with_periods.parquet")

    return PipelineOutputs(
        master_dataset_path=output_dir / "master_dataset.parquet",
        study_periods_path=output_dir / "study_periods.parquet",
        sample_count_summary_path=output_dir / "sample_count_summary.csv",
        constituent_note_path=output_dir / "constituent_reconstruction_note.md",
    )
