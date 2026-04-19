from __future__ import annotations

import pandas as pd



def summarize_period_samples(train_df: pd.DataFrame, trade_df: pd.DataFrame, period_id: int) -> dict:
    return {
        "period_id": period_id,
        "n_train_rows": int(len(train_df)),
        "n_trade_rows": int(len(trade_df)),
        "n_train_tickers": int(train_df["permno"].nunique()) if not train_df.empty else 0,
        "n_trade_tickers": int(trade_df["permno"].nunique()) if not trade_df.empty else 0,
        "n_train_sequences": int(train_df["has_full_sequence"].sum()) if "has_full_sequence" in train_df else 0,
        "n_trade_sequences": int(trade_df["has_full_sequence"].sum()) if "has_full_sequence" in trade_df else 0,
        "n_train_labels": int(train_df["label_t1"].notna().sum()) if "label_t1" in train_df else 0,
        "n_trade_labels": int(trade_df["label_t1"].notna().sum()) if "label_t1" in trade_df else 0,
    }



def summaries_to_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values("period_id").reset_index(drop=True)
