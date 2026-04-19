from __future__ import annotations

import numpy as np
import pandas as pd



def add_forward_return_and_label(
    df: pd.DataFrame,
    date_col: str,
    id_col: str,
    return_col: str,
) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values([id_col, date_col]).reset_index(drop=True)
    out["fwd_ret_1d"] = out.groupby(id_col, sort=False)[return_col].shift(-1)

    median_by_date = out.groupby(date_col)["fwd_ret_1d"].transform("median")
    out["cross_sectional_median_fwd_ret"] = median_by_date
    out["label_t1"] = np.where(
        out["fwd_ret_1d"].isna(),
        np.nan,
        np.where(out["fwd_ret_1d"] >= out["cross_sectional_median_fwd_ret"], 1, 0),
    )
    return out
