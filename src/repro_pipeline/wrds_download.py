from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import wrds

from .io_utils import ensure_dir, write_table


@dataclass(slots=True)
class WrdsDownloadConfig:
    wrds_username: str
    output_dir: Path
    month_end_start: str = "1989-12-01"
    month_end_end: str = "2015-09-30"
    price_start: str = "1990-01-01"
    price_end: str = "2015-10-31"
    use_env_password: bool = True


MONTH_END_CONSTITUENTS_SQL = """
with month_ends as (
    select max(date) as month_end_date
    from crsp.dsf
    where date between %(month_end_start)s and %(month_end_end)s
    group by date_trunc('month', date)
),
membership as (
    select
        m.month_end_date,
        b.permno,
        b.start as sp500_start,
        b.ending as sp500_end
    from month_ends as m
    inner join crsp.msp500list as b
        on b.start <= m.month_end_date
       and m.month_end_date <= b.ending
)
select distinct
    membership.month_end_date,
    membership.permno,
    n.permco,
    n.ticker,
    n.comnam,
    membership.sp500_start,
    membership.sp500_end,
    1 as in_index
from membership
left join crsp.stocknames as n
    on membership.permno = n.permno
   and n.namedt <= membership.month_end_date
   and membership.month_end_date <= n.nameenddt
order by membership.month_end_date, membership.permno
"""


EVER_CONSTITUENTS_DAILY_PRICES_SQL = """
with ever_constituents as (
    select distinct permno
    from crsp.msp500list
    where ending >= %(month_end_start)s
      and start <= %(month_end_end)s
)
select distinct
    a.date,
    a.permno,
    n.permco,
    n.ticker,
    n.comnam,
    a.ret,
    a.retx,
    a.prc,
    a.vol,
    a.shrout,
    a.cfacpr,
    a.cfacshr
from crsp.dsf as a
inner join ever_constituents as e
    on a.permno = e.permno
left join crsp.stocknames as n
    on a.permno = n.permno
   and n.namedt <= a.date
   and a.date <= n.nameenddt
where a.date between %(price_start)s and %(price_end)s
order by a.date, a.permno
"""


def _get_connection(username: str, use_env_password: bool = True) -> wrds.Connection:
    if use_env_password:
        password = os.getenv("WRDS_PASSWORD")
        if password:
            return wrds.Connection(wrds_username=username, wrds_password=password)
    return wrds.Connection(wrds_username=username)



def download_sp500_replication_inputs(config: WrdsDownloadConfig) -> dict[str, Path]:
    output_dir = ensure_dir(config.output_dir)
    conn = _get_connection(config.wrds_username, use_env_password=config.use_env_password)

    params = {
        "month_end_start": config.month_end_start,
        "month_end_end": config.month_end_end,
        "price_start": config.price_start,
        "price_end": config.price_end,
    }

    month_end = conn.raw_sql(
        MONTH_END_CONSTITUENTS_SQL,
        params=params,
        date_cols=["month_end_date", "sp500_start", "sp500_end"],
    )
    daily_prices = conn.raw_sql(
        EVER_CONSTITUENTS_DAILY_PRICES_SQL,
        params=params,
        date_cols=["date"],
    )

    month_end_path = output_dir / "constituents_month_end.csv"
    daily_prices_path = output_dir / "prices_daily.csv"
    note_path = output_dir / "wrds_download_note.md"

    write_table(month_end, month_end_path)
    write_table(daily_prices, daily_prices_path)

    note_path.write_text(
        "\n".join(
            [
                "# WRDS extraction note",
                "",
                "- Source tables: `crsp.msp500list`, `crsp.dsf`, `crsp.stocknames`, `crsp.dsi`.",
                "- Month-end constituent snapshots are extracted first and should be shifted to the **subsequent month** in downstream reconstruction to match the paper.",
                "- Daily prices are downloaded for **all securities that were ever S&P 500 constituents** in the sample window, which avoids survivorship bias.",
                "- `permno` is kept as the stable security key; `ticker` is retained for readability only.",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "month_end_constituents": month_end_path,
        "daily_prices": daily_prices_path,
        "note": note_path,
    }
