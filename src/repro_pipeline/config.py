from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class PathsConfig:
    constituents_month_end: Path
    prices_daily: Path
    output_dir: Path


@dataclass(slots=True)
class ColumnsConfig:
    date: str = "date"
    id: str = "permno"
    in_index: str = "in_index"
    price: str = "prc"
    ret: str = "ret"



@dataclass(slots=True)
class PipelineConfig:
    train_days: int = 750
    trade_days: int = 250
    sequence_length: int = 240
    min_history_days: int = 240
    start_date: str = "1990-01-01"
    end_date: str = "2015-10-31"
    save_period_samples: bool = True


@dataclass(slots=True)
class CleaningConfig:
    drop_non_positive_prices: bool = False
    winsorize_returns: bool = False
    winsorize_lower: float = 0.001
    winsorize_upper: float = 0.999


@dataclass(slots=True)
class AppConfig:
    paths: PathsConfig
    columns: ColumnsConfig
    pipeline: PipelineConfig
    cleaning: CleaningConfig



def _to_paths_config(raw: dict[str, Any]) -> PathsConfig:
    return PathsConfig(
        constituents_month_end=Path(raw["constituents_month_end"]),
        prices_daily=Path(raw["prices_daily"]),
        output_dir=Path(raw["output_dir"]),
    )



def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return AppConfig(
        paths=_to_paths_config(raw["paths"]),
        columns=ColumnsConfig(**raw.get("columns", {})),
        pipeline=PipelineConfig(**raw.get("pipeline", {})),
        cleaning=CleaningConfig(**raw.get("cleaning", {})),
    )
