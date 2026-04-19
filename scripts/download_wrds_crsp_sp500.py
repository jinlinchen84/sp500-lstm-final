from __future__ import annotations

import argparse
from pathlib import Path

from repro_pipeline.wrds_download import WrdsDownloadConfig, download_sp500_replication_inputs



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CRSP S&P 500 replication inputs from WRDS."
    )
    parser.add_argument("--wrds-username", required=True)
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--month-end-start", default="1989-12-01")
    parser.add_argument("--month-end-end", default="2015-09-30")
    parser.add_argument("--price-start", default="1990-01-01")
    parser.add_argument("--price-end", default="2015-10-31")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    outputs = download_sp500_replication_inputs(
        WrdsDownloadConfig(
            wrds_username=args.wrds_username,
            output_dir=Path(args.output_dir),
            month_end_start=args.month_end_start,
            month_end_end=args.month_end_end,
            price_start=args.price_start,
            price_end=args.price_end,
        )
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
