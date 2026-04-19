from __future__ import annotations

import argparse

from .config import load_config
from .dataset_builder import build_master_dataset, build_period_datasets



def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper reproduction data pipeline.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    master = build_master_dataset(config)
    outputs = build_period_datasets(config, master)

    print("Pipeline finished.")
    print(f"master dataset: {outputs.master_dataset_path}")
    print(f"study periods: {outputs.study_periods_path}")
    print(f"sample summary: {outputs.sample_count_summary_path}")
    print(f"constituent note: {outputs.constituent_note_path}")


if __name__ == "__main__":
    main()
