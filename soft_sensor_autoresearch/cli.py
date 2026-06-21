from __future__ import annotations

import argparse
from pathlib import Path

from soft_sensor_autoresearch.data_contracts import infer_columns, load_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soft-sensor-autoresearch",
        description="Run local offline soft-sensor AutoResearch.",
    )
    parser.add_argument("data_file", type=Path)
    parser.add_argument("target_column")
    parser.add_argument("--time-budget-minutes", type=float, default=15.0)
    parser.add_argument("--fde-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--open", action="store_true", dest="open_report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.data_file.exists():
        df = load_dataset(args.data_file)
        cols = infer_columns(df, args.target_column)
        print(
            "soft-sensor-autoresearch input ready: "
            f"time={cols.time_column} target={cols.target_column} "
            f"features={len(cols.feature_columns)}"
        )
        return 0
    print(
        "soft-sensor-autoresearch scaffold ready: "
        f"data_file={args.data_file} target={args.target_column}"
    )
    return 0
