from __future__ import annotations

import argparse
from pathlib import Path


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
    print(
        "soft-sensor-autoresearch scaffold ready: "
        f"data_file={args.data_file} target={args.target_column}"
    )
    return 0
