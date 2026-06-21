from __future__ import annotations

import argparse
from pathlib import Path
import webbrowser

from soft_sensor_autoresearch.artifacts import RunArtifacts
from soft_sensor_autoresearch.data_contracts import infer_columns, load_dataset
from soft_sensor_autoresearch.env_check import build_environment_report
from soft_sensor_autoresearch.fde_bridge import (
    FdeWindowFeatureBuilder,
    add_fde_to_path,
    find_fde_root,
    load_tabpfn3_predictor_factory,
)
from soft_sensor_autoresearch.holdout import build_holdout_plan
from soft_sensor_autoresearch.model_runner import run_candidate_holdout
from soft_sensor_autoresearch.search import SearchConfig, run_search


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
    report_path = run_autoresearch(
        data_file=args.data_file,
        target_column=args.target_column,
        time_budget_minutes=args.time_budget_minutes,
        fde_root=args.fde_root,
        output_dir=args.output_dir,
        open_report=args.open_report,
    )
    print(f"report.html: {report_path}")
    return 0


def run_autoresearch(
    data_file: Path,
    target_column: str,
    time_budget_minutes: float = 15.0,
    fde_root: Path | None = None,
    output_dir: Path | None = None,
    open_report: bool = False,
    fde_builder=None,
    predictor_factory=None,
) -> Path:
    df = load_dataset(data_file)
    columns = infer_columns(df, target_column)
    resolved_fde = find_fde_root(data_file.parent, explicit=fde_root)
    if resolved_fde is not None:
        add_fde_to_path(resolved_fde)

    if predictor_factory is None or fde_builder is None:
        report = build_environment_report(resolved_fde)
        if not report.ok:
            raise RuntimeError(report.to_text())
    if fde_builder is None:
        fde_builder = FdeWindowFeatureBuilder()
    if predictor_factory is None:
        predictor_factory = load_tabpfn3_predictor_factory()

    holdouts = build_holdout_plan(df, columns.time_column, columns.target_column)
    artifacts = RunArtifacts.create(output_dir or data_file.parent)

    def runner(candidate, holdout):
        return run_candidate_holdout(
            df,
            columns,
            holdout,
            candidate,
            fde_builder,
            predictor_factory,
        )

    run_search(
        holdouts,
        SearchConfig(
            time_budget_seconds=max(1.0, time_budget_minutes * 60.0),
            report_path=artifacts.report_path,
        ),
        runner,
    )
    if open_report:
        webbrowser.open(artifacts.report_path.as_uri())
    return artifacts.report_path
