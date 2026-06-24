from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import time
import webbrowser

import pandas as pd

from soft_sensor_autoresearch.artifacts import RunArtifacts
from soft_sensor_autoresearch.data_contracts import infer_columns, load_dataset
from soft_sensor_autoresearch.env_check import build_environment_report
from soft_sensor_autoresearch.fde_bridge import (
    FdeWindowFeatureBuilder,
    add_fde_to_path,
    find_fde_root,
    load_tabpfn3_predictor_factory,
    load_tpt_predictor_factory,
)
from soft_sensor_autoresearch.holdout import build_holdout_plan
from soft_sensor_autoresearch.model_runner import run_candidate_holdout
from soft_sensor_autoresearch.report import ReportMetadata
from soft_sensor_autoresearch.resource_logging import ResourceMonitor
from soft_sensor_autoresearch.search import SearchConfig, run_search


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soft-sensor-autoresearch",
        description="Run local offline soft-sensor AutoResearch.",
    )
    parser.add_argument("data_file", type=Path)
    parser.add_argument("target_column")
    parser.add_argument("--time-budget-minutes", type=float, default=15.0)
    parser.add_argument("--num-train-samples", type=int, default=400)
    parser.add_argument("--top-features-n", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.30)
    parser.add_argument("--window-minutes", type=int, default=None)
    parser.add_argument("--forecast-horizons", type=_parse_forecast_horizons, default=(0,))
    parser.add_argument("--model-type", choices=("tabpfn3", "tpt"), default="tabpfn3")
    parser.add_argument("--tabpfn-device", default="auto")
    parser.add_argument("--tabpfn-fit-mode", default="fit_preprocessors")
    parser.add_argument("--tabpfn-n-estimators", type=int, default=1)
    parser.add_argument("--tpt-device", default="mps")
    parser.add_argument("--tpt-fit-mode", default="fit_preprocessors")
    parser.add_argument("--tpt-n-estimators", type=int, default=1)
    parser.add_argument("--fde-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resource-log-interval-seconds", type=float, default=2.0)
    parser.add_argument("--no-resource-log", action="store_false", dest="resource_log", default=True)
    parser.add_argument("--include-frequency-candidate", action="store_true")
    parser.add_argument("--open", action="store_true", dest="open_report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_path = run_autoresearch(
        data_file=args.data_file,
        target_column=args.target_column,
        time_budget_minutes=args.time_budget_minutes,
        num_train_samples=args.num_train_samples,
        top_features_n=args.top_features_n,
        validation_fraction=args.validation_fraction,
        window_minutes=args.window_minutes,
        forecast_horizons=args.forecast_horizons,
        model_type=args.model_type,
        tabpfn_device=args.tabpfn_device,
        tabpfn_fit_mode=args.tabpfn_fit_mode,
        tabpfn_n_estimators=args.tabpfn_n_estimators,
        tpt_device=args.tpt_device,
        tpt_fit_mode=args.tpt_fit_mode,
        tpt_n_estimators=args.tpt_n_estimators,
        fde_root=args.fde_root,
        output_dir=args.output_dir,
        resource_log=args.resource_log,
        resource_log_interval_seconds=args.resource_log_interval_seconds,
        include_frequency_candidate=args.include_frequency_candidate,
        open_report=args.open_report,
    )
    print(f"report.html: {report_path}")
    if args.resource_log:
        print(f"resource_usage.csv: {report_path.parent / 'resource_usage.csv'}")
    return 0


def run_autoresearch(
    data_file: Path,
    target_column: str,
    time_budget_minutes: float = 15.0,
    num_train_samples: int = 400,
    top_features_n: int = 32,
    validation_fraction: float = 0.30,
    window_minutes: int | None = None,
    forecast_horizons: tuple[int, ...] = (0,),
    model_type: str = "tabpfn3",
    tabpfn_device: str = "auto",
    tabpfn_fit_mode: str = "fit_preprocessors",
    tabpfn_n_estimators: int = 1,
    tpt_device: str = "mps",
    tpt_fit_mode: str = "fit_preprocessors",
    tpt_n_estimators: int = 1,
    fde_root: Path | None = None,
    output_dir: Path | None = None,
    resource_log: bool = True,
    resource_log_interval_seconds: float = 2.0,
    include_frequency_candidate: bool = False,
    open_report: bool = False,
    fde_builder=None,
    predictor_factory=None,
) -> Path:
    df = load_dataset(data_file)
    columns = infer_columns(df, target_column)
    resolved_window_minutes = window_minutes or _infer_sampling_minutes(df, columns.time_column)
    resolved_fde = find_fde_root(data_file.parent, explicit=fde_root)
    if resolved_fde is not None:
        add_fde_to_path(resolved_fde)

    if predictor_factory is None or fde_builder is None:
        report = build_environment_report(resolved_fde, model_type=model_type)
        if not report.ok:
            raise RuntimeError(report.to_text())
    if fde_builder is None:
        fde_builder = FdeWindowFeatureBuilder()
    if predictor_factory is None:
        if model_type == "tabpfn3":
            predictor_factory = load_tabpfn3_predictor_factory(
                device=tabpfn_device,
                fit_mode=tabpfn_fit_mode,
                n_estimators=tabpfn_n_estimators,
            )
        elif model_type == "tpt":
            predictor_factory = load_tpt_predictor_factory(
                device=tpt_device,
                fit_mode=tpt_fit_mode,
                n_estimators=tpt_n_estimators,
            )
        else:
            raise ValueError(f"unsupported model_type: {model_type}")

    holdouts = build_holdout_plan(
        df,
        columns.time_column,
        columns.target_column,
        validation_fraction=validation_fraction,
    )
    artifacts = RunArtifacts.create(output_dir or data_file.parent)
    monitor = (
        ResourceMonitor(
            artifacts.resource_usage_path,
            interval_seconds=resource_log_interval_seconds,
            start_epoch=time.time(),
        )
        if resource_log
        else nullcontext()
    )

    def runner(candidate, holdout):
        return run_candidate_holdout(
            df,
            columns,
            holdout,
            candidate,
            fde_builder,
            predictor_factory,
        )

    previous_resource_log = os.environ.get("SOFT_SENSOR_RESOURCE_LOG_PATH")
    previous_resource_start = os.environ.get("SOFT_SENSOR_RESOURCE_LOG_START_EPOCH")
    if resource_log:
        os.environ["SOFT_SENSOR_RESOURCE_LOG_PATH"] = str(artifacts.resource_usage_path)
        os.environ["SOFT_SENSOR_RESOURCE_LOG_START_EPOCH"] = str(monitor.start_epoch)
    try:
        with monitor:
            run_search(
                holdouts,
                SearchConfig(
                    time_budget_seconds=_time_budget_seconds(time_budget_minutes),
                    report_path=artifacts.report_path,
                    default_window_minutes=resolved_window_minutes,
                    num_train_samples=num_train_samples,
                    top_features_n=top_features_n,
                    include_frequency_candidate=include_frequency_candidate,
                    forecast_horizons=forecast_horizons,
                    report_metadata=_build_report_metadata(
                        data_file=data_file,
                        target_column=target_column,
                        model_type=model_type,
                        resolved_window_minutes=resolved_window_minutes,
                        num_train_samples=num_train_samples,
                        top_features_n=top_features_n,
                        validation_fraction=validation_fraction,
                        forecast_horizons=forecast_horizons,
                        include_frequency_candidate=include_frequency_candidate,
                        tabpfn_device=tabpfn_device,
                        tabpfn_fit_mode=tabpfn_fit_mode,
                        tabpfn_n_estimators=tabpfn_n_estimators,
                        tpt_device=tpt_device,
                        tpt_fit_mode=tpt_fit_mode,
                        tpt_n_estimators=tpt_n_estimators,
                        resolved_fde=resolved_fde,
                    ),
                ),
                runner,
            )
    finally:
        _restore_env("SOFT_SENSOR_RESOURCE_LOG_PATH", previous_resource_log)
        _restore_env("SOFT_SENSOR_RESOURCE_LOG_START_EPOCH", previous_resource_start)
    if open_report:
        webbrowser.open(artifacts.report_path.as_uri())
    return artifacts.report_path


def _infer_sampling_minutes(df, time_column: str) -> int:
    parsed = pd.to_datetime(df[time_column], errors="coerce").dropna().sort_values()
    if len(parsed) < 2:
        return 60
    deltas = parsed.diff().dropna().dt.total_seconds() / 60.0
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return 60
    return max(1, int(round(float(deltas.median()))))


def _restore_env(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _time_budget_seconds(time_budget_minutes: float) -> float:
    if time_budget_minutes <= 0:
        return 0
    return max(1.0, time_budget_minutes * 60.0)


def _parse_forecast_horizons(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("forecast horizons cannot be empty")
    horizons: set[int] = set()
    for part in parts:
        if ":" in part:
            bounds = part.split(":")
            if len(bounds) != 2:
                raise argparse.ArgumentTypeError(f"invalid horizon range: {part}")
            start, end = (int(bounds[0]), int(bounds[1]))
            if start > end:
                raise argparse.ArgumentTypeError(f"horizon range start must be <= end: {part}")
            horizons.update(range(start, end + 1))
        else:
            horizons.add(int(part))
    if any(horizon < 0 for horizon in horizons):
        raise argparse.ArgumentTypeError("forecast horizons must be nonnegative")
    return tuple(sorted(horizons))


def _build_report_metadata(
    *,
    data_file: Path,
    target_column: str,
    model_type: str,
    resolved_window_minutes: int,
    num_train_samples: int,
    top_features_n: int,
    validation_fraction: float,
    forecast_horizons: tuple[int, ...],
    include_frequency_candidate: bool,
    tabpfn_device: str,
    tabpfn_fit_mode: str,
    tabpfn_n_estimators: int,
    tpt_device: str,
    tpt_fit_mode: str,
    tpt_n_estimators: int,
    resolved_fde: Path | None,
) -> ReportMetadata:
    return ReportMetadata(
        target_column=target_column,
        data_file=str(data_file),
        model_type=model_type,
        default_window_minutes=resolved_window_minutes,
        num_train_samples=num_train_samples,
        top_features_n=top_features_n,
        validation_fraction=validation_fraction,
        forecast_horizons=forecast_horizons,
        include_frequency_candidate=include_frequency_candidate,
        tabpfn_device=tabpfn_device,
        tabpfn_fit_mode=tabpfn_fit_mode,
        tabpfn_n_estimators=tabpfn_n_estimators,
        tpt_device=tpt_device,
        tpt_fit_mode=tpt_fit_mode,
        tpt_n_estimators=tpt_n_estimators,
        fde_root=str(resolved_fde) if resolved_fde is not None else None,
    )
