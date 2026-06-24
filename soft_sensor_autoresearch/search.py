from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
import math
import time
import os

from soft_sensor_autoresearch.holdout import HoldoutInterval, HoldoutPlan
from soft_sensor_autoresearch.model_runner import CandidateConfig, HoldoutRunResult
from soft_sensor_autoresearch.report import CandidateReport, ReportMetadata, ReportState, write_report
from soft_sensor_autoresearch.scoring import candidate_score


@dataclass(frozen=True)
class SearchConfig:
    time_budget_seconds: float
    report_path: Path
    default_window_minutes: int = 60
    top_features_n: int = 32
    num_train_samples: int = 400
    random_state: int = 42
    include_frequency_candidate: bool = False
    forecast_horizons: tuple[int, ...] = (0,)
    report_metadata: ReportMetadata | None = None


CandidateRunner = Callable[[CandidateConfig, HoldoutInterval], HoldoutRunResult]


def run_search(
    holdouts: HoldoutPlan,
    config: SearchConfig,
    runner: CandidateRunner,
) -> ReportState:
    _progress("search_start")
    deadline = (
        math.inf
        if config.time_budget_seconds <= 0
        else time.monotonic() + config.time_budget_seconds
    )
    reports: list[CandidateReport] = []

    horizons = _forecast_horizons(config)
    suffix_ids = len(horizons) > 1
    for horizon_step in horizons:
        if time.monotonic() >= deadline:
            break
        baseline = _baseline_candidate(config, horizon_step, suffix_ids)
        _progress(f"candidate_start id={baseline.candidate_id} holdouts={len(holdouts.intervals)}")
        baseline_results = [_safe_run(runner, baseline, holdout) for holdout in holdouts.intervals]
        reports.append(_candidate_report(baseline, baseline_results, len(holdouts.intervals)))
        write_report(config.report_path, ReportState(_rank(reports), metadata=config.report_metadata))
        _progress(f"candidate_end id={baseline.candidate_id}")

        worst = min(baseline_results, key=lambda result: result.r2).holdout_name
        quick_holdout = next(holdout for holdout in holdouts.intervals if holdout.name == worst)

        for candidate in _low_risk_candidates_for_horizon(config, horizon_step, suffix_ids):
            if time.monotonic() >= deadline:
                break
            candidate_report = _run_screened_candidate(runner, candidate, quick_holdout, holdouts, baseline_results, deadline)
            reports.append(candidate_report)
            write_report(config.report_path, ReportState(_rank(reports), metadata=config.report_metadata))
            _progress(f"candidate_end id={candidate.candidate_id}")

    final_state = ReportState(_rank(reports), metadata=config.report_metadata)
    write_report(config.report_path, final_state)
    _progress("search_end")
    return final_state


def _initial_candidates(config: SearchConfig) -> list[CandidateConfig]:
    horizons = _forecast_horizons(config)
    suffix_ids = len(horizons) > 1
    return [
        candidate
        for horizon_step in horizons
        for candidate in _low_risk_candidates_for_horizon(config, horizon_step, suffix_ids)
    ]


def _baseline_candidate(config: SearchConfig, horizon_step: int, suffix_id: bool) -> CandidateConfig:
    candidate = CandidateConfig(
        candidate_id="baseline",
        max_derived_features=0,
        window_minutes=config.default_window_minutes,
        context_policy="uniform",
        num_train_samples=config.num_train_samples,
        include_frequency=False,
        random_state=config.random_state,
        top_features_n=config.top_features_n,
        feature_mode="identity",
    )
    return _with_horizon(candidate, horizon_step, suffix_id)


def _low_risk_candidates_for_horizon(
    config: SearchConfig,
    horizon_step: int,
    suffix_id: bool,
) -> list[CandidateConfig]:
    candidates = _base_low_risk_candidates(config)
    return [_with_horizon(candidate, horizon_step, suffix_id) for candidate in candidates]


def _base_low_risk_candidates(config: SearchConfig) -> list[CandidateConfig]:
    candidates = []
    candidates.extend(
        [
            CandidateConfig("trend_default", 0, config.default_window_minutes, "uniform", config.num_train_samples, top_features_n=config.top_features_n),
            CandidateConfig("window_short", 0, max(5, config.default_window_minutes // 2), "uniform", config.num_train_samples, top_features_n=config.top_features_n),
            CandidateConfig("window_long", 0, config.default_window_minutes * 2, "uniform", config.num_train_samples, top_features_n=config.top_features_n),
            CandidateConfig("coverage", 0, config.default_window_minutes, "coverage", config.num_train_samples, top_features_n=config.top_features_n),
        ]
    )
    if config.include_frequency_candidate:
        candidates.append(
            CandidateConfig(
                "frequency",
                0,
                config.default_window_minutes,
                "uniform",
                config.num_train_samples,
                True,
                top_features_n=config.top_features_n,
            )
        )
    return candidates


def _with_horizon(candidate: CandidateConfig, horizon_step: int, suffix_id: bool) -> CandidateConfig:
    candidate_id = f"{candidate.candidate_id}_h+{horizon_step}" if suffix_id else candidate.candidate_id
    return replace(candidate, candidate_id=candidate_id, horizon_step=horizon_step)


def _forecast_horizons(config: SearchConfig) -> tuple[int, ...]:
    values = tuple(sorted(set(config.forecast_horizons)))
    if not values:
        return (0,)
    if any(value < 0 for value in values):
        raise ValueError("forecast_horizons must be nonnegative")
    return values


def _run_screened_candidate(
    runner: CandidateRunner,
    candidate: CandidateConfig,
    quick_holdout: HoldoutInterval,
    holdouts: HoldoutPlan,
    baseline_results: list[HoldoutRunResult],
    deadline: float,
) -> CandidateReport:
    _progress(f"candidate_start id={candidate.candidate_id} quick_holdout={quick_holdout.name}")
    quick_result = _safe_run(runner, candidate, quick_holdout)
    results: list[HoldoutRunResult] = [quick_result]
    baseline_worst = next(result.r2 for result in baseline_results if result.holdout_name == quick_holdout.name)
    if quick_result.r2 >= baseline_worst:
        for holdout in holdouts.intervals:
            if holdout.name == quick_holdout.name or time.monotonic() >= deadline:
                continue
            results.append(_safe_run(runner, candidate, holdout))
    return _candidate_report(candidate, results, len(holdouts.intervals))


def _worst_result_r2(results: list[HoldoutRunResult]) -> float:
    values = [result.r2 for result in results if math.isfinite(result.r2)]
    if not values:
        return float("-inf")
    return min(values)


def _safe_run(
    runner: CandidateRunner,
    candidate: CandidateConfig,
    holdout: HoldoutInterval,
) -> HoldoutRunResult:
    try:
        return runner(candidate, holdout)
    except Exception as exc:  # noqa: BLE001
        return HoldoutRunResult(
            candidate_id=candidate.candidate_id,
            holdout_name=holdout.name,
            status="error",
            actual=[],
            predictions=[],
            r2=float("nan"),
            rmse=float("nan"),
            selected_features=[],
            horizon_step=candidate.horizon_step,
            error=repr(exc),
        )


def _candidate_report(
    config: CandidateConfig,
    results: list[HoldoutRunResult],
    total_windows: int,
) -> CandidateReport:
    score = candidate_score([result.r2 for result in results], total_windows=total_windows)
    status = "complete" if len(results) == total_windows else "partial"
    return CandidateReport(candidate_id=config.candidate_id, score=score, status=status, holdouts=results, horizon_step=config.horizon_step)


def _rank(reports: list[CandidateReport]) -> list[CandidateReport]:
    return sorted(reports, key=lambda report: report.score, reverse=True)


def _progress(message: str) -> None:
    if os.environ.get("SOFT_SENSOR_PROGRESS", "1") == "0":
        return
    print(f"[soft-sensor-autoresearch] {message}", flush=True)
