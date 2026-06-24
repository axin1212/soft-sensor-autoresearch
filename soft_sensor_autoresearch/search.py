from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import math
import time
import os

from soft_sensor_autoresearch.holdout import HoldoutInterval, HoldoutPlan
from soft_sensor_autoresearch.model_runner import CandidateConfig, HoldoutRunResult
from soft_sensor_autoresearch.report import CandidateReport, ReportState, write_report
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
    search_profile: str = "baseline_first"
    cse_min_best_worst_r2: float = 0.0


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

    baseline = CandidateConfig(
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
    _progress(f"candidate_start id={baseline.candidate_id} holdouts={len(holdouts.intervals)}")
    baseline_results = [_safe_run(runner, baseline, holdout) for holdout in holdouts.intervals]
    reports.append(_candidate_report(baseline, baseline_results, len(holdouts.intervals)))
    write_report(config.report_path, ReportState(_rank(reports)))
    _progress(f"candidate_end id={baseline.candidate_id}")

    worst = min(baseline_results, key=lambda result: result.r2).holdout_name
    quick_holdout = next(holdout for holdout in holdouts.intervals if holdout.name == worst)

    low_risk_reports: list[CandidateReport] = []
    for candidate in _low_risk_candidates(config):
        if time.monotonic() >= deadline:
            break
        candidate_report = _run_screened_candidate(runner, candidate, quick_holdout, holdouts, baseline_results, deadline)
        reports.append(candidate_report)
        low_risk_reports.append(candidate_report)
        write_report(config.report_path, ReportState(_rank(reports)))
        _progress(f"candidate_end id={candidate.candidate_id}")

    best_low_risk_worst = max(
        (_worst_result_r2(report.holdouts) for report in [reports[0], *low_risk_reports]),
        default=float("nan"),
    )
    if config.search_profile == "baseline_first" and best_low_risk_worst < config.cse_min_best_worst_r2:
        _progress(
            "skip_cse reason=low_risk_worst_r2_below_threshold "
            f"best_low_risk_worst_r2={best_low_risk_worst:.4f} threshold={config.cse_min_best_worst_r2:.4f}"
        )
    else:
        for candidate in _cse_candidates(config):
            if time.monotonic() >= deadline:
                break
            candidate_report = _run_screened_candidate(runner, candidate, quick_holdout, holdouts, baseline_results, deadline)
            reports.append(candidate_report)
            write_report(config.report_path, ReportState(_rank(reports)))
            _progress(f"candidate_end id={candidate.candidate_id}")

    final_state = ReportState(_rank(reports))
    write_report(config.report_path, final_state)
    _progress("search_end")
    return final_state


def _initial_candidates(config: SearchConfig) -> list[CandidateConfig]:
    return [*_low_risk_candidates(config), *_cse_candidates(config)]


def _low_risk_candidates(config: SearchConfig) -> list[CandidateConfig]:
    candidates = []
    candidates.extend(
        [
            CandidateConfig(
                "identity_recent",
                0,
                config.default_window_minutes,
                "recent",
                config.num_train_samples,
                top_features_n=config.top_features_n,
                feature_mode="identity",
            ),
            CandidateConfig(
                "identity_coverage",
                0,
                config.default_window_minutes,
                "coverage",
                config.num_train_samples,
                top_features_n=config.top_features_n,
                feature_mode="identity",
            ),
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


def _cse_candidates(config: SearchConfig) -> list[CandidateConfig]:
    candidates = [
        CandidateConfig("sisso_256", 256, config.default_window_minutes, "uniform", config.num_train_samples, top_features_n=config.top_features_n),
    ]
    for sample_count in _larger_context_sample_counts(config.num_train_samples):
        candidates.append(
            CandidateConfig(
                f"sisso_256_samples_{sample_count}",
                256,
                config.default_window_minutes,
                "uniform",
                sample_count,
                top_features_n=config.top_features_n,
            )
        )
    return candidates


def _larger_context_sample_counts(base: int) -> list[int]:
    probes = [
        min(900, int(round(base * 1.75))),
        min(900, int(round(base * 2.25))),
    ]
    return sorted({value for value in probes if value > base})


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
            error=repr(exc),
        )


def _candidate_report(
    config: CandidateConfig,
    results: list[HoldoutRunResult],
    total_windows: int,
) -> CandidateReport:
    score = candidate_score([result.r2 for result in results], total_windows=total_windows)
    status = "complete" if len(results) == total_windows else "partial"
    return CandidateReport(candidate_id=config.candidate_id, score=score, status=status, holdouts=results)


def _rank(reports: list[CandidateReport]) -> list[CandidateReport]:
    return sorted(reports, key=lambda report: report.score, reverse=True)


def _progress(message: str) -> None:
    if os.environ.get("SOFT_SENSOR_PROGRESS", "1") == "0":
        return
    print(f"[soft-sensor-autoresearch] {message}", flush=True)
