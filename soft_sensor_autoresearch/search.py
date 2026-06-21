from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import time

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


CandidateRunner = Callable[[CandidateConfig, HoldoutInterval], HoldoutRunResult]


def run_search(
    holdouts: HoldoutPlan,
    config: SearchConfig,
    runner: CandidateRunner,
) -> ReportState:
    deadline = time.monotonic() + config.time_budget_seconds
    reports: list[CandidateReport] = []

    baseline = CandidateConfig(
        candidate_id="baseline",
        max_derived_features=0,
        window_minutes=config.default_window_minutes,
        context_policy="uniform",
        num_train_samples=config.num_train_samples,
        include_frequency=False,
        random_state=config.random_state,
    )
    baseline_results = [runner(baseline, holdout) for holdout in holdouts.intervals]
    reports.append(_candidate_report(baseline, baseline_results, len(holdouts.intervals)))
    write_report(config.report_path, ReportState(_rank(reports)))

    worst = min(baseline_results, key=lambda result: result.r2).holdout_name
    quick_holdout = next(holdout for holdout in holdouts.intervals if holdout.name == worst)

    candidates = _initial_candidates(config)
    for candidate in candidates:
        if time.monotonic() >= deadline:
            break
        quick_result = runner(candidate, quick_holdout)
        results: list[HoldoutRunResult] = [quick_result]
        baseline_worst = next(result.r2 for result in baseline_results if result.holdout_name == worst)
        if quick_result.r2 >= baseline_worst:
            for holdout in holdouts.intervals:
                if holdout.name == quick_holdout.name or time.monotonic() >= deadline:
                    continue
                results.append(runner(candidate, holdout))
        reports.append(_candidate_report(candidate, results, len(holdouts.intervals)))
        write_report(config.report_path, ReportState(_rank(reports)))

    final_state = ReportState(_rank(reports))
    write_report(config.report_path, final_state)
    return final_state


def _initial_candidates(config: SearchConfig) -> list[CandidateConfig]:
    return [
        CandidateConfig("sisso_256", 256, config.default_window_minutes, "uniform", config.num_train_samples),
        CandidateConfig("window_short", 0, max(5, config.default_window_minutes // 2), "uniform", config.num_train_samples),
        CandidateConfig("window_long", 0, config.default_window_minutes * 2, "uniform", config.num_train_samples),
        CandidateConfig("coverage", 0, config.default_window_minutes, "coverage", config.num_train_samples),
        CandidateConfig("frequency", 0, config.default_window_minutes, "uniform", config.num_train_samples, True),
    ]


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
