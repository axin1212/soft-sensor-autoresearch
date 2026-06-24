from __future__ import annotations

import numpy as np

from soft_sensor_autoresearch.holdout import HoldoutInterval, HoldoutPlan
from soft_sensor_autoresearch.model_runner import CandidateConfig, HoldoutRunResult
from soft_sensor_autoresearch.search import SearchConfig, _initial_candidates, run_search


def _holdouts() -> HoldoutPlan:
    return HoldoutPlan(
        intervals=[
            HoldoutInterval("h1", np.datetime64("2026-01-01"), np.datetime64("2026-01-02"), [1]),
            HoldoutInterval("h2", np.datetime64("2026-01-03"), np.datetime64("2026-01-04"), [2]),
            HoldoutInterval("h3", np.datetime64("2026-01-05"), np.datetime64("2026-01-06"), [3]),
        ],
        confidence="high",
    )


def test_run_search_baseline_all_holdouts_and_quick_screen(tmp_path):
    calls: list[tuple[str, str]] = []

    def fake_runner(config: CandidateConfig, holdout: HoldoutInterval) -> HoldoutRunResult:
        calls.append((config.candidate_id, holdout.name))
        score = {
            ("baseline", "h1"): 0.4,
            ("baseline", "h2"): -0.2,
            ("baseline", "h3"): 0.3,
        }.get((config.candidate_id, holdout.name), 0.5)
        return HoldoutRunResult(
            candidate_id=config.candidate_id,
            holdout_name=holdout.name,
            status="ok",
            actual=np.array([1.0, 2.0]),
            predictions=np.array([1.0, 2.0]),
            r2=score,
            rmse=0.0,
            selected_features=["f0"],
        )

    state = run_search(
        _holdouts(),
        SearchConfig(time_budget_seconds=1, report_path=tmp_path / "report.html"),
        fake_runner,
    )

    assert ("baseline", "h1") in calls
    assert ("baseline", "h2") in calls
    assert ("baseline", "h3") in calls
    assert calls[0][0] == "baseline"
    quick_screen_calls = [call for call in calls if call[0] != "baseline"]
    assert quick_screen_calls
    assert quick_screen_calls[0][1] == "h2"
    assert quick_screen_calls[0][0] == "trend_default"
    assert state.candidates[0].score >= state.candidates[-1].score
    assert (tmp_path / "report.html").exists()


def test_initial_candidates_exclude_sisso_derived_candidates():
    candidates = _initial_candidates(SearchConfig(time_budget_seconds=1, report_path="report.html"))
    ids = [candidate.candidate_id for candidate in candidates]

    assert all("sisso" not in candidate_id for candidate_id in ids)
    assert all(candidate.max_derived_features == 0 for candidate in candidates)


def test_initial_candidates_are_low_risk_only_by_default():
    candidates = _initial_candidates(SearchConfig(time_budget_seconds=1, report_path="report.html"))
    ids = [candidate.candidate_id for candidate in candidates]

    assert ids == ["trend_default", "window_short", "window_long", "coverage"]


def test_initial_candidates_expand_across_forecast_horizons():
    candidates = _initial_candidates(
        SearchConfig(time_budget_seconds=1, report_path="report.html", forecast_horizons=(0, 2))
    )
    ids = [candidate.candidate_id for candidate in candidates]

    assert "trend_default_h+0" in ids
    assert "trend_default_h+2" in ids
    assert {candidate.horizon_step for candidate in candidates} == {0, 2}


def test_frequency_candidate_is_opt_in():
    default_ids = [candidate.candidate_id for candidate in _initial_candidates(SearchConfig(1, "report.html"))]
    opt_in_ids = [
        candidate.candidate_id
        for candidate in _initial_candidates(SearchConfig(1, "report.html", include_frequency_candidate=True))
    ]

    assert "frequency" not in default_ids
    assert "frequency" in opt_in_ids


def test_zero_time_budget_runs_full_candidate_list(tmp_path):
    calls: list[tuple[str, str]] = []

    def fake_runner(config: CandidateConfig, holdout: HoldoutInterval) -> HoldoutRunResult:
        calls.append((config.candidate_id, holdout.name))
        return HoldoutRunResult(
            candidate_id=config.candidate_id,
            holdout_name=holdout.name,
            status="ok",
            actual=np.array([1.0, 2.0]),
            predictions=np.array([1.0, 2.0]),
            r2=0.5,
            rmse=0.0,
            selected_features=["f0"],
        )

    run_search(
        _holdouts(),
        SearchConfig(time_budget_seconds=0, report_path=tmp_path / "report.html"),
        fake_runner,
    )

    candidate_ids = {call[0] for call in calls}
    assert {candidate.candidate_id for candidate in _initial_candidates(SearchConfig(0, tmp_path / "report.html"))}.issubset(candidate_ids)


def test_run_search_does_not_run_sisso_candidates(tmp_path):
    calls: list[tuple[str, str]] = []

    def fake_runner(config: CandidateConfig, holdout: HoldoutInterval) -> HoldoutRunResult:
        calls.append((config.candidate_id, holdout.name))
        return HoldoutRunResult(
            candidate_id=config.candidate_id,
            holdout_name=holdout.name,
            status="ok",
            actual=np.array([1.0, 2.0]),
            predictions=np.array([2.0, 1.0]),
            r2=-2.0,
            rmse=1.0,
            selected_features=["f0"],
        )

    run_search(
        _holdouts(),
        SearchConfig(time_budget_seconds=0, report_path=tmp_path / "report.html"),
        fake_runner,
    )

    candidate_ids = {call[0] for call in calls}
    assert "trend_default" in candidate_ids
    assert all("sisso" not in candidate_id for candidate_id in candidate_ids)
