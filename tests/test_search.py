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
    assert quick_screen_calls[0][0] == "identity_recent"
    assert state.candidates[0].score >= state.candidates[-1].score
    assert (tmp_path / "report.html").exists()


def test_initial_candidates_include_larger_context_sample_probes():
    candidates = _initial_candidates(SearchConfig(time_budget_seconds=1, report_path="report.html"))
    sample_counts = {candidate.num_train_samples for candidate in candidates}

    assert 400 in sample_counts
    assert 700 in sample_counts
    assert 900 in sample_counts


def test_low_risk_candidates_run_before_cse_candidates():
    candidates = _initial_candidates(SearchConfig(time_budget_seconds=1, report_path="report.html"))
    ids = [candidate.candidate_id for candidate in candidates]

    assert ids.index("identity_recent") < ids.index("trend_default")
    assert ids.index("identity_coverage") < ids.index("trend_default")
    assert ids.index("trend_default") < ids.index("sisso_256")
    assert ids.index("coverage") < ids.index("sisso_256")
    assert ids.index("sisso_256") < ids.index("sisso_256_samples_700")


def test_low_risk_context_candidates_keep_identity_features():
    candidates = _initial_candidates(SearchConfig(time_budget_seconds=1, report_path="report.html"))
    by_id = {candidate.candidate_id: candidate for candidate in candidates}

    assert by_id["identity_recent"].feature_mode == "identity"
    assert by_id["identity_recent"].context_policy == "recent"
    assert by_id["identity_coverage"].feature_mode == "identity"
    assert by_id["identity_coverage"].context_policy == "coverage"


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


def test_baseline_first_skips_cse_when_low_risk_scores_are_abnormally_bad(tmp_path):
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
        SearchConfig(time_budget_seconds=0, report_path=tmp_path / "report.html", cse_min_best_worst_r2=-0.5),
        fake_runner,
    )

    candidate_ids = {call[0] for call in calls}
    assert "trend_default" in candidate_ids
    assert "sisso_256" not in candidate_ids
