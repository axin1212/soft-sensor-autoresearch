from __future__ import annotations

import numpy as np

from soft_sensor_autoresearch.holdout import HoldoutInterval, HoldoutPlan
from soft_sensor_autoresearch.model_runner import CandidateConfig, HoldoutRunResult
from soft_sensor_autoresearch.search import SearchConfig, run_search


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
    quick_screen_calls = [call for call in calls if call[0] != "baseline"]
    assert quick_screen_calls
    assert quick_screen_calls[0][1] == "h2"
    assert state.candidates[0].score >= state.candidates[-1].score
    assert (tmp_path / "report.html").exists()
