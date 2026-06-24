from __future__ import annotations

import numpy as np

from soft_sensor_autoresearch.model_runner import HoldoutRunResult
from soft_sensor_autoresearch.report import CandidateReport, ReportState, _subplot_titles, write_report


def test_write_report_contains_core_elements(tmp_path):
    state = ReportState(
        candidates=[
            CandidateReport(
                candidate_id="c1",
                score=0.42,
                status="complete",
                holdouts=[
                    HoldoutRunResult(
                        candidate_id="c1",
                        holdout_name="h1",
                        status="ok",
                        actual=np.array([1.0, 2.0]),
                        predictions=np.array([1.1, 1.9]),
                        r2=0.98,
                        rmse=0.1,
                        selected_features=["f0"],
                    )
                ],
            ),
            CandidateReport(candidate_id="bad", score=-999.0, status="failed", holdouts=[], error="boom"),
        ]
    )
    path = tmp_path / "report.html"

    write_report(path, state)

    html = path.read_text()
    assert "plotly" in html.lower()
    assert "R²" in html
    assert "RMSE=0.1000" in html
    assert "MAE=0.1000" in html
    assert "y_std=" in html
    assert "RMSE/std=" in html
    assert "45-degree" in html
    assert "#1" in html
    assert "Selected Features" in html
    assert "<details class='feature-details'>" in html
    assert "1 feature entry across 1 holdout" in html
    assert "<li>f0</li>" in html
    assert "f0" in html
    assert "boom" in html
    assert "c1 / h1 n=2 R²=0.980" not in html


def test_write_report_surfaces_holdout_errors(tmp_path):
    state = ReportState(
        candidates=[
            CandidateReport(
                candidate_id="c1",
                score=float("-inf"),
                status="complete",
                holdouts=[
                    HoldoutRunResult(
                        candidate_id="c1",
                        holdout_name="h1",
                        status="error",
                        actual=np.array([]),
                        predictions=np.array([]),
                        r2=float("nan"),
                        rmse=float("nan"),
                        selected_features=[],
                        error="TPT child process failed",
                    )
                ],
            )
        ]
    )
    path = tmp_path / "report.html"

    write_report(path, state)

    html = path.read_text()
    assert "<strong>h1</strong>: error: TPT child process failed" in html


def test_write_report_keeps_selected_features_collapsed_and_grouped(tmp_path):
    features = [f"feature_{index}" for index in range(35)]
    state = ReportState(
        candidates=[
            CandidateReport(
                candidate_id="very_long_candidate_name_that_needs_shortening",
                score=0.1,
                status="complete",
                holdouts=[
                    HoldoutRunResult(
                        candidate_id="c1",
                        holdout_name="very_long_holdout_name_that_needs_shortening",
                        status="ok",
                        actual=np.array([1.0, 2.0, 3.0]),
                        predictions=np.array([1.1, 1.9, 3.1]),
                        r2=0.5,
                        rmse=0.1,
                        selected_features=features,
                    )
                ],
            )
        ]
    )
    path = tmp_path / "report.html"

    write_report(path, state)

    html = path.read_text()
    assert "35 feature entries across 1 holdout" in html
    assert "<div class='feature-holdout'>very_long_holdout_name_that_needs_shortening (35)</div>" in html
    assert "<li>feature_34</li>" in html
    assert "very_long_candidate_name_that_needs_shortening / very_long_holdout_name_that_needs_shortening" not in html


def test_subplot_titles_are_short_and_aligned_to_grid():
    candidates = [
        CandidateReport(
            candidate_id="very_long_candidate_name_that_needs_shortening",
            score=0.1,
            status="complete",
            holdouts=[
                HoldoutRunResult(
                    candidate_id="c1",
                    holdout_name="very_long_holdout_name_that_needs_shortening",
                    status="ok",
                    actual=np.array([1.0]),
                    predictions=np.array([1.0]),
                    r2=0.5,
                    rmse=0.0,
                    selected_features=[],
                )
            ],
        ),
        CandidateReport(
            candidate_id="c2",
            score=0.0,
            status="partial",
            holdouts=[
                HoldoutRunResult(
                    candidate_id="c2",
                    holdout_name="h1",
                    status="ok",
                    actual=np.array([1.0]),
                    predictions=np.array([1.0]),
                    r2=0.1,
                    rmse=0.0,
                    selected_features=[],
                ),
                HoldoutRunResult(
                    candidate_id="c2",
                    holdout_name="h2",
                    status="ok",
                    actual=np.array([1.0]),
                    predictions=np.array([1.0]),
                    r2=0.2,
                    rmse=0.0,
                    selected_features=[],
                ),
            ],
        ),
    ]

    titles = _subplot_titles(candidates, holdout_count=2)

    assert titles == [
        "very_long_candida… / very_long_holdout… / R²=0.50",
        "",
        "c2 / h1 / R²=0.10",
        "c2 / h2 / R²=0.20",
    ]
