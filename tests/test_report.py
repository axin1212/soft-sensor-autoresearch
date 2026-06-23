from __future__ import annotations

import numpy as np

from soft_sensor_autoresearch.model_runner import HoldoutRunResult
from soft_sensor_autoresearch.report import CandidateReport, ReportState, write_report


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
    assert "45-degree" in html
    assert "#1" in html
    assert "Selected Features" in html
    assert "f0" in html
    assert "boom" in html


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
    assert "h1: error: TPT child process failed" in html
