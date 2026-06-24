from __future__ import annotations

import numpy as np

from soft_sensor_autoresearch.model_runner import HoldoutRunResult
from soft_sensor_autoresearch.report import CandidateReport, ReportMetadata, ReportState, write_report


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
        ],
        metadata=ReportMetadata(
            target_column="12PI-44026A",
            data_file="data.parquet",
            model_type="tabpfn3",
            default_window_minutes=10,
            num_train_samples=400,
            top_features_n=32,
            validation_fraction=0.30,
            forecast_horizons=(0, 10),
            include_frequency_candidate=False,
            tabpfn_device="auto",
            tabpfn_fit_mode="fit_preprocessors",
            tabpfn_n_estimators=1,
        ),
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
    assert "f0" in html
    assert "boom" in html
    assert "Run Parameters" in html
    assert "Target Tag" in html
    assert "12PI-44026A" in html
    assert "ICL Train Samples" in html
    assert "400" in html
    assert "Default Window Minutes" in html
    assert "10" in html
    assert "Forecast Horizons" in html
    assert "0, 10" in html


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


def test_write_report_uses_compact_subplot_titles(tmp_path):
    state = ReportState(
        candidates=[
            CandidateReport(
                candidate_id="candidate_name_that_is_long_enough_to_overlap_subplot_titles",
                score=0.98,
                status="complete",
                holdouts=[
                    HoldoutRunResult(
                        candidate_id="candidate_name_that_is_long_enough_to_overlap_subplot_titles",
                        holdout_name="holdout_name_that_is_also_long",
                        status="ok",
                        actual=np.array([1.0, 2.0, 3.0]),
                        predictions=np.array([1.1, 1.9, 3.1]),
                        r2=0.98,
                        rmse=0.1,
                        selected_features=["f0"],
                    )
                ],
            ),
            CandidateReport(
                candidate_id="second_long_candidate_name",
                score=0.90,
                status="complete",
                holdouts=[
                    HoldoutRunResult(
                        candidate_id="second_long_candidate_name",
                        holdout_name="second_long_holdout_name",
                        status="ok",
                        actual=np.array([1.0, 2.0, 3.0]),
                        predictions=np.array([1.0, 2.0, 3.0]),
                        r2=0.90,
                        rmse=0.0,
                        selected_features=["f1"],
                    )
                ],
            ),
        ]
    )
    path = tmp_path / "report.html"

    write_report(path, state)

    html = path.read_text()
    assert "candidate_name_that_is_long_enough_to_overlap_subplot_titles / holdout_name_that_is_also_long" not in html
    assert "#1 \\u00b7 holdout_name_that\\u2026\\u003cbr\\u003eR\\u00b2=0.980" in html
    assert '"font":{"size":11}' in html
    assert '"height":920' in html
