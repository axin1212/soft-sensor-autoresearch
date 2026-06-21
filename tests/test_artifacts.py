from __future__ import annotations

import pandas as pd

from soft_sensor_autoresearch.artifacts import RunArtifacts


def test_run_artifacts_creates_expected_paths(tmp_path):
    artifacts = RunArtifacts.create(tmp_path, timestamp="20260101_000000")

    assert artifacts.run_dir.exists()
    assert artifacts.report_path.name == "report.html"
    assert artifacts.best_derived_features_path.name == "best_derived_features.parquet"
    assert artifacts.candidate_dir("c1").exists()


def test_save_best_derived_features(tmp_path):
    artifacts = RunArtifacts.create(tmp_path, timestamp="20260101_000000")

    artifacts.save_best_derived_features(pd.DataFrame({"a": [1, 2]}))

    assert artifacts.best_derived_features_path.exists()
