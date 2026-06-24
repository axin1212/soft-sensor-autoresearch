from __future__ import annotations

import numpy as np
import pandas as pd

from soft_sensor_autoresearch.data_contracts import ColumnContract
from soft_sensor_autoresearch.feature_pool import WindowFeatureRequest
from soft_sensor_autoresearch.holdout import HoldoutInterval
from soft_sensor_autoresearch.model_runner import CandidateConfig, run_candidate_holdout


class FakeFdeBuilder:
    def build_feature_matrix(self, request: WindowFeatureRequest, extraction: str) -> pd.DataFrame:
        times = pd.to_datetime(request.target_times)
        values = np.arange(len(times), dtype=float)
        return pd.DataFrame({"f0": values, "f1": values * 2})


class FailingFdeBuilder:
    def build_feature_matrix(self, request: WindowFeatureRequest, extraction: str) -> pd.DataFrame:
        raise AssertionError("identity mode must not call the FDE window builder")


class FakePredictor:
    def fit(self, x, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, x):
        return np.full(len(x), self.mean_)


def test_run_candidate_holdout_returns_metrics_and_predictions():
    rows = 30
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="min"),
            "target": [float(i) if i % 2 == 0 else None for i in range(rows)],
            "x": np.arange(rows, dtype=float),
        }
    )
    holdout = HoldoutInterval(
        name="h1",
        start_time=frame.loc[20, "timestamp"],
        end_time=frame.loc[24, "timestamp"],
        label_indices=[20, 22, 24],
    )
    config = CandidateConfig(
        candidate_id="c1",
        max_derived_features=0,
        window_minutes=30,
        context_policy="uniform",
        num_train_samples=5,
    )

    result = run_candidate_holdout(
        frame,
        ColumnContract("timestamp", "target", ["x"]),
        holdout,
        config,
        FakeFdeBuilder(),
        predictor_factory=FakePredictor,
    )

    assert result.status == "ok"
    assert result.predictions.shape == (3,)
    assert result.actual.shape == (3,)
    assert result.selected_features
    assert result.r2 <= 1.0


def test_identity_feature_mode_uses_existing_columns_without_window_builder():
    rows = 30
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="min"),
            "target": [float(i) if i % 2 == 0 else None for i in range(rows)],
            "x": np.arange(rows, dtype=float),
            "x_lag": np.arange(rows, dtype=float) * 2,
        }
    )
    holdout = HoldoutInterval(
        name="h1",
        start_time=frame.loc[20, "timestamp"],
        end_time=frame.loc[24, "timestamp"],
        label_indices=[20, 22, 24],
    )
    config = CandidateConfig(
        candidate_id="identity",
        max_derived_features=0,
        window_minutes=30,
        context_policy="uniform",
        num_train_samples=5,
        feature_mode="identity",
    )

    result = run_candidate_holdout(
        frame,
        ColumnContract("timestamp", "target", ["x", "x_lag"]),
        holdout,
        config,
        FailingFdeBuilder(),
        predictor_factory=FakePredictor,
    )

    assert result.status == "ok"
    assert result.selected_features


def test_horizon_step_uses_past_anchor_features_for_future_targets():
    rows = 20
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="min"),
            "target": np.arange(rows, dtype=float),
            "x": np.arange(rows, dtype=float) * 10.0,
        }
    )
    holdout = HoldoutInterval(
        name="h1",
        start_time=frame.loc[10, "timestamp"],
        end_time=frame.loc[12, "timestamp"],
        label_indices=[10, 11, 12],
    )
    config = CandidateConfig(
        candidate_id="identity_h+2",
        max_derived_features=0,
        window_minutes=30,
        context_policy="uniform",
        num_train_samples=5,
        feature_mode="identity",
        horizon_step=2,
    )

    result = run_candidate_holdout(
        frame,
        ColumnContract("timestamp", "target", ["x"]),
        holdout,
        config,
        FailingFdeBuilder(),
        predictor_factory=FakePredictor,
    )

    assert result.horizon_step == 2
    assert result.actual.tolist() == [10.0, 11.0, 12.0]
    assert result.predictions.shape == (3,)
