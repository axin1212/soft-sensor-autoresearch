from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable

import numpy as np
import pandas as pd

from soft_sensor_autoresearch.context_sampling import sample_context_indices
from soft_sensor_autoresearch.data_contracts import ColumnContract
from soft_sensor_autoresearch.feature_pool import (
    FdeFeatureBuilder,
    WindowFeatureRequest,
    build_window_feature_pool,
    select_top_features_xgboost,
)
from soft_sensor_autoresearch.holdout import HoldoutInterval
from soft_sensor_autoresearch.scoring import r2_score_np, rmse_np


@dataclass(frozen=True)
class CandidateConfig:
    candidate_id: str
    max_derived_features: int
    window_minutes: int
    context_policy: str
    num_train_samples: int = 400
    include_frequency: bool = False
    random_state: int = 42
    top_features_n: int = 32
    feature_mode: str = "trend"
    horizon_step: int = 0


@dataclass(frozen=True)
class HoldoutRunResult:
    candidate_id: str
    holdout_name: str
    status: str
    actual: np.ndarray
    predictions: np.ndarray
    r2: float
    rmse: float
    selected_features: list[str]
    horizon_step: int = 0
    error: str | None = None


PredictorFactory = Callable[[], object]


def run_candidate_holdout(
    df: pd.DataFrame,
    columns: ColumnContract,
    holdout: HoldoutInterval,
    config: CandidateConfig,
    fde_builder: FdeFeatureBuilder,
    predictor_factory: PredictorFactory,
) -> HoldoutRunResult:
    _progress(f"holdout_start candidate={config.candidate_id} holdout={holdout.name} horizon={config.horizon_step}")
    label_pairs = _aligned_label_pairs(df, columns, config.horizon_step)
    target_times = pd.to_datetime(label_pairs[_TARGET_TIME_COLUMN])
    anchor_times = pd.to_datetime(label_pairs[columns.time_column])
    context_pairs = label_pairs[
        ~(
            (target_times >= holdout.start_time)
            & (target_times <= holdout.end_time)
        )
        & ~(
            (anchor_times >= holdout.start_time)
            & (anchor_times <= holdout.end_time)
        )
    ]
    sampled_positions = sample_context_indices(
        pd.to_datetime(context_pairs[_TARGET_TIME_COLUMN]).reset_index(drop=True),
        holdout,
        policy=config.context_policy,
        n=config.num_train_samples,
        random_state=config.random_state,
    )
    train_labels = context_pairs.reset_index(drop=True).iloc[sampled_positions]
    holdout_labels = label_pairs[
        label_pairs[_TARGET_INDEX_COLUMN].isin(set(holdout.label_indices))
    ].reset_index(drop=True)
    if holdout_labels.empty:
        raise ValueError(f"no horizon-aligned labels for holdout={holdout.name} horizon={config.horizon_step}")

    _progress(f"features_train_start candidate={config.candidate_id} holdout={holdout.name} rows={len(train_labels)}")
    train_features = _build_features(df, columns, train_labels, config, fde_builder)
    _progress(f"features_train_end candidate={config.candidate_id} holdout={holdout.name} shape={train_features.shape}")
    _progress(f"features_test_start candidate={config.candidate_id} holdout={holdout.name} rows={len(holdout_labels)}")
    test_features = _build_features(df, columns, holdout_labels, config, fde_builder)
    _progress(f"features_test_end candidate={config.candidate_id} holdout={holdout.name} shape={test_features.shape}")
    y_train = pd.to_numeric(train_labels[_TARGET_VALUE_COLUMN], errors="coerce")
    y_test = pd.to_numeric(holdout_labels[_TARGET_VALUE_COLUMN], errors="coerce").to_numpy(dtype=float)

    _progress(f"feature_select_start candidate={config.candidate_id} holdout={holdout.name} k={config.top_features_n}")
    selected, _ = select_top_features_xgboost(
        train_features,
        y_train,
        k=config.top_features_n,
        random_state=config.random_state,
    )
    if not selected:
        selected = list(train_features.columns[: min(config.top_features_n, len(train_features.columns))])

    x_train_model, x_test_model = _standardize_features(
        train_features[selected].fillna(0.0),
        test_features[selected].fillna(0.0),
    )
    y_train_model, y_center, y_scale = _standardize_target(y_train.to_numpy(dtype=float))

    _progress(f"feature_select_end candidate={config.candidate_id} holdout={holdout.name} selected={len(selected)}")
    _progress(f"predictor_create_start candidate={config.candidate_id} holdout={holdout.name}")
    predictor = predictor_factory()
    _progress(f"predictor_create_end candidate={config.candidate_id} holdout={holdout.name}")
    _progress(f"predictor_fit_start candidate={config.candidate_id} holdout={holdout.name} train_shape={x_train_model.shape}")
    predictor.fit(x_train_model, y_train_model)
    _progress(f"predictor_fit_end candidate={config.candidate_id} holdout={holdout.name}")
    _progress(f"predictor_predict_start candidate={config.candidate_id} holdout={holdout.name} test_shape={x_test_model.shape}")
    raw_predictions = predictor.predict(x_test_model)
    _progress(f"predictor_predict_end candidate={config.candidate_id} holdout={holdout.name}")
    predictions = _prediction_array(raw_predictions) * y_scale + y_center

    return HoldoutRunResult(
        candidate_id=config.candidate_id,
        holdout_name=holdout.name,
        status="ok",
        actual=y_test,
        predictions=predictions,
        r2=r2_score_np(y_test, predictions),
        rmse=rmse_np(y_test, predictions),
        selected_features=selected,
        horizon_step=config.horizon_step,
    )


_TARGET_VALUE_COLUMN = "__target_value"
_TARGET_TIME_COLUMN = "__target_time"
_TARGET_INDEX_COLUMN = "__target_index"
_SOURCE_INDEX_COLUMN = "__source_index"


def _aligned_label_pairs(
    df: pd.DataFrame,
    columns: ColumnContract,
    horizon_step: int,
) -> pd.DataFrame:
    if horizon_step < 0:
        raise ValueError("horizon_step must be nonnegative")
    work = df.copy()
    work[columns.time_column] = pd.to_datetime(work[columns.time_column], errors="coerce")
    work[_SOURCE_INDEX_COLUMN] = work.index
    work = work[work[columns.time_column].notna()].sort_values(columns.time_column).reset_index(drop=True)
    target_values = pd.to_numeric(work[columns.target_column], errors="coerce")
    target_positions = np.flatnonzero(target_values.notna().to_numpy())
    target_positions = target_positions[target_positions >= horizon_step]
    if len(target_positions) == 0:
        return pd.DataFrame(columns=[*work.columns, _TARGET_VALUE_COLUMN, _TARGET_TIME_COLUMN, _TARGET_INDEX_COLUMN])

    anchor_positions = target_positions - horizon_step
    anchors = work.iloc[anchor_positions].copy().reset_index(drop=True)
    targets = work.iloc[target_positions].reset_index(drop=True)
    anchors[_TARGET_VALUE_COLUMN] = pd.to_numeric(targets[columns.target_column], errors="coerce").to_numpy(dtype=float)
    anchors[_TARGET_TIME_COLUMN] = pd.to_datetime(targets[columns.time_column]).to_numpy(dtype="datetime64[ns]")
    anchors[_TARGET_INDEX_COLUMN] = targets[_SOURCE_INDEX_COLUMN].map(int).to_numpy()
    return anchors


def _progress(message: str) -> None:
    if os.environ.get("SOFT_SENSOR_PROGRESS", "1") == "0":
        return
    print(f"[soft-sensor-autoresearch] {message}", flush=True)


def _build_features(
    df: pd.DataFrame,
    columns: ColumnContract,
    target_rows: pd.DataFrame,
    config: CandidateConfig,
    fde_builder: FdeFeatureBuilder,
) -> pd.DataFrame:
    if config.feature_mode == "identity":
        return target_rows[columns.feature_columns].reset_index(drop=True)
    if config.feature_mode != "trend":
        raise ValueError(f"unsupported feature_mode: {config.feature_mode}")
    request = WindowFeatureRequest(
        data=df,
        time_column=columns.time_column,
        feature_columns=columns.feature_columns,
        target_times=pd.to_datetime(target_rows[columns.time_column]).to_numpy(),
        window_minutes=config.window_minutes,
        include_frequency=config.include_frequency,
    )
    return build_window_feature_pool(request, fde_builder).features


def _prediction_array(raw_predictions: object) -> np.ndarray:
    if hasattr(raw_predictions, "mean"):
        mean = getattr(raw_predictions, "mean")
        if callable(mean):
            return np.asarray(raw_predictions, dtype=float)
        return np.asarray(mean, dtype=float)
    return np.asarray(raw_predictions, dtype=float)


def _standardize_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    train_arr = train.to_numpy(dtype=float)
    test_arr = test.to_numpy(dtype=float)
    center = np.nanmean(train_arr, axis=0)
    scale = np.nanstd(train_arr, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-9)] = 1.0
    center[~np.isfinite(center)] = 0.0
    return (
        ((train_arr - center) / scale).astype("float32"),
        ((test_arr - center) / scale).astype("float32"),
    )


def _standardize_target(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    center = float(np.nanmean(y))
    scale = float(np.nanstd(y))
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return ((y - center) / scale).astype("float32"), center, scale
