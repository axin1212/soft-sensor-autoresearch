from __future__ import annotations

from dataclasses import dataclass
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
    labels = df[df[columns.target_column].notna()].copy()
    label_times = pd.to_datetime(labels[columns.time_column])
    context_labels = labels[
        ~(
            (label_times >= holdout.start_time)
            & (label_times <= holdout.end_time)
        )
    ]
    sampled_positions = sample_context_indices(
        pd.to_datetime(context_labels[columns.time_column]).reset_index(drop=True),
        holdout,
        policy=config.context_policy,
        n=config.num_train_samples,
        random_state=config.random_state,
    )
    train_labels = context_labels.reset_index(drop=True).iloc[sampled_positions]
    holdout_labels = df.loc[holdout.label_indices]

    train_features = _build_features(df, columns, train_labels, config, fde_builder)
    test_features = _build_features(df, columns, holdout_labels, config, fde_builder)
    y_train = pd.to_numeric(train_labels[columns.target_column], errors="coerce")
    y_test = pd.to_numeric(holdout_labels[columns.target_column], errors="coerce").to_numpy(dtype=float)

    selected, _ = select_top_features_xgboost(train_features, y_train, k=32, random_state=config.random_state)
    if not selected:
        selected = list(train_features.columns[: min(32, len(train_features.columns))])

    predictor = predictor_factory()
    predictor.fit(train_features[selected].fillna(0.0), y_train.to_numpy(dtype=float))
    predictions = np.asarray(predictor.predict(test_features[selected].fillna(0.0)), dtype=float)

    return HoldoutRunResult(
        candidate_id=config.candidate_id,
        holdout_name=holdout.name,
        status="ok",
        actual=y_test,
        predictions=predictions,
        r2=r2_score_np(y_test, predictions),
        rmse=rmse_np(y_test, predictions),
        selected_features=selected,
    )


def _build_features(
    df: pd.DataFrame,
    columns: ColumnContract,
    target_rows: pd.DataFrame,
    config: CandidateConfig,
    fde_builder: FdeFeatureBuilder,
) -> pd.DataFrame:
    request = WindowFeatureRequest(
        data=df,
        time_column=columns.time_column,
        feature_columns=columns.feature_columns,
        target_times=pd.to_datetime(target_rows[columns.time_column]).to_numpy(),
        window_minutes=config.window_minutes,
        include_frequency=config.include_frequency,
    )
    return build_window_feature_pool(request, fde_builder).features
