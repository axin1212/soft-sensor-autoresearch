from __future__ import annotations

import numpy as np
import pandas as pd

from soft_sensor_autoresearch.derived_features import (
    DerivedFeatureConfig,
    generate_derived_features,
    infer_row_scales,
    select_derived_features,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=20, freq="min"),
            "target": np.linspace(0, 1, 20),
            "x": np.linspace(1, 20, 20),
            "y": np.linspace(20, 1, 20),
            "z": [0.0] * 20,
        }
    )


def test_infer_row_scales_is_deterministic():
    assert infer_row_scales(_frame(), "timestamp") == [1, 3, 6, 12]


def test_generate_derived_features_is_capped_and_named():
    derived, cols = generate_derived_features(
        _frame(),
        ["x", "y", "z"],
        "timestamp",
        DerivedFeatureConfig(candidate_feature_cap=8),
    )

    assert len(cols) == 8
    assert cols == sorted(cols)
    assert all(col.startswith("sisso__") for col in cols)
    assert set(cols).issubset(derived.columns)


def test_generate_derived_features_safe_division_has_no_inf():
    frame = _frame()
    derived, cols = generate_derived_features(frame, ["x", "z"], "timestamp")

    values = derived[cols].to_numpy(dtype=float)
    assert not np.isinf(values[~np.isnan(values)]).any()


def test_generate_derived_features_can_be_disabled():
    derived, cols = generate_derived_features(
        _frame(),
        ["x", "y"],
        "timestamp",
        DerivedFeatureConfig(candidate_feature_cap=0),
    )

    assert cols == []
    assert derived.empty


def test_select_derived_features_ranks_by_target_correlation():
    frame = _frame()
    frame["sisso__strong"] = frame["target"] * 10
    frame["sisso__weak"] = 1.0

    selected = select_derived_features(frame, "target", ["sisso__weak", "sisso__strong"], max_features=1)

    assert selected == ["sisso__strong"]
