from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DerivedFeatureConfig:
    candidate_feature_cap: int = 3000
    clip_lower_q: float = 0.01
    clip_upper_q: float = 0.99


def infer_row_scales(df: pd.DataFrame, time_column: str) -> list[int]:
    if time_column not in df.columns or len(df) <= 1:
        return [1]
    times = pd.to_datetime(df[time_column], errors="coerce").dropna().sort_values()
    if len(times) <= 1:
        return [1]
    max_window = max(1, min(12, len(df) - 1))
    return [scale for scale in [1, 3, 6, 12] if scale <= max_window]


def generate_derived_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    time_column: str,
    config: DerivedFeatureConfig | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    config = config or DerivedFeatureConfig()
    if config.candidate_feature_cap <= 0:
        return pd.DataFrame(index=df.index), []

    numeric = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    scales = infer_row_scales(df, time_column)
    generated: dict[str, pd.Series] = {}

    for col in sorted(numeric.columns):
        series = numeric[col]
        generated[f"sisso__lag_{col}__1"] = series.shift(1)
        for scale in scales:
            generated[f"sisso__diff_{col}__{scale}"] = series.diff(scale)
            generated[f"sisso__roll_mean_{col}__{scale}"] = series.rolling(scale, min_periods=1).mean()
            std_min_periods = min(2, scale)
            generated[f"sisso__roll_std_{col}__{scale}"] = series.rolling(
                scale, min_periods=std_min_periods
            ).std()
            generated[f"sisso__slope_{col}__{scale}"] = series.diff(scale) / max(scale, 1)
        generated[f"sisso__pow2_{col}"] = series * series
        generated[f"sisso__sqrt_abs_{col}"] = np.sqrt(series.abs())
        generated[f"sisso__log1p_abs_{col}"] = np.log1p(series.abs())

    for left, right in combinations(sorted(numeric.columns), 2):
        a = numeric[left]
        b = numeric[right]
        generated[f"sisso__add_{left}__{right}"] = a + b
        generated[f"sisso__sub_{left}__{right}"] = a - b
        generated[f"sisso__mul_{left}__{right}"] = a * b
        generated[f"sisso__div_{left}__{right}"] = safe_div(a, b)
        generated[f"sisso__abs_sub_{left}__{right}"] = (a - b).abs()
        generated[f"sisso__ratio_sum_{left}__{right}"] = safe_div(a, a + b)
        generated[f"sisso__contrast_{left}__{right}"] = safe_div(a - b, a + b)

    selected_names = sorted(generated)[: config.candidate_feature_cap]
    frame = pd.DataFrame({name: generated[name] for name in selected_names}, index=df.index)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = _clip_frame(frame, config.clip_lower_q, config.clip_upper_q)
    return frame, selected_names


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    denom = b.where(b.abs() > 1e-12)
    return (a / denom).replace([np.inf, -np.inf], np.nan)


def select_derived_features(
    df: pd.DataFrame,
    target_column: str,
    derived_columns: list[str],
    max_features: int,
) -> list[str]:
    if max_features <= 0:
        return []
    target = pd.to_numeric(df[target_column], errors="coerce")
    scores: list[tuple[float, str]] = []
    for col in derived_columns:
        values = pd.to_numeric(df[col], errors="coerce")
        valid = target.notna() & values.notna()
        if valid.sum() < 3 or values[valid].nunique() <= 1:
            continue
        corr = values[valid].corr(target[valid])
        if pd.notna(corr):
            scores.append((abs(float(corr)), col))
    scores.sort(key=lambda item: (-item[0], item[1]))
    return [col for _, col in scores[:max_features]]


def _clip_frame(frame: pd.DataFrame, lower_q: float, upper_q: float) -> pd.DataFrame:
    clipped = frame.copy()
    for col in clipped.columns:
        values = clipped[col]
        if values.notna().sum() < 2:
            continue
        lower = values.quantile(lower_q)
        upper = values.quantile(upper_q)
        clipped[col] = values.clip(lower, upper)
    return clipped
