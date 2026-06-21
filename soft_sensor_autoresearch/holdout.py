from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class HoldoutInterval:
    name: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    label_indices: list[int]


@dataclass(frozen=True)
class HoldoutPlan:
    intervals: list[HoldoutInterval]
    confidence: str


def build_holdout_plan(df: pd.DataFrame, time_column: str, target_column: str) -> HoldoutPlan:
    if time_column not in df.columns:
        raise ValueError(f"time column not found: {time_column}")
    if target_column not in df.columns:
        raise ValueError(f"target column not found: {target_column}")

    work = df[[time_column, target_column]].copy()
    work[time_column] = pd.to_datetime(work[time_column], errors="coerce")
    labels = work[work[target_column].notna() & work[time_column].notna()].sort_values(time_column)
    label_count = len(labels)
    if label_count < 8:
        raise ValueError("need at least 8 non-null target labels to build holdout intervals")

    interval_count, confidence = _interval_count(label_count)
    centers = {
        3: [0.25, 0.50, 0.75],
        2: [1 / 3, 2 / 3],
        1: [0.50],
    }[interval_count]
    span = max(1, min(5, label_count // (interval_count * 4)))
    original_indices = list(labels.index)
    label_times = list(labels[time_column])

    used_positions: set[int] = set()
    intervals: list[HoldoutInterval] = []
    for i, center in enumerate(centers, start=1):
        center_pos = min(label_count - 1, max(0, round((label_count - 1) * center)))
        positions = _nearest_unused_span(center_pos, span, label_count, used_positions)
        used_positions.update(positions)
        idxs = [int(original_indices[pos]) for pos in positions]
        times = [pd.Timestamp(label_times[pos]) for pos in positions]
        intervals.append(
            HoldoutInterval(
                name=f"holdout_{i}",
                start_time=min(times),
                end_time=max(times),
                label_indices=idxs,
            )
        )

    return HoldoutPlan(intervals=intervals, confidence=confidence)


def _interval_count(label_count: int) -> tuple[int, str]:
    if label_count >= 30:
        return 3, "high"
    if label_count >= 15:
        return 2, "medium"
    return 1, "low"


def _nearest_unused_span(center: int, span: int, label_count: int, used: set[int]) -> list[int]:
    half = span // 2
    start = max(0, center - half)
    end = min(label_count, start + span)
    start = max(0, end - span)
    positions = [pos for pos in range(start, end) if pos not in used]
    if positions:
        return positions
    for pos in range(label_count):
        if pos not in used:
            return [pos]
    raise ValueError("could not allocate non-overlapping holdout labels")
