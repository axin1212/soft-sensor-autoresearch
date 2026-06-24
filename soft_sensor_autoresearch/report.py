from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import html

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from soft_sensor_autoresearch.model_runner import HoldoutRunResult


@dataclass(frozen=True)
class CandidateReport:
    candidate_id: str
    score: float
    status: str
    holdouts: list[HoldoutRunResult] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class ReportState:
    candidates: list[CandidateReport]


def write_report(path: Path, state: ReportState, top_n: int = 20) -> Path:
    ranked = sorted(state.candidates, key=lambda c: c.score, reverse=True)
    plot_candidates = [candidate for candidate in ranked if candidate.holdouts][:top_n]
    holdout_count = max((len(candidate.holdouts) for candidate in plot_candidates), default=1)
    rows = max(1, len(plot_candidates))
    fig = make_subplots(
        rows=rows,
        cols=holdout_count,
        subplot_titles=_subplot_titles(plot_candidates, holdout_count),
        vertical_spacing=min(0.10, 0.65 / max(rows - 1, 1)),
        horizontal_spacing=0.08,
    )

    for row, candidate in enumerate(plot_candidates, start=1):
        for col, holdout in enumerate(candidate.holdouts, start=1):
            actual = np.asarray(holdout.actual, dtype=float)
            predicted = np.asarray(holdout.predictions, dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=actual,
                    y=predicted,
                    mode="markers",
                    name=f"{candidate.candidate_id}:{holdout.holdout_name}",
                    hovertemplate="actual=%{x}<br>predicted=%{y}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            finite = np.concatenate([actual[np.isfinite(actual)], predicted[np.isfinite(predicted)]])
            if len(finite):
                low = float(np.min(finite))
                high = float(np.max(finite))
                fig.add_trace(
                    go.Scatter(
                        x=[low, high],
                        y=[low, high],
                        mode="lines",
                        name="45-degree reference",
                        line={"dash": "dash", "color": "#666"},
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )
            fig.update_xaxes(title_text="Actual", row=row, col=col)
            fig.update_yaxes(title_text="Predicted", row=row, col=col)

    fig.update_annotations(font_size=11)
    fig.update_layout(
        height=max(420, rows * 390),
        margin={"t": 90, "b": 70, "l": 70, "r": 35},
        showlegend=False,
        title_text="Soft Sensor AutoResearch",
    )
    body = fig.to_html(full_html=False, include_plotlyjs="cdn")
    index = _candidate_index(ranked)
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Soft Sensor AutoResearch</title>"
        f"<style>{_report_css()}</style></head><body><h1>Soft Sensor AutoResearch</h1>"
        "<p>Actual vs predicted plots include a 45-degree reference line.</p>"
        f"{index}{body}</body></html>",
        encoding="utf-8",
    )
    return path


def _candidate_index(candidates: list[CandidateReport]) -> str:
    rows = []
    for rank, candidate in enumerate(candidates, start=1):
        r2_values = "".join(_holdout_detail(h) for h in candidate.holdouts)
        detail = r2_values or html.escape(candidate.error or "")
        selected_features = _selected_feature_detail(candidate)
        rows.append(
            "<tr>"
            f"<td>#{rank}</td>"
            f"<td>{html.escape(candidate.candidate_id)}</td>"
            f"<td>{candidate.score:.4f}</td>"
            f"<td>{html.escape(candidate.status)}</td>"
            f"<td>{detail}</td>"
            f"<td>{selected_features}</td>"
            "</tr>"
        )
    return (
        "<table class='candidate-index'><thead><tr><th>Rank</th><th>Candidate</th><th>Mean R²</th><th>Status</th><th>R² / Error</th><th>Selected Features</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _holdout_detail(holdout: HoldoutRunResult) -> str:
    name = html.escape(holdout.holdout_name)
    if holdout.error:
        return f"<div class='holdout-detail'><strong>{name}</strong>: error: {html.escape(holdout.error)}</div>"
    actual = np.asarray(holdout.actual, dtype=float)
    predicted = np.asarray(holdout.predictions, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not mask.any():
        return (
            f"<div class='holdout-detail'><strong>{name}</strong>: "
            f"n={len(holdout.actual)}, R²={holdout.r2:.3f}, RMSE=nan, MAE=nan, y_std=nan</div>"
        )
    error = actual[mask] - predicted[mask]
    mae = float(np.mean(np.abs(error)))
    y_std = float(np.std(actual[mask], ddof=1)) if mask.sum() > 1 else float("nan")
    rmse_to_std = holdout.rmse / y_std if np.isfinite(y_std) and y_std > 0 else float("nan")
    return (
        f"<div class='holdout-detail'><strong>{name}</strong>: "
        f"n={len(holdout.actual)}, R²={holdout.r2:.3f}, "
        f"RMSE={holdout.rmse:.4f}, MAE={mae:.4f}, y_std={y_std:.4f}, RMSE/std={rmse_to_std:.2f}</div>"
    )


def _selected_feature_detail(candidate: CandidateReport) -> str:
    groups = []
    total_features = 0
    for holdout in candidate.holdouts:
        if not holdout.selected_features:
            continue
        total_features += len(holdout.selected_features)
        features = "".join(f"<li>{html.escape(feature)}</li>" for feature in holdout.selected_features)
        groups.append(
            "<div class='feature-group'>"
            f"<div class='feature-holdout'>{html.escape(holdout.holdout_name)} "
            f"({len(holdout.selected_features)})</div>"
            f"<ul>{features}</ul>"
            "</div>"
        )
    if not groups:
        return ""
    entry_label = "feature entry" if total_features == 1 else "feature entries"
    holdout_label = "holdout" if len(groups) == 1 else "holdouts"
    summary = f"{total_features} {entry_label} across {len(groups)} {holdout_label}"
    return f"<details class='feature-details'><summary>{html.escape(summary)}</summary>{''.join(groups)}</details>"


def _subplot_titles(candidates: list[CandidateReport], holdout_count: int) -> list[str]:
    titles = []
    for candidate in candidates:
        for index in range(holdout_count):
            if index >= len(candidate.holdouts):
                titles.append("")
                continue
            holdout = candidate.holdouts[index]
            titles.append(
                f"{_short_label(candidate.candidate_id)} / {_short_label(holdout.holdout_name)} / R²={holdout.r2:.2f}"
            )
    return titles


def _short_label(value: str, max_len: int = 18) -> str:
    if len(value) <= max_len:
        return value
    return f"{value[: max_len - 1]}…"


def _report_css() -> str:
    return """
body {
  color: #1f2933;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 24px;
}
.candidate-index {
  border-collapse: collapse;
  font-size: 13px;
  table-layout: fixed;
  width: 100%;
}
.candidate-index th,
.candidate-index td {
  border: 1px solid #d9e2ec;
  padding: 8px;
  text-align: left;
  vertical-align: top;
  word-break: break-word;
}
.candidate-index th {
  background: #f5f7fa;
  font-weight: 600;
}
.candidate-index th:nth-child(1),
.candidate-index td:nth-child(1) {
  width: 56px;
}
.candidate-index th:nth-child(2),
.candidate-index td:nth-child(2) {
  width: 150px;
}
.candidate-index th:nth-child(3),
.candidate-index td:nth-child(3) {
  width: 90px;
}
.candidate-index th:nth-child(4),
.candidate-index td:nth-child(4) {
  width: 86px;
}
.holdout-detail + .holdout-detail {
  margin-top: 5px;
}
.feature-details summary {
  cursor: pointer;
  white-space: nowrap;
}
.feature-group {
  margin-top: 8px;
}
.feature-holdout {
  font-weight: 600;
  margin-bottom: 4px;
}
.feature-details ul {
  margin: 0 0 0 18px;
  padding: 0;
}
.feature-details li {
  line-height: 1.35;
  margin: 2px 0;
}
"""
