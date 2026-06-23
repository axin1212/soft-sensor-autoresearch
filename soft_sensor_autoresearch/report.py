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
        subplot_titles=[
            f"{candidate.candidate_id} / {holdout.holdout_name} n={len(holdout.actual)} R²={holdout.r2:.3f}"
            for candidate in plot_candidates
            for holdout in candidate.holdouts
        ],
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

    fig.update_layout(height=max(360, rows * 320), showlegend=False, title_text="Soft Sensor AutoResearch")
    body = fig.to_html(full_html=False, include_plotlyjs="cdn")
    index = _candidate_index(ranked)
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Soft Sensor AutoResearch</title>"
        "</head><body><h1>Soft Sensor AutoResearch</h1>"
        "<p>Actual vs predicted plots include a 45-degree reference line.</p>"
        f"{index}{body}</body></html>",
        encoding="utf-8",
    )
    return path


def _candidate_index(candidates: list[CandidateReport]) -> str:
    rows = []
    for rank, candidate in enumerate(candidates, start=1):
        r2_values = ", ".join(_holdout_detail(h) for h in candidate.holdouts)
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
        "<table><thead><tr><th>Rank</th><th>Candidate</th><th>Mean R²</th><th>Status</th><th>R² / Error</th><th>Selected Features</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _holdout_detail(holdout: HoldoutRunResult) -> str:
    name = html.escape(holdout.holdout_name)
    if holdout.error:
        return f"{name}: error: {html.escape(holdout.error)}"
    return f"{name}: n={len(holdout.actual)}, R²={holdout.r2:.3f}"


def _selected_feature_detail(candidate: CandidateReport) -> str:
    by_holdout = []
    for holdout in candidate.holdouts:
        if not holdout.selected_features:
            continue
        features = ", ".join(html.escape(feature) for feature in holdout.selected_features)
        by_holdout.append(f"{html.escape(holdout.holdout_name)}: {features}")
    return "<br>".join(by_holdout)
