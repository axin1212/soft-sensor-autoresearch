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
    horizon_step: int = 0
    error: str | None = None


@dataclass(frozen=True)
class ReportMetadata:
    target_column: str
    data_file: str
    model_type: str
    default_window_minutes: int
    num_train_samples: int
    top_features_n: int
    validation_fraction: float
    forecast_horizons: tuple[int, ...]
    include_frequency_candidate: bool
    tabpfn_device: str | None = None
    tabpfn_fit_mode: str | None = None
    tabpfn_n_estimators: int | None = None
    tpt_device: str | None = None
    tpt_fit_mode: str | None = None
    tpt_n_estimators: int | None = None
    fde_root: str | None = None


@dataclass(frozen=True)
class ReportState:
    candidates: list[CandidateReport]
    metadata: ReportMetadata | None = None


def write_report(path: Path, state: ReportState, top_n: int = 20) -> Path:
    ranked = sorted(state.candidates, key=lambda c: c.score, reverse=True)
    plot_candidates = [candidate for candidate in ranked if candidate.holdouts][:top_n]
    holdout_count = max((len(candidate.holdouts) for candidate in plot_candidates), default=1)
    rows = max(1, len(plot_candidates))
    fig = make_subplots(
        rows=rows,
        cols=holdout_count,
        subplot_titles=[
            _subplot_title(rank, candidate, holdout)
            for rank, candidate in enumerate(plot_candidates, start=1)
            for holdout in candidate.holdouts
        ],
        vertical_spacing=_vertical_spacing(rows),
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

    fig.update_annotations(font_size=11, yshift=6)
    fig.update_layout(
        height=max(420, rows * 460),
        margin={"t": 100, "r": 36, "b": 72, "l": 72},
        showlegend=False,
        title_text="Soft Sensor AutoResearch",
        title={"y": 0.995},
    )
    body = fig.to_html(full_html=False, include_plotlyjs="cdn")
    metadata = _metadata_table(state.metadata)
    index = _candidate_index(ranked)
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Soft Sensor AutoResearch</title>"
        "</head><body><h1>Soft Sensor AutoResearch</h1>"
        "<p>Actual vs predicted plots include a 45-degree reference line.</p>"
        f"{metadata}{index}{body}</body></html>",
        encoding="utf-8",
    )
    return path


def _metadata_table(metadata: ReportMetadata | None) -> str:
    if metadata is None:
        return ""
    rows = [
        ("Target Tag", metadata.target_column),
        ("Data File", metadata.data_file),
        ("Model Type", metadata.model_type),
        ("Default Window Minutes", str(metadata.default_window_minutes)),
        ("ICL Train Samples", str(metadata.num_train_samples)),
        ("Top Features", str(metadata.top_features_n)),
        ("Validation Fraction", f"{metadata.validation_fraction:.2f}"),
        ("Forecast Horizons", ", ".join(str(value) for value in metadata.forecast_horizons)),
        ("Include Frequency Candidate", str(metadata.include_frequency_candidate)),
    ]
    if metadata.model_type == "tabpfn3":
        rows.extend(
            [
                ("TabPFN Device", metadata.tabpfn_device or ""),
                ("TabPFN Fit Mode", metadata.tabpfn_fit_mode or ""),
                ("TabPFN Estimators", "" if metadata.tabpfn_n_estimators is None else str(metadata.tabpfn_n_estimators)),
            ]
        )
    if metadata.model_type == "tpt":
        rows.extend(
            [
                ("TPT Device", metadata.tpt_device or ""),
                ("TPT Fit Mode", metadata.tpt_fit_mode or ""),
                ("TPT Estimators", "" if metadata.tpt_n_estimators is None else str(metadata.tpt_n_estimators)),
            ]
        )
    if metadata.fde_root:
        rows.append(("FDE Root", metadata.fde_root))
    rendered = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(value)}</td></tr>"
        for key, value in rows
    )
    return f"<section><h2>Run Parameters</h2><table><tbody>{rendered}</tbody></table></section>"


def _subplot_title(rank: int, candidate: CandidateReport, holdout: HoldoutRunResult) -> str:
    prefix = f"#{rank}"
    if candidate.horizon_step:
        prefix = f"{prefix} · h+{candidate.horizon_step}"
    return f"{prefix} · {_short_label(holdout.holdout_name, 18)}<br>R²={holdout.r2:.3f}"


def _short_label(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max(1, max_chars - 1)] + "…"


def _vertical_spacing(rows: int) -> float:
    if rows <= 1:
        return 0.08
    return min(0.08, max(0.035, 0.28 / rows))


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
            f"<td>h+{candidate.horizon_step}</td>"
            f"<td>{candidate.score:.4f}</td>"
            f"<td>{html.escape(candidate.status)}</td>"
            f"<td>{detail}</td>"
            f"<td>{selected_features}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Rank</th><th>Candidate</th><th>Horizon</th><th>Mean R²</th><th>Status</th><th>R² / Error</th><th>Selected Features</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _holdout_detail(holdout: HoldoutRunResult) -> str:
    name = html.escape(holdout.holdout_name)
    if holdout.error:
        return f"{name}: error: {html.escape(holdout.error)}"
    actual = np.asarray(holdout.actual, dtype=float)
    predicted = np.asarray(holdout.predictions, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not mask.any():
        return f"{name}: n={len(holdout.actual)}, R²={holdout.r2:.3f}, RMSE=nan, MAE=nan, y_std=nan"
    error = actual[mask] - predicted[mask]
    mae = float(np.mean(np.abs(error)))
    y_std = float(np.std(actual[mask], ddof=1)) if mask.sum() > 1 else float("nan")
    rmse_to_std = holdout.rmse / y_std if np.isfinite(y_std) and y_std > 0 else float("nan")
    return (
        f"{name}: n={len(holdout.actual)}, R²={holdout.r2:.3f}, "
        f"RMSE={holdout.rmse:.4f}, MAE={mae:.4f}, y_std={y_std:.4f}, RMSE/std={rmse_to_std:.2f}"
    )


def _selected_feature_detail(candidate: CandidateReport) -> str:
    by_holdout = []
    for holdout in candidate.holdouts:
        if not holdout.selected_features:
            continue
        features = ", ".join(html.escape(feature) for feature in holdout.selected_features)
        by_holdout.append(f"{html.escape(holdout.holdout_name)}: {features}")
    return "<br>".join(by_holdout)
