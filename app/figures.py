"""Interactive plots for rounds and cumulative evolution within one report."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from reports import (SCORE_COLUMNS, pareto, read_csv, read_json, round_file,
                     run_evolution, run_front_overlays, scored_through)

COLORS = {"ink": "#20304a", "muted": "#a4afbf", "pass": "#168a75",
          "fail": "#7c8ba0", "front": "#e16b49", "selected": "#1c55d1"}


def _layout(figure: go.Figure, x: str, y: str) -> go.Figure:
    figure.update_layout(template="plotly_white", margin=dict(l=42, r=18, t=76, b=55),
                         height=430, hovermode="closest", clickmode="event+select",
                         font=dict(family="Inter, system-ui, sans-serif", color=COLORS["ink"]),
                         legend=dict(orientation="h", x=0, y=1.04, yanchor="bottom", font=dict(size=11)))
    figure.update_xaxes(title=x, zeroline=False, showgrid=True, gridcolor="#edf1f5")
    figure.update_yaxes(title=y, zeroline=False, showgrid=True, gridcolor="#edf1f5")
    return figure


def _selected(figure: go.Figure, frame: pd.DataFrame, selected: str | None,
              x: str, y: str) -> None:
    if not selected or frame.empty or "canonical_smiles" not in frame:
        return
    row = frame[frame["canonical_smiles"] == selected]
    if row.empty:
        return
    figure.add_trace(go.Scatter(x=row[x], y=row[y], mode="markers", name="Selected molecule",
                                marker=dict(size=16, color=COLORS["selected"],
                                            line=dict(width=2, color="white")),
                                customdata=row["canonical_smiles"], hovertemplate="Selected<extra></extra>"))


def docking_figure(run: Path, index: int, selected: str | None) -> go.Figure:
    docked = read_csv(round_file(run, index, "docked.csv"))
    figure = go.Figure()
    baseline = read_json(run / "caffeine_reference.json").get("docking_score")
    if not docked.empty and "docking_score" in docked:
        docked = docked.copy()
        docked["molecule_number"] = range(1, len(docked) + 1)
        docked["docking_score"] = pd.to_numeric(docked["docking_score"], errors="coerce")
        if "canonical_smiles" not in docked:
            docked["canonical_smiles"] = docked["smiles"]
        passing = docked["docking_score"] > baseline if baseline is not None else pd.Series(False, index=docked.index)
        for mask, label, color in ((~passing, "Below reference / failed", COLORS["fail"]),
                                   (passing, "Passed docking", COLORS["pass"])):
            group = docked[mask & docked["docking_score"].notna()]
            figure.add_trace(go.Scatter(x=group["molecule_number"], y=group["docking_score"],
                                        mode="markers", name=label, customdata=group["canonical_smiles"],
                                        marker=dict(color=color, size=8, opacity=.8),
                                        hovertemplate="Molecule %{x}<br>Docking %{y:.3f}<extra></extra>"))
        _selected(figure, docked, selected, "molecule_number", "docking_score")
    if baseline is not None:
        figure.add_hline(y=float(baseline), line_color=COLORS["front"], line_dash="dash",
                         annotation_text=f"Input SMILES reference: {float(baseline):.3f}",
                         annotation_position="top left")
    return _layout(figure, "Molecule number in round", "Docking score · higher is better")


def property_figure(scored: pd.DataFrame, selected: str | None, title: str = "Cumulative Pareto") -> go.Figure:
    figure = go.Figure()
    if scored.empty:
        figure.add_annotation(text="No molecules passed docking in the completed rounds.",
                              x=.5, y=.5, xref="paper", yref="paper", showarrow=False,
                              font=dict(color=COLORS["fail"], size=14))
    else:
        points = scored.drop_duplicates("canonical_smiles", keep="first")
        front = pareto(points)
        figure.add_trace(go.Scatter(x=points[SCORE_COLUMNS[0]], y=points[SCORE_COLUMNS[1]],
                                    mode="markers", name="Property scored", customdata=points["canonical_smiles"],
                                    marker=dict(size=7, color=COLORS["muted"], opacity=.5),
                                    hovertemplate="Lipo %{x:.3f}<br>Solubility %{y:.3f}<extra></extra>"))
        figure.add_trace(go.Scatter(x=front[SCORE_COLUMNS[0]], y=front[SCORE_COLUMNS[1]],
                                    mode="lines+markers", name=title, customdata=front["canonical_smiles"],
                                    line=dict(color=COLORS["front"], width=2),
                                    marker=dict(size=10, color=COLORS["front"]),
                                    hovertemplate="Lipo %{x:.3f}<br>Solubility %{y:.3f}<extra></extra>"))
        _selected(figure, points, selected, *SCORE_COLUMNS)
    figure = _layout(figure, "Lipophilicity score · higher is better",
                     "Solubility score · higher is better")
    if scored.empty:
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
    return figure


def overlay_figure(overlays: list[tuple[str, pd.DataFrame]], selected: str | None) -> go.Figure:
    figure = go.Figure()
    for name, front in overlays:
        figure.add_trace(go.Scatter(x=front[SCORE_COLUMNS[0]], y=front[SCORE_COLUMNS[1]],
                                    mode="lines+markers", name=name, customdata=front["canonical_smiles"],
                                    marker=dict(size=6), line=dict(width=2)))
    if overlays:
        _selected(figure, overlays[-1][1], selected, *SCORE_COLUMNS)
    figure = _layout(figure, "Lipophilicity score", "Solubility score")
    figure.update_layout(height=480, margin=dict(l=42, r=18, t=28, b=115),
                         legend=dict(orientation="h", x=0, y=-0.3, yanchor="top",
                                     entrywidth=92, entrywidthmode="pixels", font=dict(size=11)))
    return figure


def coverage_figure(points: list[dict]) -> go.Figure:
    figure = go.Figure()
    if points:
        figure.add_trace(go.Scatter(x=[point["label"] for point in points],
                                    y=[point["coverage"] for point in points],
                                    customdata=[point["run"] for point in points],
                                    mode="lines+markers", name="Fixed-reference coverage",
                                    line=dict(color=COLORS["pass"], width=3),
                                    hovertemplate="%{customdata}<br>%{x}<br>Coverage %{y:.3f}<extra></extra>"))
    figure = _layout(figure, "Completed round, in run order", "Pareto coverage · larger is better")
    figure.update_xaxes(tickangle=-30)
    return figure


def round_views(run: Path, index: int, selected: str | None):
    overlays = run_front_overlays(run, index)
    return (docking_figure(run, index, selected),
            property_figure(scored_through(run, index), selected),
            overlay_figure(overlays, selected), coverage_figure(run_evolution(overlays)))
