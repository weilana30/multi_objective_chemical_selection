"""Read saved workflow reports and calculate display-only Pareto views."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOTS = {
    "GenMol": PROJECT / "outputs" / "iterative_dock_pareto",
    "CReM": PROJECT / "outputs" / "crem_dock_pareto",
}
SCORE_COLUMNS = ("lipophilicity_score", "solubility_score")


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def round_file(run: Path, index: int, suffix: str) -> Path:
    name = f"round_{index:02d}_{suffix}"
    nested = run / f"round_{index:02d}" / name
    return nested if nested.is_file() else run / name


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def available_rounds(run: Path) -> list[int]:
    summary = read_json(run / "workflow_summary.json")
    completed = summary.get("completed_rounds")
    if isinstance(completed, list):
        return sorted(int(index) for index in completed if round_file(run, int(index), "docked.csv").is_file())
    indexes = set()
    for path in list(run.glob("round_*_docked.csv")) + list(run.glob("round_*/round_*_docked.csv")):
        try:
            indexes.add(int(path.name.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return sorted(indexes)


def scored_round(run: Path, index: int) -> pd.DataFrame:
    frame = read_csv(round_file(run, index, "property_scored.csv"))
    if not set(SCORE_COLUMNS).issubset(frame.columns):
        return pd.DataFrame()
    frame = frame.copy()
    for column in SCORE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(SCORE_COLUMNS))
    if "canonical_smiles" not in frame:
        frame["canonical_smiles"] = frame["smiles"]
    frame["round"] = index
    frame["run"] = run.name
    return frame


def docked_through(run: Path, index: int) -> pd.DataFrame:
    """Docking records from every completed round through the requested round."""
    frames = []
    next_number = 1
    for number in available_rounds(run):
        if number > index:
            continue
        frame = read_csv(round_file(run, number, "docked.csv"))
        if frame.empty:
            continue
        frame = frame.copy()
        if "canonical_smiles" not in frame and "smiles" in frame:
            frame["canonical_smiles"] = frame["smiles"]
        frame["round"] = number
        frame["molecule_number"] = range(next_number, next_number + len(frame))
        next_number += len(frame)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def scored_through(run: Path, index: int | None = None, cutoff: float | None = None) -> pd.DataFrame:
    rounds = [number for number in available_rounds(run)
              if (index is None or number <= index)
              and (cutoff is None or (round_file(run, number, "property_scored.csv").is_file()
                                     and round_file(run, number, "property_scored.csv").stat().st_mtime <= cutoff))]
    frames = [scored_round(run, number) for number in rounds]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def pareto(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    frame = scored.drop_duplicates("canonical_smiles", keep="first")
    frame = frame.sort_values(list(SCORE_COLUMNS), ascending=False).copy()
    kept = []
    best_solubility = float("-inf")
    last_pair = None
    for index, row in frame.iterrows():
        pair = (row[SCORE_COLUMNS[0]], row[SCORE_COLUMNS[1]])
        if pair[1] > best_solubility or pair == last_pair:
            kept.append(index)
        best_solubility = max(best_solubility, pair[1])
        last_pair = pair
    return frame.loc[kept]


def report_time(run: Path) -> float:
    path = run / "service_status.json"
    return path.stat().st_mtime if path.is_file() else run.stat().st_mtime


def discover_reports() -> list[dict]:
    reports = []
    for generator, root in ROOTS.items():
        if not root.is_dir():
            continue
        for run in root.iterdir():
            if run.is_dir() and ((run / "workflow_summary.json").is_file()
                                 or (run / "run_settings.json").is_file()):
                settings = read_json(run / "run_settings.json")
                reports.append({"generator": generator, "path": str(run),
                                "name": settings.get("report_name") or run.name,
                                "time": report_time(run)})
    return sorted(reports, key=lambda item: item["time"], reverse=True)


def run_front_overlays(selected: Path, through: int | None = None) -> list[tuple[str, pd.DataFrame]]:
    """Cumulative Pareto fronts for one report, excluding all other reports."""
    cumulative = []
    overlays = []
    for index in available_rounds(selected):
        if through is not None and index > through:
            continue
        frame = scored_round(selected, index)
        if frame.empty:
            continue
        cumulative.append(frame)
        overlays.append((f"Round {index + 1}", pareto(pd.concat(cumulative, ignore_index=True))))
    return overlays


def run_evolution(overlays: list[tuple[str, pd.DataFrame]]) -> list[dict]:
    """Measure one run's front size and fixed-reference coverage by completed round."""
    if not overlays:
        return []
    first = overlays[0][1]
    reference = (float(first[SCORE_COLUMNS[0]].min()) - 1,
                 float(first[SCORE_COLUMNS[1]].min()) - 1)
    return [{"label": name, "run": "This run", "coverage": hypervolume(front, reference),
             "front_size": len(front)} for name, front in overlays]


def hypervolume(front: pd.DataFrame, reference: tuple[float, float]) -> float:
    if front.empty:
        return 0.0
    ordered = front.sort_values(SCORE_COLUMNS[0])
    previous_x, reference_y = reference
    area = 0.0
    for _, row in ordered.iterrows():
        x = float(row[SCORE_COLUMNS[0]])
        y = float(row[SCORE_COLUMNS[1]])
        area += max(0.0, x - previous_x) * max(0.0, y - reference_y)
        previous_x = max(previous_x, x)
    return area

