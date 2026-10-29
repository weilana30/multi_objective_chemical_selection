#!/usr/bin/env python3
"""Local dashboard for iterative docking and Pareto reports."""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from urllib.parse import quote

import dash
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, ctx, dcc, html
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from advanced import CHOICES, CREM_DB, NUMBERS, PORTS, TEXT, model_options, normalize as normalize_advanced
from figures import round_views
from reports import (available_rounds, discover_reports, docked_through,
                     read_json, scored_through, ROOTS)
from runner import active_run, etas, launch, preflight, run_status, stop

app = dash.Dash(__name__, title="Molecule search", suppress_callback_exceptions=True)
server = app.server


@server.get("/vendor/3Dmol-min.js")
def vendor_3dmol():
    """Serve the pinned viewer bundle without loading it on every dashboard page."""
    from flask import send_file

    bundle = Path(__file__).resolve().parent / "vendor" / "3Dmol-min.js"
    if not bundle.is_file():
        from flask import abort
        abort(404)
    return send_file(bundle, mimetype="text/javascript", conditional=True, max_age=86400)


@server.get("/report-file")
def report_file():
    """Serve only pose/receptor files contained in a managed report folder."""
    from flask import abort, request, send_file

    report_arg, relative_arg = request.args.get("report", ""), request.args.get("file", "")
    if not report_arg or not relative_arg:
        abort(404)
    report = Path(report_arg).resolve()
    allowed = False
    for root in ROOTS.values():
        try:
            report.relative_to(root.resolve())
            allowed = report.is_dir()
        except ValueError:
            continue
        if allowed:
            break
    candidate = (report / relative_arg).resolve()
    try:
        candidate.relative_to(report)
    except ValueError:
        abort(404)
    if not allowed or not candidate.is_file() or candidate.suffix.lower() not in {".sdf", ".pdb"}:
        abort(404)
    return send_file(candidate, mimetype="chemical/x-mdl-sdfile" if candidate.suffix.lower() == ".sdf"
                     else "chemical/x-pdb", conditional=True)


def card(label, component, help_text=None):
    contents = [html.Span(label, className="field-label"), component]
    if help_text:
        contents.append(html.Small(help_text, className="field-help"))
    return html.Label(contents, className="field")


def advanced_number(key):
    label, default, kind, minimum, maximum, _ = NUMBERS[key]
    return card(label, dcc.Input(id={"type": "advanced-setting", "key": key}, type="number",
                                 min=minimum, max=maximum, step=1 if kind == "int" else "any",
                                 value=default))


def advanced_text(key):
    label, default, _ = TEXT[key]
    return card(label, dcc.Input(id={"type": "advanced-setting", "key": key}, type="text",
                                 value=default, debounce=True))


def advanced_choice(key):
    label, default, choices, _ = CHOICES[key]
    return card(label, dcc.Dropdown(id={"type": "advanced-setting", "key": key},
                                   options=choices, value=default, clearable=False))


def advanced_models(objective):
    return card(f"{objective.title()} predictions",
                dcc.Dropdown(id={"type": "advanced-setting", "key": objective + "_members"},
                             options=model_options()[objective], value=[], multi=True,
                             placeholder="Leave blank to use the default model set"),
                "Select one or more model outputs to include in this score.")


def advanced_check(key, label):
    return card(label, dcc.Checklist(id={"type": "advanced-setting", "key": key},
                                    options=[{"label": "Enable", "value": "enabled"}], value=[]))


app.layout = html.Div([
    dcc.Store(id="active-report"), dcc.Store(id="selected-molecule"),
    dcc.Store(id="molecule-modal-open", data=False),
    dcc.Store(id="plot-state"), dcc.Store(id="completed-rounds"),
    dcc.Interval(id="refresh", interval=4000, n_intervals=0),
    html.Header([html.Div("MOLECULE SEARCH", className="eyebrow"),
                 html.H1("Docking → Pareto"),
                 html.P("Explore candidate molecules as each round finishes.")], className="masthead"),
    html.Main([
        html.Section([
            html.Div([
                html.H2("New run"),
                html.P("Start from a molecule and choose how widely to search."),
                card("Starting SMILES", dcc.Input(id="smiles", type="text",
                    value="Cn1c(=O)c2c(ncn2C)n(C)c1=O", debounce=True)),
                card("Report name", dcc.Input(id="report-name", type="text", maxLength=80,
                                               placeholder="e.g. caffeine search")),
                html.Div([
                    card("Molecule generator", dcc.Dropdown(id="generator", options=["GenMol", "CReM"],
                                                               value="GenMol", clearable=False)),
                    card("Approximate molecules", dcc.Input(id="budget", type="number", min=1, step=1, value=100)),
                    card("Total rounds", dcc.Input(id="rounds", type="number", min=1, step=1, value=2)),
                ], className="field-grid"),
                html.Details([
                    html.Summary("Advanced settings"),
                    html.P("Adjust how molecules are generated, docked, and scored. Blank model selections use the current defaults.",
                           className="advanced-intro"),
                    html.H3("Search and generation"),
                    html.Div([advanced_number(key) for key in (
                        "seed", "samples_per_mask", "initial_mask_count", "min_mask_tokens",
                        "max_mask_tokens", "candidate_decay", "max_pareto_parents",
                        "max_generation_batches")], className="advanced-grid"),
                    html.Div([
                        html.H4("GenMol controls"),
                        html.Div([advanced_text("model_path"), advanced_choice("device"),
                                  advanced_number("temperature"), advanced_number("randomness"),
                                  advanced_number("gamma")], className="advanced-grid"),
                    ], id="genmol-options"),
                    html.Div([
                        html.H4("CReM controls"),
                        html.Div([advanced_text("crem_db"), advanced_choice("crem_mode"),
                                  advanced_number("crem_radius"), advanced_number("crem_max_size"),
                                  advanced_number("crem_growth_max_atoms"),
                                  advanced_number("generator_max_output")], className="advanced-grid"),
                    ], id="crem-options", style={"display": "none"}),
                    html.H3("Docking"),
                    html.Div([advanced_text("reference_smiles"), advanced_text("receptor"),
                              advanced_number("vina_exhaustiveness"), advanced_number("vina_poses"),
                              advanced_text("vina_url")], className="advanced-grid"),
                    html.H3("Property scoring"),
                    html.Div([advanced_models("lipophilicity"), advanced_models("solubility")],
                             className="advanced-grid"),
                    html.Details([html.Summary("Prediction service ports"),
                                  html.Div([card(f"{label} port", dcc.Input(
                                      id={"type": "advanced-setting", "key": key}, type="number",
                                      min=1, max=65535, step=1, value=default))
                                             for key, (label, default) in PORTS.items()],
                                           className="advanced-grid")]),
                    html.H3("Cache and checks"),
                    html.Div([advanced_text("cache_db"),
                              advanced_text("service_probe_smiles")], className="advanced-grid"),
                    html.Div([advanced_check("retry_failed_cache", "Retry previously failed predictions"),
                              advanced_check("require_all_services", "Require every selected prediction model")],
                             className="advanced-grid"),
                    html.P("The dashboard manages each report folder and starts a fresh run. It does not expose resume or output-folder overrides.",
                           className="field-help"),
                ], id="advanced-settings", className="advanced-panel"),
                html.Button("Run search", id="run-button", className="primary"),
            ], className="surface"),
            html.Div([
                html.H2("Open report"),
                html.P("Review a saved search with the same interactive plots."),
                html.Div([html.Strong("RUNNING NOW", className="active-run-label"),
                          html.Div(id="active-run-name"),
                          html.Button("Go to running report", id="active-run-button", className="primary")],
                         id="active-run-card", className="active-run-card", style={"display": "none"}),
                dcc.Dropdown(id="report-picker", placeholder="Choose a saved report"),
                html.Button("Open report", id="open-button", className="secondary"),
            ], className="surface"),
            html.Div(id="notice", className="notice"),
        ], id="landing", className="landing-grid"),
        html.Section([
            html.Div([html.Button("← New / open report", id="back-button", className="text-button"),
                      html.Div(id="report-heading")], className="report-header"),
            html.Div(id="status-panel", className="status-panel"),
            html.Details([html.Summary("Run settings"), html.Div(id="report-settings")],
                         id="run-settings-panel", className="settings-panel"),
            html.Div([html.Button("Stop run", id="stop-button", className="danger"),
                      html.Span(id="stop-message")], className="stop-row"),
            dcc.Tabs(id="view-tabs", value="current", children=[
                dcc.Tab(label="This run", value="current"),
            ]),
            html.P("Click a molecule in either plot to see its structure and scores. Matching points highlight together.",
                   className="caption"),
            html.P(id="plot-context", className="caption"),
            html.Div(id="plot-waiting", className="plot-waiting"),
            html.Div([
                html.Div([html.H3(id="dock-heading", children="Docking"), dcc.Graph(id="dock-plot")],
                         id="dock-panel", className="plot-card"),
                html.Div([html.H3(id="pareto-heading", children="Lipophilicity vs solubility"),
                          dcc.Graph(id="pareto-plot")], className="plot-card"),
                html.Div([html.H3("Pareto evolution by round"), dcc.Graph(id="overlay-plot")],
                         id="overlay-panel", className="plot-card"),
                html.Div([html.H3("Pareto coverage over time"), dcc.Graph(id="coverage-plot"),
                          html.P("Coverage uses one fixed reference point within this run. Higher is better.",
                                 className="caption")], id="coverage-panel", className="plot-card"),
            ], id="plots-grid", className="plots-grid"),
        ], id="report-view", style={"display": "none"}),
    ]),
    html.Div([html.Div([html.Button("×", id="close-modal", className="close-modal"),
                        html.Div(id="molecule-detail")], className="modal-card")],
             id="molecule-modal", className="modal-backdrop", style={"display": "none"}),
], className="app-shell")


@app.callback(Output("report-picker", "options"),
              Input("refresh", "n_intervals"), Input("active-report", "data"),
              State("report-picker", "options"))
def report_options(_, _report, previous):
    running = active_run()
    running_path = str(running) if running else None
    reports = discover_reports()
    reports.sort(key=lambda item: item["path"] != running_path)
    options = [{"label": f"{'● RUNNING · ' if item['path'] == running_path else ''}"
                         f"{item['generator']} · {item['name']}", "value": item["path"]}
               for item in reports]
    return dash.no_update if options == previous else options


@app.callback(Output("active-run-card", "style"), Output("active-run-name", "children"),
              Input("refresh", "n_intervals"), Input("active-report", "data"))
def running_link(_, _report):
    running = active_run()
    if not running:
        return {"display": "none"}, ""
    settings = read_json(running / "run_settings.json")
    return {}, settings.get("report_name") or running.name


@app.callback(Output("active-report", "data"), Output("notice", "children"),
              Output("view-tabs", "value"),
              Input("run-button", "n_clicks"), Input("open-button", "n_clicks"),
              Input("back-button", "n_clicks"), Input("active-run-button", "n_clicks"),
              State("smiles", "value"), State("generator", "value"),
              State("budget", "value"), State("rounds", "value"),
              State("report-name", "value"), State("report-picker", "value"),
              State({"type": "advanced-setting", "key": ALL}, "id"),
              State({"type": "advanced-setting", "key": ALL}, "value"),
              prevent_initial_call=True)
def choose_report(_run, _open, _back, _active, smiles, generator, budget, rounds, name, existing,
                  advanced_ids, advanced_values):
    trigger = ctx.triggered_id
    if trigger == "back-button":
        return None, "", "current"
    if trigger == "active-run-button":
        running = active_run()
        return (str(running), "", "current") if running else (
            dash.no_update, "No search is currently running.", dash.no_update)
    if trigger == "open-button":
        if not existing or not Path(existing).is_dir():
            return dash.no_update, "Choose a saved report first.", dash.no_update
        return existing, "", "current"
    try:
        budget = int(budget)
        rounds = int(rounds)
    except (ValueError, TypeError):
        return dash.no_update, "Enter whole numbers for molecules and rounds.", dash.no_update
    if active_run():
        return dash.no_update, "A search is already running. Open it or stop it before starting another.", dash.no_update
    if len((name or "").strip()) > 80:
        return dash.no_update, "Keep the report name under 80 characters.", dash.no_update
    raw_advanced = {component_id["key"]: value
                    for component_id, value in zip(advanced_ids or [], advanced_values or [])}
    try:
        advanced = normalize_advanced(raw_advanced, generator)
    except ValueError as exc:
        return dash.no_update, str(exc), dash.no_update
    errors = preflight(generator, (smiles or "").strip(), budget, rounds, advanced)
    if errors:
        return dash.no_update, html.Ul([html.Li(error) for error in errors]), dash.no_update
    try:
        report = launch(generator, smiles.strip(), budget, rounds, (name or "").strip(), advanced)
    except (OSError, RuntimeError) as exc:
        return dash.no_update, str(exc), dash.no_update
    return str(report), "", "current"


@app.callback(Output("landing", "style"), Output("report-view", "style"),
              Input("active-report", "data"))
def switch_screen(report):
    return ({"display": "none"}, {}) if report else ({}, {"display": "none"})


@app.callback(Output("run-settings-panel", "open"), Input("active-report", "data"))
def collapse_run_settings(_report):
    return False


@app.callback(Output("genmol-options", "style"), Output("crem-options", "style"),
              Input("generator", "value"))
def generator_controls(generator):
    return ({}, {"display": "none"}) if generator == "GenMol" else (
        {"display": "none"}, {})


@app.callback(Output("status-panel", "children"), Output("report-heading", "children"),
              Output("view-tabs", "children"),
              Output("stop-button", "style"), Output("completed-rounds", "data"),
              Input("refresh", "n_intervals"), Input("active-report", "data"),
              State("completed-rounds", "data"))
def update_status(_, path, previous_rounds):
    if not path:
        tabs = [dcc.Tab(label="This run", value="current")]
        return "", "", tabs if previous_rounds else dash.no_update, {"display": "none"}, None
    run = Path(path)
    state = run_status(run)
    progress, summary, settings = state["progress"], state["summary"], state["settings"]
    completed = available_rounds(run)
    revision = {"report": path, "rounds": completed}
    tabs = [dcc.Tab(label="This run", value="current")]
    tabs += [dcc.Tab(label=f"Round {index}", value=f"round-{index}") for index in completed]
    if revision == previous_rounds:
        tabs = dash.no_update
        revision = dash.no_update
    phase = progress.get("phase", "saved report" if summary else "starting")
    phase_done, phase_total = progress.get("phase_completed", 0), progress.get("phase_total", 0)
    round_total = settings.get("total_rounds") or summary.get("iterations_requested", 0) + 1
    round_number = progress.get("current_round", summary.get("current_round", max(completed, default=0)))
    eta_round, eta_all = etas(progress)
    log_tail = ""
    if state["status"] == "failed" and state["log"].is_file():
        log_tail = "\n".join(state["log"].read_text(encoding="utf-8", errors="replace").splitlines()[-5:])
    panel = [html.Div([html.Div("STATUS", className="metric-label"),
                       html.Strong(state["status"].replace("_", " ").title())], className="metric"),
             html.Div([html.Div("CURRENT PHASE", className="metric-label"),
                       html.Strong(f"Round {round_number + 1}/{round_total} · {phase}"),
                       html.Small(f"{phase_done}/{phase_total} in this phase" if phase_total else
                                  "Preparing" if state["status"] in ("running", "starting") else "")],
                      className="metric"),
             html.Div([html.Div("ROUNDS FINISHED", className="metric-label"),
                       html.Strong(f"{len(completed)}/{round_total}")], className="metric"),
             html.Div([html.Div("CURRENT ROUND ETA", className="metric-label"), html.Strong(eta_round)], className="metric"),
             html.Div([html.Div("WHOLE RUN ETA", className="metric-label"), html.Strong(eta_all)], className="metric")]
    if log_tail:
        panel.append(html.Pre(log_tail, className="error-log"))
    generator = settings.get("generator") or ("GenMol" if run.parent.name == "iterative_dock_pareto" else "CReM")
    heading = [html.Div(generator, className="eyebrow"),
               html.H2(settings.get("report_name") or run.name),
               html.Small(run.name, className="caption")]
    stop_style = {} if state["status"] in ("running", "starting") and settings.get("pid") else {"display": "none"}
    return panel, heading, tabs, stop_style, revision


@app.callback(Output("report-settings", "children"), Input("active-report", "data"))
def show_report_settings(path):
    if not path:
        return ""
    run = Path(path)
    settings = read_json(run / "run_settings.json")
    workflow = read_json(run / "workflow_settings.json")
    summary = read_json(run / "workflow_summary.json")
    service = read_json(run / "service_status.json")
    reference = read_json(run / "caffeine_reference.json")
    vina = service.get("vina", {})
    objectives = service.get("active_objectives", {})
    starting_smiles = settings.get("starting_smiles") or workflow.get("starting_smiles")
    reference_smiles = settings.get("reference_smiles") or workflow.get("caffeine_smiles") or reference.get("smiles")
    shown_smiles = starting_smiles or reference_smiles
    molecule = [html.Code(shown_smiles, className="smiles-code")] if shown_smiles else [
        html.P("Starting molecule was not saved in this report.")]
    if shown_smiles:
        picture = molecule_svg(shown_smiles)
        if picture:
            molecule.append(html.Img(src=picture, alt="2D drawing of starting or reference molecule",
                                     className="settings-molecule"))
    if not starting_smiles and reference_smiles:
        molecule.append(html.Small("The starting SMILES was not saved; the molecule shown is the docking reference.",
                                   className="caption"))
    rows = [
        ("Generator", settings.get("generator") or ("GenMol" if run.parent.name == "iterative_dock_pareto" else "CReM")),
        ("Molecule budget", settings.get("total_candidate_budget") or workflow.get("total_candidate_budget")
         or summary.get("total_candidate_budget")),
        ("Total rounds", settings.get("total_rounds") or
         (workflow["iterations"] + 1 if "iterations" in workflow else
          summary["iterations_requested"] + 1 if "iterations_requested" in summary else None)),
        ("Random seed", workflow.get("seed")),
        ("Docking reference SMILES", reference_smiles),
        ("Docking threshold", summary.get("candidate_filter") or reference.get("filter")),
        ("Receptor", vina.get("receptor") or reference.get("receptor")),
        ("Vina exhaustiveness", vina.get("exhaustiveness")),
        ("Vina poses", vina.get("poses")),
        ("Lipophilicity models", ", ".join(objectives.get("lipophilicity", [])) or None),
        ("Solubility models", ", ".join(objectives.get("solubility", [])) or None),
        ("Candidate decay", summary.get("candidate_decay")),
        ("Maximum Pareto parents", summary.get("max_pareto_parents")),
        ("Maximum generation batches", workflow.get("max_generation_batches")),
        ("Initial mask tokens", summary.get("initial_mask_tokens")),
        ("Initial samples per mask", summary.get("samples_per_mask_initial")),
        ("Round candidate budgets", summary.get("round_candidate_budgets")),
    ]
    if workflow.get("crem_db"):
        rows.extend((label, workflow.get(key)) for label, key in (
            ("CReM fragment database", "crem_db"), ("CReM mode", "crem_mode"),
            ("CReM radius", "crem_radius"), ("CReM maximum fragment size", "crem_max_size"),
            ("CReM growth maximum atoms", "crem_growth_max_atoms")))
    saved = {"dashboard settings": settings, "workflow settings": workflow, "workflow summary": summary,
             "docking and property models": service, "docking reference": reference}
    return [html.Div([html.H4("Molecule"), *molecule], className="settings-molecule-card"),
            html.Div([html.Div([html.Span(label), html.Strong(str(item) if item is not None else "Not recorded")])
                      for label, item in rows], className="settings-grid"),
            html.Details([html.Summary("Advanced run options"),
                          html.Pre(json.dumps(settings.get("advanced_settings", {}), indent=2),
                                   className="settings-json")]),
            html.Details([html.Summary("Complete saved metadata"),
                          html.Pre(json.dumps(saved, indent=2), className="settings-json")])]


@app.callback(Output("stop-message", "children"), Input("stop-button", "n_clicks"),
              State("active-report", "data"), prevent_initial_call=True)
def stop_search(_, path):
    if not path:
        return ""
    try:
        stop(Path(path))
    except (OSError, RuntimeError) as exc:
        return str(exc)
    return "Stopped. Completed round reports remain available."


@app.callback(Output("dock-plot", "figure"), Output("pareto-plot", "figure"),
              Output("overlay-plot", "figure"), Output("coverage-plot", "figure"),
              Output("dock-panel", "style"), Output("overlay-panel", "style"),
              Output("coverage-panel", "style"), Output("dock-heading", "children"),
              Output("pareto-heading", "children"),
              Output("plot-context", "children"), Output("plot-state", "data"),
              Input("active-report", "data"), Input("view-tabs", "value"),
              Input("selected-molecule", "data"), Input("completed-rounds", "data"))
def update_plots(path, tab, selected, _revision):
    blank = go.Figure()
    hidden = {"display": "none"}
    if not path:
        return blank, blank, blank, blank, hidden, hidden, hidden, "Docking", "Lipophilicity vs solubility", "", None
    run = Path(path)
    rounds = available_rounds(run)

    def plotted(ready):
        return {"report": path, "tab": tab, "rounds": rounds, "ready": ready}

    if tab == "current":
        if not rounds:
            return blank, blank, blank, blank, hidden, hidden, hidden, "Docking", "Lipophilicity vs solubility", "", plotted(False)
        index = rounds[-1]
        docking, properties, overlays, coverage = round_views(run, index, selected)
        return docking, properties, overlays, coverage, {}, {}, {}, f"Docking · Round {index} (latest)", (
            f"Properties · cumulative through Round {index}"
        ), (
            f"Showing the latest completed round ({index}). Choose a numbered tab to revisit an earlier round."
        ), plotted(True)
    if tab and tab.startswith("round-"):
        index = int(tab.split("-")[1])
        if index not in rounds:
            return blank, blank, blank, blank, hidden, hidden, hidden, "Docking", "Lipophilicity vs solubility", "", plotted(False)
        docking, properties, overlays, coverage = round_views(run, index, selected)
        return docking, properties, overlays, coverage, {}, {}, {}, f"Docking · Round {index}", (
            f"Properties · cumulative through Round {index}"
        ), (
            f"Round {index} of this run. The Pareto plot includes this and earlier rounds of this run."
        ), plotted(True)
    return blank, blank, blank, blank, hidden, hidden, hidden, "Docking", (
        "Lipophilicity vs solubility"), "Choose a completed round to view its results.", plotted(False)


app.clientside_callback(
    """function(path, tab, plotted, revision) {
        if (!path) return [{display: 'none'}, '', {display: 'none'}];
        if (!plotted || !revision || plotted.report !== path || plotted.tab !== tab ||
            revision.report !== path || JSON.stringify(plotted.rounds) !== JSON.stringify(revision.rounds)) {
            return [{display: 'none'}, 'Loading this report…', {display: 'none'}];
        }
        if (!plotted.ready) {
            return [{display: 'none'}, 'No completed rounds yet. The first plots will appear when Round 0 finishes.',
                    {display: 'none'}];
        }
        return [{}, '', {}];
    }""",
    Output("plots-grid", "style"), Output("plot-waiting", "children"), Output("plot-context", "style"),
    Input("active-report", "data"), Input("view-tabs", "value"),
    Input("plot-state", "data"), Input("completed-rounds", "data"),
)


@app.callback(Output("selected-molecule", "data"), Output("molecule-modal-open", "data"),
              Input("dock-plot", "clickData"), Input("pareto-plot", "clickData"),
              Input("overlay-plot", "clickData"), Input("close-modal", "n_clicks"),
              Input("active-report", "data"),
              prevent_initial_call=True)
def select_molecule(dock_click, pareto_click, overlay_click, _, _report):
    trigger = ctx.triggered_id
    if trigger == "close-modal":
        return dash.no_update, False
    if trigger == "active-report":
        return None, False
    data = {"dock-plot": dock_click, "pareto-plot": pareto_click,
            "overlay-plot": overlay_click}.get(trigger)
    if not data or not data.get("points"):
        return dash.no_update, dash.no_update
    return data["points"][0].get("customdata"), True


def molecule_svg(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DSVG(380, 240)
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    data = base64.b64encode(drawer.GetDrawingText().encode("utf-8")).decode("ascii")
    return "data:image/svg+xml;base64," + data


def value(row, field):
    if row is None:
        return "—"
    number = row.get(field)
    try:
        value = float(number)
        return f"{value:.3f}" if math.isfinite(value) else "—"
    except (ValueError, TypeError):
        return "—"


@app.callback(Output("molecule-detail", "children"), Output("molecule-modal", "style"),
              Input("selected-molecule", "data"), Input("active-report", "data"),
              Input("view-tabs", "value"), Input("molecule-modal-open", "data"))
def molecule_details(selected, path, tab, modal_open):
    if not selected or not path or not modal_open:
        return "", {"display": "none"}
    run = Path(path)
    docked = None
    if tab == "current" or (tab and tab.startswith("round-")):
        index = max(available_rounds(run), default=-1) if tab == "current" else int(tab.split("-")[1])
        docked = docked_through(run, index)
        scored = scored_through(run, index)
    else:
        return "", {"display": "none"}
    dock_row = None
    if docked is not None and not docked.empty and "canonical_smiles" in docked:
        found = docked[docked["canonical_smiles"] == selected]
        if not found.empty:
            dock_row = found.iloc[0]
    score_row = None
    if not scored.empty:
        found = scored[scored["canonical_smiles"] == selected]
        if not found.empty:
            score_row = found.iloc[0]
    row = dock_row if dock_row is not None else score_row
    if row is None:
        return "", {"display": "none"}
    smiles = str(row.get("smiles", selected))
    picture = molecule_svg(smiles)
    number = str(row.get("candidate_id", "—"))
    details = [html.Div("MOLECULE", className="eyebrow"), html.H2(number),
               html.Code(smiles, className="smiles-code")]
    if picture:
        details.append(html.Img(src=picture, alt="2D molecular structure", className="molecule-image"))
        details.append(html.Small("2D drawing from the SMILES; this is not a docked 3D pose.", className="caption"))
    pose_path = row.get("pose_path") if row is not None else None
    pose_path = str(pose_path) if pose_path is not None and str(pose_path).strip() not in ("", "nan", "None") else None
    summary = read_json(run / "workflow_summary.json")
    workflow = read_json(run / "workflow_settings.json")
    service = read_json(run / "service_status.json")
    receptor_path = (service.get("receptor_path") or summary.get("receptor_path")
                     or workflow.get("receptor_path"))
    if pose_path and receptor_path:
        pose_url = f"/report-file?report={quote(str(run), safe='')}&file={quote(pose_path, safe='/')}"
        receptor_url = f"/report-file?report={quote(str(run), safe='')}&file={quote(str(receptor_path), safe='/')}"
        details.extend([
            html.Div([html.Div(id="molecule-3d-viewer", **{"data-pose-url": pose_url,
                         "data-receptor-url": receptor_url}),
                      html.Div("Loading docked pose…", id="molecule-3d-status", className="viewer-status")],
                     className="viewer-shell"),
            html.Div([html.Button("Reset view", id="viewer-reset", className="viewer-control"),
                      html.Button("Hide receptor", id="viewer-toggle-receptor", className="viewer-control",
                                  **{"data-visible": "true"})], className="viewer-controls"),
        ])
    else:
        details.append(html.P("A saved docked 3D pose is not available for this molecule.",
                              className="viewer-fallback"))
    details.append(html.Div([
        html.Div([html.Span("Docking"), html.Strong(value(row, "docking_score"))]),
        html.Div([html.Span("Lipophilicity"), html.Strong(value(score_row, "lipophilicity_score"))]),
        html.Div([html.Span("Solubility"), html.Strong(value(score_row, "solubility_score"))]),
    ], className="molecule-values"))
    if score_row is None:
        details.append(html.P("No property point: this molecule did not pass docking or property scoring failed."))
    return details, {}


if __name__ == "__main__":
    app.run_server(host="127.0.0.1", port=8050, debug=False)
