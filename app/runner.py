"""Launch and monitor one local workflow process for the dashboard."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import psutil
import requests
from rdkit import Chem

from advanced import PORTS
from reports import PROJECT, ROOTS, read_json

GENMOL_PYTHON = PROJECT.parent / "genmol_molecule_generator" / ".conda-env" / "bin" / "python"
CREM_PYTHON = PROJECT / ".venv-crem" / "bin" / "python"
CREM_DB = PROJECT / "assets" / "crem" / "chembl33_sa2_f5.db"
SERVICE_PORT_KEYS = {model: key for key, (model, _) in PORTS.items()}
SERVICE_DEFAULTS = {model: default for model, default in (
    ("Admet-AI", 18019), ("Admetica", 8000), ("Admethyst", 18001),
    ("BoiledEgg", 18002), ("ClassicalGSG", 10001), ("MolProperty", 18003),
)}
_lock = threading.Lock()
_process: subprocess.Popen | None = None
_run: Path | None = None


def workflow_python(generator: str) -> Path:
    if generator == "GenMol":
        return Path(os.getenv("GENMOL_PYTHON", str(GENMOL_PYTHON))).expanduser()
    return Path(os.getenv("CREM_PYTHON", str(CREM_PYTHON))).expanduser()


def _health(name: str, base_url: str) -> str | None:
    try:
        response = requests.get(base_url.rstrip("/") + "/health", timeout=2)
        if response.ok and str(response.json().get("status", "")).lower() == "healthy":
            return None
    except (requests.RequestException, ValueError):
        pass
    from urllib.parse import urlparse
    return f"{name} (port {urlparse(base_url).port or 'unknown'})"


def preflight(generator: str, smiles: str, budget: int, rounds: int,
              advanced: dict | None = None) -> list[str]:
    advanced = advanced or {}
    errors = []
    if Chem.MolFromSmiles(smiles or "") is None:
        errors.append("Enter a valid SMILES string.")
    if generator not in ROOTS:
        errors.append("Choose GenMol or CReM.")
        return errors
    if rounds < 1 or budget < rounds:
        errors.append("Set at least one molecule per round.")
    python = workflow_python(generator)
    if not python.is_file():
        errors.append(f"{generator} Python environment is missing: {python}")
    if generator == "CReM" and not Path(advanced.get("crem_db") or CREM_DB).is_file():
        errors.append(f"CReM fragment database is missing: {advanced.get('crem_db') or CREM_DB}")
    if generator == "GenMol" and advanced.get("model_path") and not Path(advanced["model_path"]).is_file():
        errors.append(f"GenMol checkpoint is missing: {advanced['model_path']}")
    reference = advanced.get("reference_smiles") or smiles
    if Chem.MolFromSmiles(reference or "") is None:
        errors.append("Enter a valid docking reference SMILES in Advanced settings.")
    if errors:
        return errors
    selected = list(advanced.get("lipophilicity_members") or [])
    selected += list(advanced.get("solubility_members") or [])
    models = {name.split(":", 1)[0] for name in selected} if selected else set(SERVICE_DEFAULTS)
    services = [("AutoDock Vina", advanced.get("vina_url") or "http://localhost:18025")]
    for model in sorted(models):
        key = SERVICE_PORT_KEYS.get(model)
        port = advanced.get(key) if key else None
        port = port or SERVICE_DEFAULTS.get(model)
        if port is None:
            errors.append(f"Set a service port for {model} in Advanced settings.")
            continue
        services.append((model, f"http://localhost:{port}"))
    if errors:
        return errors
    with ThreadPoolExecutor(max_workers=len(services)) as pool:
        missing = list(filter(None, pool.map(lambda item: _health(*item), services)))
    if missing:
        errors.append("Start these prediction services: " + ", ".join(missing) + ".")
    return errors


def active_run() -> Path | None:
    global _process, _run
    if _process and _process.poll() is None:
        return _run
    _process = None
    for root in ROOTS.values():
        if not root.is_dir():
            continue
        for run in root.glob("dashboard_*"):
            pid = read_json(run / "run_settings.json").get("pid")
            if pid and psutil.pid_exists(pid):
                try:
                    if str(PROJECT) in " ".join(psutil.Process(pid).cmdline()):
                        _run = run
                        return run
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    return None


def launch(generator: str, smiles: str, budget: int, rounds: int, report_name: str = "",
           advanced: dict | None = None) -> Path:
    global _process, _run
    with _lock:
        if active_run():
            raise RuntimeError(f"A run is already active: {_run.name}")
        root = ROOTS[generator]
        options = advanced or {}
        root.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        report_name = report_name.strip() or f"Run {now:%Y-%m-%d %H:%M}"
        slug = re.sub(r"[^a-z0-9]+", "_", report_name.lower()).strip("_")[:48] or "run"
        name = f"dashboard_{slug}_{now:%Y%m%d_%H%M%S_%f}"
        run = root / name
        run.mkdir()
        script = (PROJECT / "scripts" / "iterative_dock_pareto" / "iterative_dock_pareto.py"
                  if generator == "GenMol" else PROJECT / "scripts" / "crem_dock_pareto" / "run_workflow.py")
        reference = options.get("reference_smiles") or smiles
        probe = options.get("service_probe_smiles") or reference
        command = [str(workflow_python(generator)), str(script), "--starting-smiles", smiles,
                   "--caffeine-smiles", reference, "--service-probe-smiles", probe,
                   "--iterations", str(rounds - 1), "--total-candidate-budget", str(budget),
                   "--output-dir", str(run), "--seed", str(options.get("seed", 0)),
                   "--samples-per-mask", str(options.get("samples_per_mask", 2)),
                   "--initial-mask-count", str(options.get("initial_mask_count", 11)),
                   "--min-mask-tokens", str(options.get("min_mask_tokens", 5)),
                   "--max-mask-tokens", str(options.get("max_mask_tokens", 15)),
                   "--candidate-decay", str(options.get("candidate_decay", 0.5)),
                   "--max-pareto-parents", str(options.get("max_pareto_parents", 3)),
                   "--max-generation-batches", str(options.get("max_generation_batches", 5)),
                   "--receptor", str(options.get("receptor", "A2A_3RFM")),
                   "--vina-exhaustiveness", str(options.get("vina_exhaustiveness", 8)),
                   "--vina-poses", str(options.get("vina_poses", 5)),
                   "--vina-url", str(options.get("vina_url", "http://localhost:18025"))]
        if options.get("cache_db"):
            command.extend(["--cache-db", options["cache_db"]])
        if options.get("lipophilicity_members"):
            command.extend(["--lipophilicity-members", *options["lipophilicity_members"]])
        if options.get("solubility_members"):
            command.extend(["--solubility-members", *options["solubility_members"]])
        port_flags = {
            "admet_ai_port": "--admet-ai-port", "admetica_port": "--admetica-port",
            "admethyst_port": "--admethyst-port", "boiled_egg_port": "--boiled-egg-port",
            "classical_gsg_port": "--classical-gsg-port", "mol_property_port": "--mol-property-port",
            "logpprepml_port": "--logpprepml-port",
        }
        for key, value in options.items():
            if key.endswith("_port") and value is not None:
                command.extend([port_flags[key], str(value)])
        if options.get("retry_failed_cache"):
            command.append("--retry-failed-cache")
        if options.get("require_all_services"):
            command.append("--require-all-services")
        if generator == "CReM":
            command.extend(["--crem-db", str(options.get("crem_db") or CREM_DB),
                            "--crem-radius", str(options.get("crem_radius", 1)),
                            "--crem-max-size", str(options.get("crem_max_size", 8)),
                            "--crem-growth-max-atoms", str(options.get("crem_growth_max_atoms", 4)),
                            "--crem-mode", str(options.get("crem_mode", "both")),
                            "--generator-max-output", str(options.get("generator_max_output", 100))])
        else:
            command.extend(["--device", str(options.get("device", "auto")),
                            "--temperature", str(options.get("temperature", 0.8)),
                            "--randomness", str(options.get("randomness", 0.5)),
                            "--gamma", str(options.get("gamma", 0.0))])
            if options.get("model_path"):
                command.extend(["--model-path", options["model_path"]])
        settings = {"generator": generator, "report_name": report_name,
                    "starting_smiles": smiles, "reference_smiles": reference,
                    "total_candidate_budget": budget, "total_rounds": rounds,
                    "advanced_settings": options,
                    "created_at": datetime.now().isoformat(), "command": command}
        (run / "run_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        log = (run / "workflow.log").open("w", encoding="utf-8")
        try:
            _process = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        finally:
            log.close()
        _run = run
        settings["pid"] = _process.pid
        (run / "run_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        return run


def stop(run: Path) -> None:
    global _process
    settings = read_json(run / "run_settings.json")
    pid = settings.get("pid")
    if not pid or not psutil.pid_exists(pid):
        return
    process = psutil.Process(pid)
    if str(PROJECT) not in " ".join(process.cmdline()):
        raise RuntimeError("The saved process ID no longer belongs to this workflow.")
    os.killpg(os.getpgid(pid), signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except psutil.TimeoutExpired:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    progress_path = run / "progress.json"
    progress = read_json(progress_path)
    progress.update({"status": "stopped", "phase": "stopped"})
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    _process = None


def run_status(run: Path) -> dict:
    progress = read_json(run / "progress.json")
    summary = read_json(run / "workflow_summary.json")
    settings = read_json(run / "run_settings.json")
    status = progress.get("status") or summary.get("status") or "starting"
    pid = settings.get("pid")
    if status == "running" and not pid:
        status = "incomplete"
    if status in ("running", "starting") and pid and not psutil.pid_exists(pid):
        status = summary.get("status", "failed")
        if status == "running":
            status = "failed"
    if _run == run and _process and _process.poll() is not None and status in ("running", "starting"):
        status = "failed" if _process.returncode else summary.get("status", "completed")
    return {"status": status, "progress": progress, "summary": summary,
            "settings": settings, "log": run / "workflow.log"}


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "Estimating"
    if seconds < 60:
        return "less than 1 min"
    minutes = max(0, round(seconds / 60))
    return f"about {minutes // 60} h {minutes % 60} min" if minutes >= 60 else f"about {minutes} min"


def etas(progress: dict) -> tuple[str, str]:
    if progress.get("status") != "running":
        return "—", "—"
    timings = progress.get("timings", [])
    rates = {}
    for phase in ("generation", "docking", "property scoring"):
        observed = [item["seconds"] / item["count"] for item in timings
                    if item.get("phase") == phase and item.get("count", 0) > 0]
        if observed:
            rates[phase] = sum(observed) / len(observed)
    phase = progress.get("phase")
    done, total = progress.get("phase_completed", 0), progress.get("phase_total", 0)
    started = progress.get("phase_started_at", 0)
    if done > 0:
        import time
        rates[phase] = (time.time() - started) / done
    budget = progress.get("round_budgets", [])
    index = progress.get("current_round", 0)
    if not budget or index >= len(budget) or phase not in ("generation", "docking", "property scoring"):
        return "Estimating", "Estimating"
    expected_pass_rate = 0.6
    prior = [item for item in timings if item.get("phase") == "property scoring"]
    docks = [item for item in timings if item.get("phase") == "docking"]
    if prior and docks and docks[-1].get("count"):
        expected_pass_rate = min(1, prior[-1].get("count", 0) / docks[-1]["count"])
    weights = {"generation": budget[index], "docking": budget[index],
               "property scoring": max(1, round(budget[index] * expected_pass_rate))}
    phases = ["generation", "docking", "property scoring"]
    remaining = max(0, total - done) * rates[phase] if phase in rates else None
    for next_phase in phases[phases.index(phase) + 1:]:
        if remaining is None or next_phase not in rates:
            remaining = None
            break
        remaining += weights[next_phase] * rates[next_phase]
    if remaining is None or len(rates) < 3:
        return _format_seconds(remaining), "Estimating"
    future = sum(budget[index + 1:])
    per_candidate = rates["generation"] + rates["docking"] + expected_pass_rate * rates["property scoring"]
    return _format_seconds(remaining), _format_seconds(remaining + future * per_candidate)
