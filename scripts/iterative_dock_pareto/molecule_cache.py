"""Persistent molecule-level score and Pareto-history cache."""
from __future__ import annotations

import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonicalize_smiles(smiles: list[str], python: Path) -> list[str]:
    """Canonicalize a batch with the RDKit installed in the GenMol environment."""
    if not smiles:
        return []
    code = (
        "import json,sys; from rdkit import Chem; "
        "xs=json.load(sys.stdin); "
        "print(json.dumps([Chem.MolToSmiles(m,canonical=True,isomericSmiles=True) "
        "if (m:=Chem.MolFromSmiles(s)) is not None else s.strip() for s in xs]))"
    )
    result = subprocess.run(
        [str(python), "-c", code],
        input=json.dumps(smiles),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


class MoleculeCache:
    """Cache docking/property outcomes and round-by-round Pareto membership."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS molecules (
                protocol TEXT NOT NULL,
                canonical_smiles TEXT NOT NULL,
                smiles TEXT NOT NULL,
                docking_affinity REAL,
                docking_score REAL,
                result_id TEXT,
                lipophilicity_score REAL,
                solubility_score REAL,
                status TEXT NOT NULL,
                error TEXT,
                ever_pareto INTEGER NOT NULL DEFAULT 0,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                PRIMARY KEY (protocol, canonical_smiles)
            );
            CREATE TABLE IF NOT EXISTS run_candidates (
                protocol TEXT NOT NULL,
                run_id TEXT NOT NULL,
                round_index INTEGER NOT NULL,
                canonical_smiles TEXT NOT NULL,
                status TEXT NOT NULL,
                PRIMARY KEY (protocol, run_id, round_index, canonical_smiles)
            );
            CREATE TABLE IF NOT EXISTS pareto_membership (
                protocol TEXT NOT NULL,
                run_id TEXT NOT NULL,
                round_index INTEGER NOT NULL,
                canonical_smiles TEXT NOT NULL,
                PRIMARY KEY (protocol, run_id, round_index, canonical_smiles)
            );
            CREATE TABLE IF NOT EXISTS imported_runs (
                protocol TEXT NOT NULL,
                run_id TEXT NOT NULL,
                PRIMARY KEY (protocol, run_id)
            );
            """
        )
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(molecules)")}
        if "result_id" not in columns:
            self.db.execute("ALTER TABLE molecules ADD COLUMN result_id TEXT")
        self.db.commit()

    def get(self, protocol: str, canonical_smiles: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT * FROM molecules WHERE protocol=? AND canonical_smiles=?",
            (protocol, canonical_smiles),
        ).fetchone()
        return dict(row) if row else None

    def store(self, protocol: str, canonical_smiles: str, smiles: str, **values: Any) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            """INSERT INTO molecules (
                protocol, canonical_smiles, smiles, docking_affinity, docking_score, result_id,
                lipophilicity_score, solubility_score, status, error, first_seen, last_seen
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(protocol, canonical_smiles) DO UPDATE SET
                smiles=excluded.smiles,
                docking_affinity=COALESCE(excluded.docking_affinity, molecules.docking_affinity),
                docking_score=COALESCE(excluded.docking_score, molecules.docking_score),
                result_id=COALESCE(excluded.result_id, molecules.result_id),
                lipophilicity_score=COALESCE(excluded.lipophilicity_score, molecules.lipophilicity_score),
                solubility_score=COALESCE(excluded.solubility_score, molecules.solubility_score),
                status=CASE WHEN excluded.status='scored' THEN 'scored'
                            WHEN molecules.status='scored' THEN 'scored'
                            ELSE excluded.status END,
                error=CASE WHEN excluded.status='scored' THEN NULL
                          ELSE COALESCE(excluded.error, molecules.error) END,
                last_seen=excluded.last_seen""",
            (
                protocol, canonical_smiles, smiles, values.get("docking_affinity"),
                values.get("docking_score"), values.get("result_id"), values.get("lipophilicity_score"),
                values.get("solubility_score"), values.get("status", "docked"),
                values.get("error"), now, now,
            ),
        )
        self.db.commit()

    def mark_candidate(self, protocol: str, run_id: str, round_index: int, canonical_smiles: str, status: str) -> None:
        self.db.execute(
            """INSERT INTO run_candidates VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(protocol, run_id, round_index, canonical_smiles)
            DO UPDATE SET status=excluded.status""",
            (protocol, run_id, round_index, canonical_smiles, status),
        )
        self.db.commit()

    def mark_pareto(self, protocol: str, run_id: str, round_index: int, smiles: list[str]) -> None:
        self.db.executemany(
            "INSERT OR IGNORE INTO pareto_membership VALUES (?, ?, ?, ?)",
            [(protocol, run_id, round_index, canonical) for canonical in smiles],
        )
        self.db.executemany(
            "UPDATE molecules SET ever_pareto=1 WHERE protocol=? AND canonical_smiles=?",
            [(protocol, canonical) for canonical in smiles],
        )
        self.db.commit()

    def run_scored(self, protocol: str, run_id: str) -> list[dict[str, Any]]:
        rows = self.db.execute(
            """SELECT DISTINCT m.* FROM molecules m JOIN run_candidates c
            ON c.protocol=m.protocol AND c.canonical_smiles=m.canonical_smiles
            WHERE m.protocol=? AND c.run_id=? AND m.status='scored'""",
            (protocol, run_id),
        ).fetchall()
        return [dict(row) for row in rows]

    def global_scored(self, protocol: str) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM molecules WHERE protocol=? AND status='scored'",
            (protocol,),
        ).fetchall()
        return [dict(row) for row in rows]

    def record_imported_run(self, protocol: str, run_id: str) -> None:
        self.db.execute("INSERT OR IGNORE INTO imported_runs VALUES (?, ?)", (protocol, run_id))
        self.db.commit()

    def was_imported(self, protocol: str, run_id: str) -> bool:
        return self.db.execute(
            "SELECT 1 FROM imported_runs WHERE protocol=? AND run_id=?",
            (protocol, run_id),
        ).fetchone() is not None

    def report(self, protocol: str) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM molecules WHERE protocol=? ORDER BY canonical_smiles",
            (protocol,),
        ).fetchall()
        report = []
        for row in rows:
            item = dict(row)
            history = self.db.execute(
                """SELECT run_id, round_index FROM pareto_membership
                WHERE protocol=? AND canonical_smiles=? ORDER BY run_id, round_index""",
                (protocol, item["canonical_smiles"]),
            ).fetchall()
            item["pareto_history"] = ";".join(
                f"{entry['run_id']}:round_{entry['round_index']:02d}" for entry in history
            )
            report.append(item)
        return report

    def close(self) -> None:
        self.db.close()
