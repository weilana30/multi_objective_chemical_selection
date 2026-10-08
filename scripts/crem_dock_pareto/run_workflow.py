#!/usr/bin/env python3
"""Run the iterative dock-first Pareto workflow with CReM analog generation."""
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS / "generator_adapters"))
from common import install

if __name__ == "__main__":
    install("crem")
