#!/usr/bin/env python3
"""Run all local, redistributable reproduction steps in a fixed order."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
commands = [
    [sys.executable, "-m", "pytest", "-q"],
    [sys.executable, "scripts/reproduce_all.py"],
    [sys.executable, "scripts/generate_eigenvector_sign_variation.py"],
    [sys.executable, "scripts/parameter_sensitivity.py"],
    [sys.executable, "scripts/runtime_scaling.py"],
    [sys.executable, "scripts/summarize_external_tntool.py"],
    [sys.executable, "scripts/validate_repository.py"],
]
for command in commands:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
