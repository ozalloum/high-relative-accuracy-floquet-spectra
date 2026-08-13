#!/usr/bin/env python3
"""Summarize the retained measured TNTool validation table."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "results" / "koev_tntool_comparison.csv"
frame = pd.read_csv(path)
summary = {
    "source": "retained measured output from the official TNTool TNEigenValues routine",
    "modes": int(len(frame)),
    "all_returned_multipliers_positive": bool((frame["koev_tntool"] > 0).all()),
    "maximum_absolute_error": float(frame["absolute_error"].max()),
    "maximum_relative_error": float(frame["relative_error"].max()),
    "elapsed_seconds_recorded": float(frame["elapsed_seconds"].iloc[0]),
    "regeneration": "Install TNTool separately and run external/run_tntool_comparison.m",
}
(ROOT / "results" / "koev_tntool_validation_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print(json.dumps(summary, indent=2))
