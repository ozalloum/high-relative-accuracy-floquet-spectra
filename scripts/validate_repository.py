#!/usr/bin/env python3
"""Validate the repository's retained results and core reproducibility assets."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
required = [
    "src/tn_floquet.py",
    "scripts/reproduce_all.py",
    "scripts/generate_eigenvector_sign_variation.py",
    "scripts/parameter_sensitivity.py",
    "scripts/runtime_scaling.py",
    "tests/test_numerics.py",
    "data/koev_tntool_bd.csv",
    "data/koev_tntool_reference.csv",
    "data/runtime_scaling_reference.csv",
    "results/koev_tntool_comparison.csv",
    "results/numerical_summary.json",
    "results/eigenvector_summary.json",
    "results/koev_tntool_validation_summary.json",
    "figures/figure_7_eigenvector_sign_variation.pdf",
]
missing = [path for path in required if not (ROOT / path).exists()]
if missing:
    raise SystemExit("Missing required repository files: " + ", ".join(missing))

numerical = json.loads((ROOT / "results/numerical_summary.json").read_text())
eigen = json.loads((ROOT / "results/eigenvector_summary.json").read_text())
tntool = json.loads((ROOT / "results/koev_tntool_validation_summary.json").read_text())
sensitivity = pd.read_csv(ROOT / "data/parameter_sensitivity.csv")
runtime = pd.read_csv(ROOT / "data/runtime_scaling.csv", comment="#")

checks = {
    "structured_error_below_1e-13": numerical["spectrum_benchmark"]["max_relative_error_structured"] < 1e-13,
    "dense_stress_test_has_two_negative_values": numerical["spectrum_benchmark"]["negative_dense_eigenvalues"] == 2,
    "tntool_has_24_positive_modes": tntool["modes"] == 24 and tntool["all_returned_multipliers_positive"],
    "tntool_relative_error_below_2e-15": tntool["maximum_relative_error"] < 2e-15,
    "reference_right_sign_counts_complete": eigen["all_reference_right_sign_counts_match_k_minus_1"],
    "reference_left_sign_counts_complete": eigen["all_reference_left_sign_counts_match_k_minus_1"],
    "dense_right_failed_modes_23_24": eigen["numpy_dense_right_failed_modes"] == [23, 24],
    "dense_left_failed_modes_23_24": eigen["scipy_dense_left_failed_modes"] == [23, 24],
    "sensitivity_has_three_levels": len(sensitivity) == 3,
    "runtime_scan_has_eight_dimensions": len(runtime) == 8,
}
failed = [name for name, value in checks.items() if not value]
summary = {"checks": checks, "all_checks_passed": not failed}
(ROOT / "results/repository_validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if failed:
    raise SystemExit("Validation failures: " + ", ".join(failed))
