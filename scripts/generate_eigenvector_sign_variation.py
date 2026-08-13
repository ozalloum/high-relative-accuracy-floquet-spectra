#!/usr/bin/env python3
"""Regenerate the eigenvector sign-variation comparison and validation summary."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tn_floquet import build_dense_matrix, make_graded_parameters  # noqa: E402

DATA = ROOT / "data" / "eigenvectors"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.35,
    "lines.markersize": 4.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_reference_vectors(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    mode_columns = [c for c in frame.columns if c.startswith("mode_")]
    return frame[mode_columns].to_numpy(dtype=float)


def normalize_columns(vectors: np.ndarray) -> np.ndarray:
    out = np.asarray(vectors, dtype=complex).copy()
    for j in range(out.shape[1]):
        norm = np.linalg.norm(out[:, j])
        if norm == 0:
            raise ValueError(f"zero eigenvector in column {j}")
        out[:, j] /= norm
        k = int(np.argmax(np.abs(out[:, j])))
        if abs(out[k, j]) > 0:
            out[:, j] /= out[k, j] / abs(out[k, j])
    return out


def sign_changes(vector: np.ndarray) -> int:
    values = np.real_if_close(vector, tol=1000).real
    signs = np.sign(values)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def match_by_overlap(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = normalize_columns(reference)
    candidate = normalize_columns(candidate)
    overlap = np.abs(reference.conj().T @ candidate)
    rows, cols = linear_sum_assignment(-overlap)
    assignment = np.full(reference.shape[1], -1, dtype=int)
    assignment[rows] = cols
    matched = candidate[:, assignment]
    matched_overlap = overlap[np.arange(reference.shape[1]), assignment]
    return matched, matched_overlap


def patterns_to_sign_matrix(patterns: pd.Series) -> np.ndarray:
    rows = []
    for pattern in patterns.astype(str):
        if len(pattern) != 24 or any(ch not in "+-" for ch in pattern):
            raise ValueError(f"Unexpected sign pattern: {pattern!r}")
        rows.append([1 if ch == "+" else -1 for ch in pattern])
    return np.asarray(rows, dtype=int)


def main() -> None:
    diagnostics = pd.read_csv(DATA / "sign_variation_diagnostics.csv")
    right_reference = load_reference_vectors(DATA / "right_eigenvectors_120dps.csv")
    left_reference = load_reference_vectors(DATA / "left_eigenvectors_120dps.csv")

    params = make_graded_parameters(
        n=24,
        retention_span_decades=40,
        coupling_strength=0.08,
        asymmetry=1.5,
        target_rho=0.85,
    )
    matrix = build_dense_matrix(params)

    numpy_values, numpy_right = np.linalg.eig(matrix)
    scipy_values, scipy_left, _ = linalg.eig(
        matrix, left=True, right=True, check_finite=False, overwrite_a=False
    )

    matched_numpy_right, right_overlap = match_by_overlap(right_reference, numpy_right)
    matched_scipy_left, left_overlap = match_by_overlap(left_reference, scipy_left)

    dense_right_counts = np.array(
        [sign_changes(matched_numpy_right[:, j]) for j in range(params.n)], dtype=int
    )
    dense_left_counts = np.array(
        [sign_changes(matched_scipy_left[:, j]) for j in range(params.n)], dtype=int
    )
    expected = diagnostics["expected_sign_changes"].to_numpy(dtype=int)

    # Recover the eigenvalues associated with the overlap-matched vectors for traceability.
    nr = normalize_columns(numpy_right)
    sr = normalize_columns(scipy_left)
    rr = normalize_columns(right_reference)
    lr = normalize_columns(left_reference)
    overlap_r = np.abs(rr.conj().T @ nr)
    overlap_l = np.abs(lr.conj().T @ sr)
    rows_r, cols_r = linear_sum_assignment(-overlap_r)
    rows_l, cols_l = linear_sum_assignment(-overlap_l)
    assignment_r = np.full(params.n, -1, dtype=int); assignment_r[rows_r] = cols_r
    assignment_l = np.full(params.n, -1, dtype=int); assignment_l[rows_l] = cols_l

    comparison = pd.DataFrame({
        "mode": np.arange(1, params.n + 1),
        "expected_sign_changes": expected,
        "reference_right_sign_changes": diagnostics["right_sign_changes"].astype(int),
        "reference_left_sign_changes": diagnostics["left_sign_changes"].astype(int),
        "numpy_dense_right_sign_changes": dense_right_counts,
        "scipy_dense_left_sign_changes": dense_left_counts,
        "numpy_dense_right_overlap_abs": right_overlap,
        "scipy_dense_left_overlap_abs": left_overlap,
        "numpy_matched_eigenvalue_real": np.real(numpy_values[assignment_r]),
        "numpy_matched_eigenvalue_imag": np.imag(numpy_values[assignment_r]),
        "scipy_matched_eigenvalue_real": np.real(scipy_values[assignment_l]),
        "scipy_matched_eigenvalue_imag": np.imag(scipy_values[assignment_l]),
    })
    comparison.to_csv(DATA / "dense_sign_variation_comparison.csv", index=False)

    right_signs = patterns_to_sign_matrix(diagnostics["right_sign_pattern"])
    left_signs = patterns_to_sign_matrix(diagnostics["left_sign_pattern"])

    fig = plt.figure(figsize=(7.15, 8.1), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.9])
    axes = [fig.add_subplot(grid[i, 0]) for i in range(3)]

    for ax, sign_matrix, title in [
        (axes[0], right_signs, "(a) Right reference eigenvectors"),
        (axes[1], left_signs, "(b) Left reference eigenvectors"),
    ]:
        ax.imshow(sign_matrix, cmap="gray", vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_xticks([0, 3, 7, 11, 15, 19, 23], [1, 4, 8, 12, 16, 20, 24])
        ax.set_yticks([0, 3, 7, 11, 15, 19, 23], [1, 4, 8, 12, 16, 20, 24])
        ax.set_xlabel("Component index")
        ax.set_ylabel("Mode index $k$")
        ax.set_title(title)

    modes = np.arange(1, params.n + 1)
    axes[2].plot(modes, expected, "-", label="Expected / 120-digit reference")
    axes[2].plot(modes, dense_right_counts, "o", fillstyle="none", label="NumPy dense right")
    axes[2].plot(modes, dense_left_counts, "x", label="SciPy dense left")
    axes[2].set_xlabel("Mode index $k$")
    axes[2].set_ylabel("Sign changes")
    axes[2].set_title("(c) Consecutive sign changes")
    axes[2].set_xlim(1, 24)
    axes[2].set_ylim(-0.5, 23.5)
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(frameon=False, ncol=1)

    for ext, kwargs in [("pdf", {}), ("png", {"dpi": 600})]:
        fig.savefig(FIGURES / f"figure_7_eigenvector_sign_variation.{ext}", **kwargs)
    plt.close(fig)

    dense_fail_right = comparison.loc[
        comparison["numpy_dense_right_sign_changes"] != comparison["expected_sign_changes"], "mode"
    ].astype(int).tolist()
    dense_fail_left = comparison.loc[
        comparison["scipy_dense_left_sign_changes"] != comparison["expected_sign_changes"], "mode"
    ].astype(int).tolist()

    reference_condition_estimates = []
    for j in range(params.n):
        r = right_reference[:, j]
        l = left_reference[:, j]
        reference_condition_estimates.append(
            float(np.linalg.norm(r) * np.linalg.norm(l) / abs(np.dot(l, r)))
        )

    summary = {
        "reference_precision_decimal_digits": 120,
        "maximum_reference_eigenvalue_relative_difference": float(
            diagnostics["eigenvalue_relative_difference"].max()
        ),
        "maximum_right_backward_error_120dps": float(
            diagnostics["right_backward_error_120dps"].max()
        ),
        "maximum_left_backward_error_120dps": float(
            diagnostics["left_backward_error_120dps"].max()
        ),
        "maximum_reference_simple_eigenvalue_condition_estimate": float(
            max(reference_condition_estimates)
        ),
        "all_reference_right_sign_counts_match_k_minus_1": bool(
            diagnostics["right_matches_expected"].astype(bool).all()
        ),
        "all_reference_left_sign_counts_match_k_minus_1": bool(
            diagnostics["left_matches_expected"].astype(bool).all()
        ),
        "numpy_dense_right_failed_modes": dense_fail_right,
        "scipy_dense_left_failed_modes": dense_fail_left,
        "minimum_numpy_dense_right_overlap_abs": float(np.min(right_overlap)),
        "minimum_scipy_dense_left_overlap_abs": float(np.min(left_overlap)),
    }
    (RESULTS / "eigenvector_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
