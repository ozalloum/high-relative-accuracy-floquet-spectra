from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import pandas as pd
from scipy import linalg as sla
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tn_floquet import build_dense_matrix, make_graded_parameters  # noqa: E402

DATA_DIR = ROOT / "data" / "eigenvectors"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def _factor_native_matrix_mp(params, dps: int) -> mp.matrix:
    mp.mp.dps = dps
    n = params.n
    A = mp.eye(n)
    for r in range(1, n):
        row = params.lower[r]
        assert row is not None
        for i in range(1, r + 1):
            index = n - r + i
            value = mp.mpf(str(row[i]))
            source = index - 1
            target = index - 2
            for j in range(n):
                A[j, target] += value * A[j, source]
    for col, value_float in enumerate(params.diagonal):
        value = mp.mpf(str(value_float))
        for row_index in range(n):
            A[row_index, col] *= value
    for r in range(n - 1, 0, -1):
        row = params.upper[r]
        assert row is not None
        for i in range(1, r + 1):
            index = n - i + 1
            value = mp.mpf(str(row[i]))
            source = index - 2
            target = index - 1
            for j in range(n):
                A[j, target] += value * A[j, source]
    return A


def _normalize_orient(v: mp.matrix) -> mp.matrix:
    out = mp.matrix([mp.re(z) for z in v])
    norm = mp.sqrt(mp.fsum(z * z for z in out))
    out /= norm
    pivot = max(range(len(out)), key=lambda j: abs(out[j]))
    if out[pivot] < 0:
        out = -out
    return out


def _sign_metrics(v: mp.matrix) -> tuple[int, str, list[int], mp.mpf]:
    signs = [1 if z > 0 else -1 if z < 0 else 0 for z in v]
    nonzero = [s for s in signs if s != 0]
    changes = sum(a != b for a, b in zip(nonzero, nonzero[1:]))
    pattern = "".join("+" if s > 0 else "-" if s < 0 else "0" for s in signs)
    locations = [i + 1 for i in range(len(signs) - 1) if signs[i] * signs[i + 1] < 0]
    min_relative = min(abs(z) for z in v) / max(abs(z) for z in v)
    return changes, pattern, locations, min_relative


def _write_vectors(path: Path, vectors: list[mp.matrix]) -> None:
    n = len(vectors)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["component"] + [f"mode_{k + 1}" for k in range(n)])
        for i in range(n):
            writer.writerow([i + 1] + [mp.nstr(vectors[k][i], 60) for k in range(n)])


def _normalize_columns(V: np.ndarray) -> np.ndarray:
    V = V.astype(complex)
    return V / np.linalg.norm(V, axis=0, keepdims=True)


def _assignment(reference: np.ndarray, dense: np.ndarray) -> dict[int, int]:
    overlaps = np.abs(reference.conj().T @ dense)
    rows, cols = linear_sum_assignment(-overlaps)
    return {int(r): int(c) for r, c in zip(rows, cols)}


def _orient_against(reference: np.ndarray, vector: np.ndarray) -> np.ndarray:
    pivot = int(np.argmax(np.abs(vector)))
    oriented = vector * np.exp(-1j * np.angle(vector[pivot]))
    oriented = oriented.real
    if np.dot(reference, oriented) < 0:
        oriented = -oriented
    return oriented


def _sign_count_numpy(v: np.ndarray) -> tuple[int, str]:
    signs = np.sign(v)
    nonzero = signs[signs != 0]
    count = int(np.sum(nonzero[1:] != nonzero[:-1]))
    pattern = "".join("+" if q > 0 else "-" if q < 0 else "0" for q in signs)
    return count, pattern


def _plot_sign_variation(sign_frame: pd.DataFrame, dense_frame: pd.DataFrame) -> None:
    right_patterns = np.array(
        [[1 if c == "+" else -1 if c == "-" else 0 for c in pattern]
         for pattern in sign_frame["right_sign_pattern"]],
        dtype=float,
    )
    left_patterns = np.array(
        [[1 if c == "+" else -1 if c == "-" else 0 for c in pattern]
         for pattern in sign_frame["left_sign_pattern"]],
        dtype=float,
    )

    fig = plt.figure(figsize=(8.0, 8.8), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.8])
    axes = [fig.add_subplot(grid[i, 0]) for i in range(3)]

    for ax, data, title in [
        (axes[0], right_patterns, "(a)"),
        (axes[1], left_patterns, "(b)"),
    ]:
        ax.imshow(data, aspect="auto", interpolation="nearest", cmap="binary", vmin=-1, vmax=1)
        ax.set_title(title, loc="left")
        ax.set_ylabel("Mode index $k$")
        ax.set_yticks([0, 3, 7, 11, 15, 19, 23])
        ax.set_yticklabels([1, 4, 8, 12, 16, 20, 24])
        ax.set_xticks([0, 3, 7, 11, 15, 19, 23])
        ax.set_xticklabels([1, 4, 8, 12, 16, 20, 24])
        ax.set_xlabel("Component index")

    mode = dense_frame["mode"].to_numpy()
    expected = dense_frame["expected_sign_changes"].to_numpy()
    axes[2].plot(mode, expected, "--", label="Expected / 120-digit reference")
    axes[2].plot(mode, dense_frame["numpy_right_sign_changes"], "o", ms=4, label="NumPy dense right")
    axes[2].plot(mode, dense_frame["scipy_left_sign_changes"], "s", ms=4, label="SciPy dense left")
    axes[2].set_title("(c)", loc="left")
    axes[2].set_xlabel("Mode index $k$")
    axes[2].set_ylabel("Sign changes")
    axes[2].set_xlim(1, 24)
    axes[2].set_ylim(-0.5, 23.8)
    axes[2].legend(loc="lower right", ncol=1, fontsize=8)
    axes[2].grid(True, alpha=0.25)

    pdf_path = FIGURES_DIR / "figure_7_eigenvector_sign_variation.pdf"
    png_path = FIGURES_DIR / "figure_7_eigenvector_sign_variation.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    dps = 120
    mp.mp.dps = dps
    params = make_graded_parameters(
        n=24,
        retention_span_decades=40.0,
        coupling_strength=0.08,
        asymmetry=1.5,
        target_rho=0.85,
    )
    n = params.n
    A = _factor_native_matrix_mp(params, dps)

    right_values, right_vectors_raw = mp.eig(A, left=False, right=True)
    left_values, left_vectors_raw = mp.eig(A.T, left=False, right=True)
    right_order = sorted(range(n), key=lambda j: mp.re(right_values[j]), reverse=True)
    left_order = sorted(range(n), key=lambda j: mp.re(left_values[j]), reverse=True)

    reference_frame = pd.read_csv(ROOT / "data" / "koev_tntool_reference.csv", dtype=str)
    reference_values = [mp.mpf(x) for x in reference_frame["independent_reference_decimal"]]

    matrix_norm = mp.norm(A, 2)
    right_vectors: list[mp.matrix] = []
    left_vectors: list[mp.matrix] = []
    rows: list[dict[str, object]] = []
    condition_estimates: list[mp.mpf] = []

    for k in range(n):
        ir = right_order[k]
        il = left_order[k]
        lam = mp.re(right_values[ir])
        lam_left = mp.re(left_values[il])
        vr = _normalize_orient(right_vectors_raw[:, ir])
        vl = _normalize_orient(left_vectors_raw[:, il])

        right_residual = mp.norm(A * vr - lam * vr, 2) / ((matrix_norm + abs(lam)) * mp.norm(vr, 2))
        left_residual = mp.norm(A.T * vl - lam_left * vl, 2) / ((matrix_norm + abs(lam_left)) * mp.norm(vl, 2))
        right_count, right_pattern, right_locations, right_min_relative = _sign_metrics(vr)
        left_count, left_pattern, left_locations, left_min_relative = _sign_metrics(vl)
        eigenvalue_relative_difference = abs(lam - reference_values[k]) / abs(reference_values[k])
        inner = abs(mp.fsum(vl[j] * vr[j] for j in range(n)))
        condition_estimate = (mp.norm(vl, 2) * mp.norm(vr, 2)) / inner

        right_vectors.append(vr)
        left_vectors.append(vl)
        condition_estimates.append(condition_estimate)
        rows.append(
            {
                "mode": k + 1,
                "eigenvalue_120dps": mp.nstr(lam, 60),
                "independent_reference_decimal": mp.nstr(reference_values[k], 60),
                "eigenvalue_relative_difference": mp.nstr(eigenvalue_relative_difference, 22),
                "right_backward_error_120dps": mp.nstr(right_residual, 22),
                "left_backward_error_120dps": mp.nstr(left_residual, 22),
                "eigenvector_condition_estimate": mp.nstr(condition_estimate, 22),
                "expected_sign_changes": k,
                "right_sign_changes": right_count,
                "left_sign_changes": left_count,
                "right_matches_expected": right_count == k,
                "left_matches_expected": left_count == k,
                "right_sign_pattern": right_pattern,
                "left_sign_pattern": left_pattern,
                "right_change_locations": ";".join(map(str, right_locations)),
                "left_change_locations": ";".join(map(str, left_locations)),
                "right_min_component_over_max": mp.nstr(right_min_relative, 22),
                "left_min_component_over_max": mp.nstr(left_min_relative, 22),
            }
        )

    _write_vectors(DATA_DIR / "right_eigenvectors_120dps.csv", right_vectors)
    _write_vectors(DATA_DIR / "left_eigenvectors_120dps.csv", left_vectors)
    sign_frame = pd.DataFrame(rows)
    sign_frame.to_csv(DATA_DIR / "sign_variation_diagnostics.csv", index=False)

    # Binary64 dense eigenvectors are used only as a comparison with the
    # high-precision reference vectors, not as an independent reference.
    A64 = build_dense_matrix(params)
    numpy_values, numpy_right = np.linalg.eig(A64)
    scipy_values, scipy_left, scipy_right = sla.eig(A64, left=True, right=True)
    R64 = np.column_stack([[float(right_vectors[k][i]) for i in range(n)] for k in range(n)])
    L64 = np.column_stack([[float(left_vectors[k][i]) for i in range(n)] for k in range(n)])
    numpy_right = _normalize_columns(numpy_right)
    scipy_right = _normalize_columns(scipy_right)
    scipy_left = _normalize_columns(scipy_left)

    numpy_map = _assignment(R64, numpy_right)
    scipy_right_map = _assignment(R64, scipy_right)
    scipy_left_map = _assignment(L64, scipy_left)

    dense_rows: list[dict[str, object]] = []
    for k in range(n):
        vn = _orient_against(R64[:, k], numpy_right[:, numpy_map[k]])
        vsr = _orient_against(R64[:, k], scipy_right[:, scipy_right_map[k]])
        vsl = _orient_against(L64[:, k], scipy_left[:, scipy_left_map[k]])
        count_n, pattern_n = _sign_count_numpy(vn)
        count_sr, pattern_sr = _sign_count_numpy(vsr)
        count_sl, pattern_sl = _sign_count_numpy(vsl)
        overlap_n = abs(float(np.dot(R64[:, k], vn)))
        overlap_sr = abs(float(np.dot(R64[:, k], vsr)))
        overlap_sl = abs(float(np.dot(L64[:, k], vsl)))
        dense_rows.append(
            {
                "mode": k + 1,
                "expected_sign_changes": k,
                "numpy_right_overlap": f"{overlap_n:.17e}",
                "scipy_right_overlap": f"{overlap_sr:.17e}",
                "scipy_left_overlap": f"{overlap_sl:.17e}",
                "numpy_right_angle_deg": f"{math.degrees(math.acos(min(1.0, overlap_n))):.12e}",
                "scipy_right_angle_deg": f"{math.degrees(math.acos(min(1.0, overlap_sr))):.12e}",
                "scipy_left_angle_deg": f"{math.degrees(math.acos(min(1.0, overlap_sl))):.12e}",
                "numpy_right_sign_changes": count_n,
                "scipy_right_sign_changes": count_sr,
                "scipy_left_sign_changes": count_sl,
                "numpy_right_matches_expected": count_n == k,
                "scipy_right_matches_expected": count_sr == k,
                "scipy_left_matches_expected": count_sl == k,
                "numpy_right_sign_pattern": pattern_n,
                "scipy_right_sign_pattern": pattern_sr,
                "scipy_left_sign_pattern": pattern_sl,
                "numpy_matched_eigenvalue": f"{numpy_values[numpy_map[k]].real:.17e}",
                "scipy_matched_eigenvalue": f"{scipy_values[scipy_right_map[k]].real:.17e}",
            }
        )

    dense_frame = pd.DataFrame(dense_rows)
    dense_frame.to_csv(DATA_DIR / "dense_vs_high_precision_eigenvectors.csv", index=False)

    summary = {
        "matrix_dimension": n,
        "working_precision_decimal_digits": dps,
        "max_eigenvalue_relative_difference_vs_stored_multiprecision_reference": mp.nstr(
            max(mp.mpf(r["eigenvalue_relative_difference"]) for r in rows), 22
        ),
        "max_right_backward_error_120dps": mp.nstr(
            max(mp.mpf(r["right_backward_error_120dps"]) for r in rows), 22
        ),
        "max_left_backward_error_120dps": mp.nstr(
            max(mp.mpf(r["left_backward_error_120dps"]) for r in rows), 22
        ),
        "max_simple_eigenvalue_condition_estimate": mp.nstr(max(condition_estimates), 22),
        "all_24_right_eigenvectors_have_k_minus_1_sign_changes": all(
            bool(r["right_matches_expected"]) for r in rows
        ),
        "all_24_left_eigenvectors_have_k_minus_1_sign_changes": all(
            bool(r["left_matches_expected"]) for r in rows
        ),
        "minimum_numpy_right_overlap": min(float(r["numpy_right_overlap"]) for r in dense_rows),
        "minimum_scipy_right_overlap": min(float(r["scipy_right_overlap"]) for r in dense_rows),
        "minimum_scipy_left_overlap": min(float(r["scipy_left_overlap"]) for r in dense_rows),
        "dense_numpy_right_sign_pattern_failure_modes": [
            r["mode"] for r in dense_rows if not bool(r["numpy_right_matches_expected"])
        ],
        "dense_scipy_right_sign_pattern_failure_modes": [
            r["mode"] for r in dense_rows if not bool(r["scipy_right_matches_expected"])
        ],
        "dense_scipy_left_sign_pattern_failure_modes": [
            r["mode"] for r in dense_rows if not bool(r["scipy_left_matches_expected"])
        ],
        "smallest_relative_component_in_right_reference": mp.nstr(
            min(mp.mpf(r["right_min_component_over_max"]) for r in rows), 22
        ),
        "smallest_relative_component_in_left_reference": mp.nstr(
            min(mp.mpf(r["left_min_component_over_max"]) for r in rows), 22
        ),
    }
    (RESULTS_DIR / "eigenvector_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    _plot_sign_variation(sign_frame, dense_frame)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
