"""Factor-native Floquet spectra for periodically gated positive chains.

The routines implement a new switched-system formulation whose physical pulse
parameters are already the canonical bidiagonal data required by accurate
totally nonnegative eigensolvers.  The Cauchon route of Rasheed, Adm, and
Garloff (Electronic Journal of Linear Algebra 42, 2026) is one solver used to
extract the spectrum; a TNTool-compatible bidiagonal data export is also
provided for direct comparison with Koev's TNEigenValues implementation.

The monodromy matrix is represented by

    M = L^(1) ... L^(n-1) D U^(n-1) ... U^(1),

where all bidiagonal parameters and diagonal entries are nonnegative.  The
physical substeps are exact exponentials of nilpotent nearest-neighbour pulse
generators and a diagonal decay dwell.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import ctypes
import ctypes.util
import glob
import os
import warnings

import mpmath as mp
import numpy as np


NestedParameters = List[Dict[int, float] | None]


@dataclass(frozen=True)
class FloquetParameters:
    """Canonical bidiagonal parameters for one positive switching cycle."""

    lower: NestedParameters
    upper: NestedParameters
    diagonal: np.ndarray
    target_rho: float
    retention_span_decades: float
    coupling_strength: float
    asymmetry: float

    @property
    def n(self) -> int:
        return int(self.diagonal.size)


def _elementary_lower(n: int, index_1based: int, value: float) -> np.ndarray:
    out = np.eye(n, dtype=float)
    out[index_1based - 1, index_1based - 2] = value
    return out


def _elementary_upper(n: int, index_1based: int, value: float) -> np.ndarray:
    out = np.eye(n, dtype=float)
    out[index_1based - 2, index_1based - 1] = value
    return out


def build_dense_matrix(params: FloquetParameters) -> np.ndarray:
    """Form the dense monodromy matrix in ordinary floating-point arithmetic.

    This routine is useful for visualization and for the unstructured baseline.
    Accurate small eigenvalues should be computed from the factors instead.
    """
    lower, upper, diagonal = params.lower, params.upper, params.diagonal
    n = params.n
    matrix = np.eye(n, dtype=float)

    for r in range(1, n):
        assert lower[r] is not None
        for i in range(1, r + 1):
            index = n - r + i
            matrix = matrix @ _elementary_lower(n, index, lower[r][i])

    matrix = matrix @ np.diag(diagonal)

    for r in range(n - 1, 0, -1):
        assert upper[r] is not None
        for i in range(1, r + 1):
            index = n - i + 1
            matrix = matrix @ _elementary_upper(n, index, upper[r][i])

    return matrix


def make_graded_parameters(
    n: int = 24,
    retention_span_decades: float = 40.0,
    coupling_strength: float = 0.08,
    asymmetry: float = 1.5,
    target_rho: float = 0.85,
) -> FloquetParameters:
    """Construct a smooth, physically interpretable graded pulse protocol.

    The unscaled diagonal retentions span ``retention_span_decades``.  A common
    scale is then applied so that the largest Floquet multiplier equals
    ``target_rho``.  Scaling the whole diagonal scales every eigenvalue by the
    same factor, so this stability adjustment is exact.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if not (0.0 < target_rho < 1.0):
        raise ValueError("target_rho must lie in (0, 1)")
    if coupling_strength <= 0.0 or asymmetry <= 0.0:
        raise ValueError("coupling_strength and asymmetry must be positive")

    lower: NestedParameters = [None]
    upper: NestedParameters = [None]

    for r in range(1, n):
        lower_r: Dict[int, float] = {}
        upper_r: Dict[int, float] = {}
        sweep_fraction = (r - 1) / max(1, n - 2)
        for i in range(1, r + 1):
            local_fraction = (i - 1) / max(1, r - 1) if r > 1 else 0.0
            lower_r[i] = float(
                coupling_strength
                * (1.0 + 0.35 * sweep_fraction)
                * (1.0 + 0.15 * np.sin(np.pi * local_fraction))
            )
            upper_r[i] = float(
                coupling_strength
                * asymmetry
                * (1.0 + 0.20 * (1.0 - sweep_fraction))
                * (1.0 + 0.10 * np.cos(np.pi * local_fraction))
            )
        lower.append(lower_r)
        upper.append(upper_r)

    diagonal_relative = 10.0 ** (
        -retention_span_decades * np.linspace(0.0, 1.0, n)
    )
    trial = FloquetParameters(
        lower=lower,
        upper=upper,
        diagonal=diagonal_relative,
        target_rho=target_rho,
        retention_span_decades=retention_span_decades,
        coupling_strength=coupling_strength,
        asymmetry=asymmetry,
    )
    # Determine the Perron root from the factor-native structured solver.
    # This keeps parameter generation independent of dense matrix formation.
    rho_unscaled = float(cauchon_eigenvalues(trial)[0])
    scale = target_rho / rho_unscaled
    diagonal = scale * diagonal_relative
    if np.max(diagonal) >= 1.0:
        raise ValueError(
            "The requested parameters require a retention factor >= 1. "
            "Decrease target_rho or coupling_strength."
        )

    return FloquetParameters(
        lower=lower,
        upper=upper,
        diagonal=np.asarray(diagonal, dtype=float),
        target_rho=target_rho,
        retention_span_decades=retention_span_decades,
        coupling_strength=coupling_strength,
        asymmetry=asymmetry,
    )


def rescale_upper(params: FloquetParameters, factor: float) -> FloquetParameters:
    """Multiply every upper-sweep pulse strength by a positive factor."""
    if factor <= 0.0:
        raise ValueError("factor must be positive")
    upper: NestedParameters = [None]
    for r in range(1, params.n):
        assert params.upper[r] is not None
        upper.append({i: factor * params.upper[r][i] for i in params.upper[r]})
    return FloquetParameters(
        lower=params.lower,
        upper=upper,
        diagonal=params.diagonal.copy(),
        target_rho=params.target_rho,
        retention_span_decades=params.retention_span_decades,
        coupling_strength=params.coupling_strength,
        asymmetry=params.asymmetry * factor,
    )


def cauchon_matrix_from_factors(params: FloquetParameters) -> np.ndarray:
    """Algorithm 3.4: compute the Cauchon matrix from bidiagonal parameters."""
    n = params.n
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        out[i, i] = params.diagonal[n - 1 - i]

    # Descending r is essential because the recurrences use entries farther
    # from the diagonal that must already have been generated.
    for r in range(n - 1, 0, -1):
        assert params.lower[r] is not None and params.upper[r] is not None
        for i in range(1, r + 1):
            row = n - i
            col = r - i
            out[row, col] = params.lower[r][i] * out[row, col + 1]

            row_u = i - 1
            col_u = n - r + i - 1
            out[row_u, col_u] = params.upper[r][i] * out[row_u + 1, col_u]
    return out


def _add_to_next_row(
    matrix: np.ndarray, x: float, y: float, row_0based: int
) -> np.ndarray:
    """Algorithm 4.1 / Appendix B translated to zero-based NumPy indexing."""
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError("matrix must be square")
    if not (0 <= row_0based < n - 1):
        raise ValueError("row_0based must identify a row with a successor")
    if x <= 0.0 or y <= 0.0:
        raise ValueError("x and y must be positive")

    i = row_0based
    out = matrix.copy()
    out[i + 1, n - 1] = x * matrix[i, n - 1] + y * matrix[i + 1, n - 1]
    out[i, n - 1] = matrix[i, n - 1] / y

    for j in range(n - 2, -1, -1):
        if matrix[i + 1, j + 1] != 0.0:
            z = out[i + 1, j + 1] / matrix[i + 1, j + 1]
            out[i + 1, j] = x * matrix[i, j] + matrix[i + 1, j] * z
            out[i, j] = matrix[i, j] / z
        elif matrix[i + 1, j] == 0.0:
            out[i + 1, j] = x * matrix[i, j]
            out[i, j] = 0.0
        else:
            h = j + 1
            while h < n - 1 and matrix[i + 1, h] == 0.0:
                h += 1
            if h == n - 1 and matrix[i + 1, h] == 0.0:
                out[i + 1, j] = x * matrix[i, j] + y * matrix[i + 1, j]
                out[i, j] = matrix[i, j] / y
            else:
                z = out[i + 1, h] / matrix[i + 1, h]
                out[i + 1, j] = x * matrix[i, j] + matrix[i + 1, j] * z
                out[i, j] = matrix[i, j] / z
    return out


def reduce_cauchon_to_bidiagonal(
    params: FloquetParameters,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Algorithms 4.1 and 5.1 up to the final bidiagonal singular-value step.

    Returns
    -------
    diagonal, superdiagonal, reduced_cauchon_matrix
        The first two arrays define the upper bidiagonal Cholesky factor whose
        squared singular values are the eigenvalues of the monodromy matrix.
    """
    reduced = cauchon_matrix_from_factors(params)
    n = params.n

    for i in range(n - 1, 1, -1):
        for k in range(0, i - 1):
            if reduced[i, k + 1] != 0.0:
                x = reduced[i, k] / reduced[i, k + 1]
                reduced[i, k] = 0.0
                reduced = _add_to_next_row(reduced, x, 1.0, k)

            if reduced[k + 1, i] != 0.0:
                x = reduced[k, i] / reduced[k + 1, i]
                reduced[k, i] = 0.0
                reduced = _add_to_next_row(reduced.T, x, 1.0, k).T

    diagonal = np.sqrt(np.diag(reduced)[::-1])
    superdiagonal = np.sqrt(
        np.asarray(
            [
                reduced[n - i - 2, n - i - 1]
                * reduced[n - i - 1, n - i - 2]
                / reduced[n - i - 1, n - i - 1]
                for i in range(n - 1)
            ],
            dtype=float,
        )
    )
    return diagonal, superdiagonal, reduced


def _candidate_lapack_libraries() -> List[str]:
    candidates: List[str] = []
    numpy_libs = os.path.abspath(os.path.join(os.path.dirname(np.__file__), "..", "numpy.libs"))
    candidates.extend(glob.glob(os.path.join(numpy_libs, "*openblas*")))
    candidates.extend(glob.glob(os.path.join(numpy_libs, "*lapack*")))
    try:
        import scipy  # type: ignore

        scipy_libs = os.path.abspath(
            os.path.join(os.path.dirname(scipy.__file__), "..", "scipy.libs")
        )
        candidates.extend(glob.glob(os.path.join(scipy_libs, "*openblas*")))
        candidates.extend(glob.glob(os.path.join(scipy_libs, "*lapack*")))
    except Exception:
        pass
    for name in ("lapack", "openblas", "blas"):
        found = ctypes.util.find_library(name)
        if found:
            candidates.append(found)
    unique: List[str] = []
    for item in candidates:
        if item not in unique:
            unique.append(item)
    return unique


def bidiagonal_singular_values_dlasq1(
    diagonal: Sequence[float], superdiagonal: Sequence[float]
) -> np.ndarray:
    """Compute bidiagonal singular values with LAPACK DLASQ1 (dqds).

    A small ctypes bridge is used because SciPy does not expose DLASQ1 on all
    builds.  Google Colab normally provides ``dlasq1_`` in the system LAPACK.
    If DLASQ1 is unavailable, the routine falls back to a dense SVD and warns;
    the fallback does not carry the same relative-accuracy guarantee.
    """
    d = np.ascontiguousarray(np.asarray(diagonal, dtype=np.float64)).copy()
    e = np.ascontiguousarray(np.asarray(superdiagonal, dtype=np.float64)).copy()
    n = int(d.size)
    if e.size != max(0, n - 1):
        raise ValueError("superdiagonal must have length n-1")
    if n == 0:
        return d
    if n == 1:
        return np.abs(d)

    symbol_candidates = (
        ("scipy_dlasq1_64_", ctypes.c_longlong),
        ("dlasq1_", ctypes.c_int),
        ("dlasq1", ctypes.c_int),
        ("scipy_dlasq1_", ctypes.c_int),
    )

    for library_path in _candidate_lapack_libraries():
        try:
            library = ctypes.CDLL(library_path)
        except OSError:
            continue
        for symbol, integer_type in symbol_candidates:
            if not hasattr(library, symbol):
                continue
            routine = getattr(library, symbol)
            work = np.empty(4 * n, dtype=np.float64)
            n_c = integer_type(n)
            info = integer_type(0)
            routine.argtypes = [
                ctypes.POINTER(integer_type),
                np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
                ctypes.POINTER(integer_type),
            ]
            routine.restype = None
            routine(ctypes.byref(n_c), d, e, work, ctypes.byref(info))
            if info.value == 0:
                return d

    warnings.warn(
        "LAPACK DLASQ1 was not found; falling back to a dense SVD. "
        "The smallest singular values may lose relative accuracy.",
        RuntimeWarning,
        stacklevel=2,
    )
    bidiagonal = np.diag(np.asarray(diagonal, dtype=float))
    bidiagonal += np.diag(np.asarray(superdiagonal, dtype=float), k=1)
    return np.linalg.svd(bidiagonal, compute_uv=False)


def cauchon_eigenvalues(params: FloquetParameters) -> np.ndarray:
    """Compute all Floquet multipliers from the TN bidiagonal factors."""
    diagonal, superdiagonal, _ = reduce_cauchon_to_bidiagonal(params)
    singular_values = bidiagonal_singular_values_dlasq1(diagonal, superdiagonal)
    eigenvalues = np.asarray(singular_values, dtype=float) ** 2
    return np.sort(eigenvalues)[::-1]


def dense_eigenvalues(params: FloquetParameters) -> np.ndarray:
    """Unstructured dense eigensolver baseline, sorted by real part."""
    values = np.linalg.eigvals(build_dense_matrix(params))
    order = np.argsort(values.real)[::-1]
    return values[order]


def _mp_add_to_next_row(
    matrix: List[List[mp.mpf]], x: mp.mpf, y: mp.mpf, row: int
) -> List[List[mp.mpf]]:
    n = len(matrix)
    out = [line[:] for line in matrix]
    out[row + 1][n - 1] = x * matrix[row][n - 1] + y * matrix[row + 1][n - 1]
    out[row][n - 1] = matrix[row][n - 1] / y
    for j in range(n - 2, -1, -1):
        if matrix[row + 1][j + 1] != 0:
            z = out[row + 1][j + 1] / matrix[row + 1][j + 1]
            out[row + 1][j] = x * matrix[row][j] + matrix[row + 1][j] * z
            out[row][j] = matrix[row][j] / z
        elif matrix[row + 1][j] == 0:
            out[row + 1][j] = x * matrix[row][j]
            out[row][j] = mp.mpf("0")
        else:
            h = j + 1
            while h < n - 1 and matrix[row + 1][h] == 0:
                h += 1
            if h == n - 1 and matrix[row + 1][h] == 0:
                out[row + 1][j] = x * matrix[row][j] + y * matrix[row + 1][j]
                out[row][j] = matrix[row][j] / y
            else:
                z = out[row + 1][h] / matrix[row + 1][h]
                out[row + 1][j] = x * matrix[row][j] + matrix[row + 1][j] * z
                out[row][j] = matrix[row][j] / z
    return out


def high_precision_structured_reference(
    params: FloquetParameters, decimal_digits: int = 100
) -> np.ndarray:
    """Arbitrary-precision cross-check using the subtraction-free reduction.

    This calculation intentionally mirrors the structured algorithm.  It is a
    useful implementation cross-check, but it is not the independent dense
    reference used for the publication error curves.
    """
    if decimal_digits < 40:
        raise ValueError("Use at least 40 decimal digits for a reference calculation")
    old_dps = mp.mp.dps
    mp.mp.dps = decimal_digits
    try:
        n = params.n
        reduced: List[List[mp.mpf]] = [
            [mp.mpf("0") for _ in range(n)] for _ in range(n)
        ]
        for i in range(n):
            reduced[i][i] = mp.mpf(str(params.diagonal[n - 1 - i]))
        for r in range(n - 1, 0, -1):
            assert params.lower[r] is not None and params.upper[r] is not None
            for i in range(1, r + 1):
                row = n - i
                col = r - i
                reduced[row][col] = (
                    mp.mpf(str(params.lower[r][i])) * reduced[row][col + 1]
                )
                row_u = i - 1
                col_u = n - r + i - 1
                reduced[row_u][col_u] = (
                    mp.mpf(str(params.upper[r][i])) * reduced[row_u + 1][col_u]
                )

        one = mp.mpf("1")
        for i in range(n - 1, 1, -1):
            for k in range(0, i - 1):
                if reduced[i][k + 1] != 0:
                    x = reduced[i][k] / reduced[i][k + 1]
                    reduced[i][k] = mp.mpf("0")
                    reduced = _mp_add_to_next_row(reduced, x, one, k)
                if reduced[k + 1][i] != 0:
                    x = reduced[k][i] / reduced[k + 1][i]
                    reduced[k][i] = mp.mpf("0")
                    transposed = [list(line) for line in zip(*reduced)]
                    transposed = _mp_add_to_next_row(transposed, x, one, k)
                    reduced = [list(line) for line in zip(*transposed)]

        diagonal = [mp.sqrt(reduced[n - i - 1][n - i - 1]) for i in range(n)]
        superdiagonal = [
            mp.sqrt(
                reduced[n - i - 2][n - i - 1]
                * reduced[n - i - 1][n - i - 2]
                / reduced[n - i - 1][n - i - 1]
            )
            for i in range(n - 1)
        ]

        normal_matrix = mp.matrix(n)
        for i in range(n):
            normal_matrix[i, i] = diagonal[i] ** 2
            if i > 0:
                normal_matrix[i, i] += superdiagonal[i - 1] ** 2
            if i < n - 1:
                offdiag = diagonal[i] * superdiagonal[i]
                normal_matrix[i, i + 1] = offdiag
                normal_matrix[i + 1, i] = offdiag

        values = mp.eigsy(normal_matrix, eigvals_only=True)
        ordered = sorted((mp.mpf(value) for value in values), reverse=True)
        return np.asarray([float(value) for value in ordered], dtype=float)
    finally:
        mp.mp.dps = old_dps


def high_precision_dense_eigenvalues_mp(
    params: FloquetParameters, decimal_digits: int = 100
) -> List[mp.mpf]:
    """Compute an independent arbitrary-precision dense eigenvalue reference.

    The dense monodromy matrix is assembled directly in multiprecision from the
    physical bidiagonal factors.  Elementary right multiplications are applied
    as column updates, avoiding binary64 matrix formation.  The general dense
    eigenproblem is then solved with :func:`mpmath.eig`.
    """
    if decimal_digits < 40:
        raise ValueError("Use at least 40 decimal digits for a reference calculation")

    with mp.workdps(decimal_digits):
        n = params.n
        matrix = mp.eye(n)

        # Right multiplication by L_j(a)=I+aE_{j,j-1} adds a times column j
        # to column j-1.
        for r in range(1, n):
            assert params.lower[r] is not None
            for i in range(1, r + 1):
                index = n - r + i  # one-based elementary factor index
                value = mp.mpf(str(params.lower[r][i]))
                source = index - 1
                target = index - 2
                for row in range(n):
                    matrix[row, target] += value * matrix[row, source]

        # Right multiplication by D scales each column.
        for column, value_float in enumerate(params.diagonal):
            value = mp.mpf(str(value_float))
            for row in range(n):
                matrix[row, column] *= value

        # Right multiplication by U_j(a)=I+aE_{j-1,j} adds a times column j-1
        # to column j.
        for r in range(n - 1, 0, -1):
            assert params.upper[r] is not None
            for i in range(1, r + 1):
                index = n - i + 1
                value = mp.mpf(str(params.upper[r][i]))
                source = index - 2
                target = index - 1
                for row in range(n):
                    matrix[row, target] += value * matrix[row, source]

        values = mp.eig(matrix, left=False, right=False)
        imag_tolerance = mp.power(10, -(decimal_digits // 2))
        real_values: List[mp.mpf] = []
        for value in values:
            if abs(mp.im(value)) > imag_tolerance * max(mp.mpf("1"), abs(value)):
                raise ArithmeticError(
                    "The multiprecision dense reference returned a non-negligible "
                    f"imaginary part: {mp.nstr(value, 20)}"
                )
            real_values.append(mp.re(value))
        return sorted(real_values, reverse=True)


def high_precision_dense_reference(
    params: FloquetParameters, decimal_digits: int = 100
) -> np.ndarray:
    """Return the independent multiprecision dense reference as binary64 values."""
    values = high_precision_dense_eigenvalues_mp(params, decimal_digits=decimal_digits)
    return np.asarray([float(value) for value in values], dtype=float)


def high_precision_reference(
    params: FloquetParameters, decimal_digits: int = 100
) -> np.ndarray:
    """Backward-compatible alias for the independent dense reference."""
    return high_precision_dense_reference(params, decimal_digits=decimal_digits)


def pulse_schedule(params: FloquetParameters) -> List[dict]:
    """Return the chronological substeps for the column-vector convention.

    For ``x_{k+1}=M x_k`` and
    ``M=L^(1)...L^(n-1) D U^(n-1)...U^(1)``, physical substeps are applied
    from right to left: ``U^(1),...,U^(n-1),D,L^(n-1),...,L^(1)``.  The
    returned schedule follows that true chronological order.
    """
    records: List[dict] = []
    step = 1
    n = params.n

    # Rightmost upper factors act first.  Within each U^(r), factors also act
    # right-to-left on a column state.
    for r in range(1, n):
        assert params.upper[r] is not None
        for i in range(r, 0, -1):
            source = n - i + 1
            target = source - 1
            records.append(
                {
                    "step": step,
                    "phase": "upstream feedback sweep",
                    "direction": "upper",
                    "source_compartment": source,
                    "target_compartment": target,
                    "active_link": target,
                    "parameter": params.upper[r][i],
                    "factor_r": r,
                    "factor_i": i,
                    "chronological_factor": f"U_{source}",
                }
            )
            step += 1

    records.append(
        {
            "step": step,
            "phase": "diagonal decay dwell",
            "direction": "decay",
            "source_compartment": np.nan,
            "target_compartment": np.nan,
            "active_link": np.nan,
            "parameter": np.nan,
            "factor_r": np.nan,
            "factor_i": np.nan,
            "chronological_factor": "D",
        }
    )
    step += 1

    # Left factors act after the dwell, again in right-to-left order.
    for r in range(n - 1, 0, -1):
        assert params.lower[r] is not None
        for i in range(r, 0, -1):
            target = n - r + i
            source = target - 1
            records.append(
                {
                    "step": step,
                    "phase": "downstream pulse sweep",
                    "direction": "lower",
                    "source_compartment": source,
                    "target_compartment": target,
                    "active_link": source,
                    "parameter": params.lower[r][i],
                    "factor_r": r,
                    "factor_i": i,
                    "chronological_factor": f"L_{target}",
                }
            )
            step += 1
    return records


def factor_table(params: FloquetParameters) -> List[dict]:
    """Flatten all factor parameters for CSV export."""
    rows: List[dict] = []
    for r in range(1, params.n):
        assert params.lower[r] is not None and params.upper[r] is not None
        for i in range(1, r + 1):
            rows.append(
                {
                    "family": "lower",
                    "r": r,
                    "i": i,
                    "parameter": params.lower[r][i],
                }
            )
            rows.append(
                {
                    "family": "upper",
                    "r": r,
                    "i": i,
                    "parameter": params.upper[r][i],
                }
            )
    for i, value in enumerate(params.diagonal, start=1):
        rows.append(
            {
                "family": "diagonal retention",
                "r": np.nan,
                "i": i,
                "parameter": value,
            }
        )
    return rows



@dataclass(frozen=True)
class PhysicalProtocol:
    """Dimensional interpretation of a factor-native switching protocol.

    The rates are illustrative design values for a source-clamped microfluidic
    catalytic chain, not fitted experimental parameters.  A short valve-open
    pulse implements ``I + rate * pulse_duration * E`` exactly under the
    source-clamped linearization, and the dwell implements independent decay.
    """

    parameters: FloquetParameters
    pulse_duration_s: float
    decay_dwell_s: float
    chamber_volume_uL: float
    nominal_cycle_label: str

    @property
    def cycle_duration_s(self) -> float:
        pulse_count = self.parameters.n * (self.parameters.n - 1)
        return pulse_count * self.pulse_duration_s + self.decay_dwell_s


def make_microreactor_protocol(
    n: int = 8,
    pulse_duration_s: float = 0.050,
    decay_dwell_s: float = 0.600,
    chamber_volume_uL: float = 5.0,
) -> PhysicalProtocol:
    """Return a dimensional, experimentally plausible illustrative protocol.

    Fifty-millisecond pulses are longer than the approximately 17 ms opening
    response reported for an electromagnetic microvalve.  The resulting
    effective source-clamped coupling rates are about 1.6--3.2 s^-1, while the
    independent dwell decay rates are about 0.45--12 s^-1 for the default
    three-decade retention gradient.
    """
    if pulse_duration_s <= 0.0 or decay_dwell_s <= 0.0:
        raise ValueError("pulse and dwell durations must be positive")
    if chamber_volume_uL <= 0.0:
        raise ValueError("chamber volume must be positive")
    params = make_graded_parameters(
        n=n,
        retention_span_decades=3.0,
        coupling_strength=0.08,
        asymmetry=1.5,
        target_rho=0.78,
    )
    return PhysicalProtocol(
        parameters=params,
        pulse_duration_s=float(pulse_duration_s),
        decay_dwell_s=float(decay_dwell_s),
        chamber_volume_uL=float(chamber_volume_uL),
        nominal_cycle_label="source-clamped catalytic microreactor chain",
    )


def dimensional_protocol_table(protocol: PhysicalProtocol) -> List[dict]:
    """Return dimensional pulse and dwell parameters for CSV export."""
    rows: List[dict] = []
    params = protocol.parameters
    for record in pulse_schedule(params):
        if record["direction"] == "decay":
            continue
        parameter = float(record["parameter"])
        rows.append(
            {
                "step": int(record["step"]),
                "phase": record["phase"],
                "direction": record["direction"],
                "source_compartment": int(record["source_compartment"]),
                "target_compartment": int(record["target_compartment"]),
                "pulse_duration_s": protocol.pulse_duration_s,
                "dimensionless_pulse_parameter": parameter,
                "effective_source_clamped_rate_per_s": parameter
                / protocol.pulse_duration_s,
                "chamber_volume_uL": protocol.chamber_volume_uL,
            }
        )
    return rows


def dimensional_decay_table(protocol: PhysicalProtocol) -> List[dict]:
    """Return compartment retentions and dimensional decay rates."""
    return [
        {
            "compartment": i,
            "decay_dwell_s": protocol.decay_dwell_s,
            "retention_factor": float(value),
            "decay_rate_per_s": float(-np.log(value) / protocol.decay_dwell_s),
            "chamber_volume_uL": protocol.chamber_volume_uL,
        }
        for i, value in enumerate(protocol.parameters.diagonal, start=1)
    ]


def tntool_bd_matrix(params: FloquetParameters) -> np.ndarray:
    """Pack the canonical factorization in Koev TNTool ``BD(A)`` format.

    The diagonal stores the positive pivots.  Below the diagonal, entry
    ``(n-r+i, i)`` stores ``l_i^(r)``; above the diagonal, entry
    ``(i, n-r+i)`` stores ``u_i^(r)``.  Indices in this docstring are one-based.
    The returned matrix can be passed directly to TNTool's ``TNEigenValues``.
    """
    n = params.n
    bd = np.zeros((n, n), dtype=float)
    np.fill_diagonal(bd, params.diagonal)
    for r in range(1, n):
        assert params.lower[r] is not None and params.upper[r] is not None
        for i in range(1, r + 1):
            row = n - r + i
            bd[row - 1, i - 1] = params.lower[r][i]
            bd[i - 1, row - 1] = params.upper[r][i]
    return bd


def parameters_from_tntool_bd(
    bd: np.ndarray,
    *,
    target_rho: float = float("nan"),
    retention_span_decades: float = float("nan"),
    coupling_strength: float = float("nan"),
    asymmetry: float = float("nan"),
) -> FloquetParameters:
    """Unpack a TNTool ``BD(A)`` matrix into this module's factor structure."""
    bd = np.asarray(bd, dtype=float)
    if bd.ndim != 2 or bd.shape[0] != bd.shape[1]:
        raise ValueError("bd must be a square matrix")
    n = bd.shape[0]
    diagonal = np.diag(bd).copy()
    if np.any(diagonal <= 0.0) or np.any(bd < 0.0):
        raise ValueError("TNTool data must be nonnegative with positive diagonal")
    lower: NestedParameters = [None]
    upper: NestedParameters = [None]
    for r in range(1, n):
        lower_r: Dict[int, float] = {}
        upper_r: Dict[int, float] = {}
        for i in range(1, r + 1):
            row = n - r + i
            lower_r[i] = float(bd[row - 1, i - 1])
            upper_r[i] = float(bd[i - 1, row - 1])
        lower.append(lower_r)
        upper.append(upper_r)
    return FloquetParameters(
        lower=lower,
        upper=upper,
        diagonal=diagonal,
        target_rho=target_rho,
        retention_span_decades=retention_span_decades,
        coupling_strength=coupling_strength,
        asymmetry=asymmetry,
    )

def validate_against_dense(params: FloquetParameters, tolerance: float = 1e-10) -> dict:
    """Diagnostic for moderate problems where the dense eigensolver is reliable."""
    structured = cauchon_eigenvalues(params)
    dense = dense_eigenvalues(params)
    dense_real = dense.real
    relative = np.abs(dense_real - structured) / structured
    return {
        "max_relative_difference": float(np.max(relative)),
        "all_dense_eigenvalues_real_to_tolerance": bool(
            np.max(np.abs(dense.imag)) <= tolerance
        ),
        "structured_positive": bool(np.all(structured > 0.0)),
        "spectral_radius": float(structured[0]),
        "smallest_multiplier": float(structured[-1]),
    }
