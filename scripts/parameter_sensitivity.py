"""Bounded parameter-sensitivity check for the eight-chamber example.

Reproduces the retained sensitivity table for the computational study.  Independent
uniform relative perturbations are applied to every effective lower-pulse,
upper-pulse, and dwell-decay rate.  A fixed seed makes the Monte Carlo sample
exactly reproducible.
"""
from pathlib import Path
import math
import numpy as np
import mpmath as mp

SEED = 20260813
N_SAMPLES = 1000
LEVELS = (1e-3, 1e-2, 5e-2)
N = 8
C = 0.08
ALPHA = 1.5
RETENTION_DECADES = 3.0
TARGET_RHO = 0.78
TAU = 0.60
OUT = Path(__file__).resolve().parents[1] / "data" / "parameter_sensitivity.csv"


def parameter_groups(n=N, c=C, alpha=ALPHA):
    lower, upper = {}, {}
    for r in range(1, n):
        lower[r], upper[r] = [], []
        for i in range(1, r + 1):
            lower[r].append(
                c * (1 + 0.35 * (r - 1) / (n - 2))
                * (1 + 0.15 * math.sin(math.pi * (i - 1) / max(1, r - 1)))
            )
            upper[r].append(
                c * alpha * (1 + 0.20 * (1 - (r - 1) / (n - 2)))
                * (1 + 0.10 * math.cos(math.pi * (i - 1) / max(1, r - 1)))
            )
    return lower, upper


def elementary_lower(n, j, a, dtype=float):
    A = np.eye(n, dtype=dtype)
    A[j - 1, j - 2] = a
    return A


def elementary_upper(n, j, a, dtype=float):
    A = np.eye(n, dtype=dtype)
    A[j - 2, j - 1] = a
    return A


def monodromy(lower, upper, d):
    n = len(d)
    M = np.eye(n)
    for r in range(1, n):
        G = np.eye(n)
        for i, a in enumerate(lower[r], start=1):
            j = n - r + i
            G = G @ elementary_lower(n, j, a)
        M = M @ G
    M = M @ np.diag(d)
    for r in range(n - 1, 0, -1):
        G = np.eye(n)
        for i, a in enumerate(upper[r], start=1):
            j = n - i + 1
            G = G @ elementary_upper(n, j, a)
        M = M @ G
    return M


def eigvals_sorted(M):
    vals = np.linalg.eigvals(M)
    if np.max(np.abs(vals.imag)) > 1e-10:
        raise RuntimeError("unexpected complex part")
    return np.sort(vals.real)[::-1]


def flatten(groups):
    return np.array([v for r in range(1, N) for v in groups[r]], dtype=float)


def unflatten(values):
    out, p = {}, 0
    for r in range(1, N):
        out[r] = list(values[p : p + r])
        p += r
    return out


def mp_reference(lower, upper, d, digits=80):
    mp.mp.dps = digits
    n = len(d)
    I = mp.eye(n)
    M = mp.eye(n)
    for r in range(1, n):
        G = mp.eye(n)
        for i, a in enumerate(lower[r], start=1):
            j = n - r + i
            E = mp.eye(n)
            E[j - 1, j - 2] = mp.mpf(str(a))
            G = G * E
        M = M * G
    D = mp.diag([mp.mpf(str(x)) for x in d])
    M = M * D
    for r in range(n - 1, 0, -1):
        G = mp.eye(n)
        for i, a in enumerate(upper[r], start=1):
            j = n - i + 1
            E = mp.eye(n)
            E[j - 2, j - 1] = mp.mpf(str(a))
            G = G * E
        M = M * G
    vals = mp.eig(M, left=False, right=False)
    vals = sorted((mp.re(v) for v in vals), reverse=True)
    return vals


def main():
    lower, upper = parameter_groups()
    d_hat = 10.0 ** (-RETENTION_DECADES * np.arange(N) / (N - 1))
    rho_hat = eigvals_sorted(monodromy(lower, upper, d_hat))[0]
    d = (TARGET_RHO / rho_hat) * d_hat
    gamma = -np.log(d) / TAU
    lam0 = eigvals_sorted(monodromy(lower, upper, d))

    ref = mp_reference(lower, upper, d, digits=80)
    ref_float = np.array([float(x) for x in ref])
    baseline_rel = np.max(np.abs(lam0 - ref_float) / ref_float)

    lf, uf = flatten(lower), flatten(upper)
    rng = np.random.default_rng(SEED)
    rows = []
    for level in LEVELS:
        max_all, dominant = [], []
        for _ in range(N_SAMPLES):
            lp = lf * (1.0 + rng.uniform(-level, level, lf.size))
            up = uf * (1.0 + rng.uniform(-level, level, uf.size))
            gp = gamma * (1.0 + rng.uniform(-level, level, gamma.size))
            dp = np.exp(-gp * TAU)
            lamp = eigvals_sorted(monodromy(unflatten(lp), unflatten(up), dp))
            rel = np.abs(lamp - lam0) / lam0
            max_all.append(rel.max())
            dominant.append(rel[0])
        max_all = np.asarray(max_all)
        dominant = np.asarray(dominant)
        rows.append(
            (
                level,
                np.median(max_all),
                np.quantile(max_all, 0.95),
                max_all.max(),
                np.median(dominant),
                np.quantile(dominant, 0.95),
                dominant.max(),
            )
        )

    with OUT.open("w", encoding="utf-8") as f:
        f.write(
            "relative_parameter_bound,median_max_relative_spectral_change,"
            "p95_max_relative_spectral_change,maximum_max_relative_spectral_change,"
            "median_dominant_relative_change,p95_dominant_relative_change,"
            "maximum_dominant_relative_change\n"
        )
        for row in rows:
            f.write(",".join(f"{x:.16e}" for x in row) + "\n")

    print(f"80-digit baseline max relative difference: {baseline_rel:.16e}")
    print(f"decay-rate range: {gamma.min():.10f} to {gamma.max():.10f} s^-1")
    print(f"wrote {OUT}")
    for row in rows:
        print(", ".join(f"{x:.8e}" for x in row))


if __name__ == "__main__":
    main()
