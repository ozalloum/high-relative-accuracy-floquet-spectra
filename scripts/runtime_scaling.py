"""Prototype dimension-scaling check for the Cauchon--dqds route.

Requires NumPy and a LAPACK library exporting the Fortran symbol dlasq1_.
The script times only the structured spectrum routine after construction of
one benchmark case per dimension.  It is a reproducibility check, not a
hardware-portability benchmark.
"""
from pathlib import Path
import ctypes
import ctypes.util
import math
import platform
import time
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "data" / "runtime_scaling.csv"
SIZES = (8, 12, 16, 20, 24, 32, 40, 48)
REPEATS = 15


def cpu_model():
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def load_dlasq1():
    candidates = [ctypes.util.find_library("lapack"), "liblapack.so.3", "liblapack.so"]
    last = None
    for name in candidates:
        if not name:
            continue
        try:
            lib = ctypes.cdll.LoadLibrary(name)
            fn = getattr(lib, "dlasq1_")
            fn.argtypes = [
                ctypes.POINTER(ctypes.c_int),
                np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
                ctypes.POINTER(ctypes.c_int),
            ]
            return fn, name
        except (OSError, AttributeError) as exc:
            last = exc
    raise RuntimeError(f"could not load LAPACK dlasq1_: {last}")


_DLASQ1, LAPACK_LIBRARY = load_dlasq1()


def parameters(n, c=0.08, alpha=1.5):
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


def tilde_g_from_factors(lower, upper, d):
    n = len(d)
    B = np.zeros((n, n))
    B[np.arange(n), np.arange(n)] = d[::-1]
    for r in range(n - 1, 0, -1):
        for i in range(1, r + 1):
            B[n - i, r - i] = lower[r][i - 1] * B[n - i, r + 1 - i]
            B[i - 1, n - r + i - 1] = upper[r][i - 1] * B[i, n - r + i - 1]
    return B


def add_to_next_row_inplace(A, x, y, i):
    n = A.shape[0]
    a0, a1 = A[i, :].copy(), A[i + 1, :].copy()
    b0, b1 = np.empty(n), np.empty(n)
    b1[n - 1] = x * a0[n - 1] + y * a1[n - 1]
    b0[n - 1] = a0[n - 1] / y
    for j in range(n - 2, -1, -1):
        if a1[j + 1] != 0.0:
            z = b1[j + 1] / a1[j + 1]
            b1[j] = x * a0[j] + a1[j] * z
            b0[j] = a0[j] / z
        elif a1[j] == 0.0:
            b1[j] = x * a0[j]
            b0[j] = 0.0
        else:
            h = j + 1
            while h < n - 1 and a1[h] == 0.0:
                h += 1
            if h == n - 1 and a1[h] == 0.0:
                b1[j] = x * a0[j] + y * a1[j]
                b0[j] = a0[j] / y
            else:
                z = b1[h] / a1[h]
                b1[j] = x * a0[j] + a1[j] * z
                b0[j] = a0[j] / z
    A[i, :], A[i + 1, :] = b0, b1


def reduce_to_bidiagonal_data(B):
    B = B.copy()
    n = B.shape[0]
    for i1 in range(n, 2, -1):
        i = i1 - 1
        for k1 in range(1, i1 - 1):
            k = k1 - 1
            if B[i, k + 1] != 0.0:
                x = B[i, k] / B[i, k + 1]
                B[i, k] = 0.0
                add_to_next_row_inplace(B, x, 1.0, k)
            if B[k + 1, i] != 0.0:
                x = B[k, i] / B[k + 1, i]
                B[k, i] = 0.0
                add_to_next_row_inplace(B.T, x, 1.0, k)
    diag = np.sqrt(np.diag(B)[::-1])
    off = np.empty(n - 1)
    for i1 in range(1, n):
        off[i1 - 1] = math.sqrt(
            B[n - i1 - 1, n - i1] * B[n - i1, n - i1 - 1]
            / B[n - i1, n - i1]
        )
    return diag, off


def dlasq1(diag, off):
    n = len(diag)
    d = np.ascontiguousarray(diag).copy()
    e = np.zeros(n)
    e[:-1] = off
    work = np.empty(4 * n)
    nn, info = ctypes.c_int(n), ctypes.c_int(0)
    _DLASQ1(ctypes.byref(nn), d, e, work, ctypes.byref(info))
    if info.value:
        raise RuntimeError(f"DLASQ1 failed with INFO={info.value}")
    return d


def structured_eigs(lower, upper, d):
    diag, off = reduce_to_bidiagonal_data(tilde_g_from_factors(lower, upper, d))
    singular_values = dlasq1(diag, off)
    return np.sort(singular_values * singular_values)[::-1]


def case(n, decades=12):
    lower, upper = parameters(n)
    d = 10.0 ** (-decades * np.arange(n) / (n - 1))
    vals = structured_eigs(lower, upper, d)
    d *= 0.85 / vals[0]
    return lower, upper, d


def main():
    lower, upper, d = case(24, 40)
    vals = structured_eigs(lower, upper, d)
    expected_smallest = 2.5271419228293403e-42
    assert abs(vals[0] - 0.85) < 2e-15
    assert abs(vals[-1] - expected_smallest) / expected_smallest < 1e-13

    rows = []
    for n in SIZES:
        lower, upper, d = case(n, 12)
        structured_eigs(lower, upper, d)
        samples = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            structured_eigs(lower, upper, d)
            samples.append(time.perf_counter() - t0)
        rows.append((n, float(np.median(samples)), min(samples), max(samples)))

    x = np.log(np.array([r[0] for r in rows], dtype=float))
    y = np.log(np.array([r[1] for r in rows], dtype=float))
    slope_all = np.polyfit(x, y, 1)[0]
    mask = np.array([r[0] >= 16 for r in rows])
    slope_ge16 = np.polyfit(x[mask], y[mask], 1)[0]

    with OUT.open("w", encoding="utf-8") as f:
        f.write(f"# platform,{platform.platform()}\n")
        f.write(f"# python,{platform.python_version()}\n")
        f.write(f"# numpy,{np.__version__}\n")
        f.write(f"# cpu_model,{cpu_model()}\n")
        f.write(f"# repeats,{REPEATS}\n")
        f.write(f"# lapack_library,{LAPACK_LIBRARY}\n")
        f.write("n,median_seconds,min_seconds,max_seconds\n")
        for row in rows:
            f.write(f"{row[0]},{row[1]:.16e},{row[2]:.16e},{row[3]:.16e}\n")
        f.write(f"# loglog_slope_all,{slope_all:.8f}\n")
        f.write(f"# loglog_slope_n_ge_16,{slope_ge16:.8f}\n")

    print(f"LAPACK: {LAPACK_LIBRARY}")
    for row in rows:
        print(row[0], f"{row[1]:.9f}")
    print("log-log slope all:", f"{slope_all:.8f}")
    print("log-log slope n>=16:", f"{slope_ge16:.8f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
