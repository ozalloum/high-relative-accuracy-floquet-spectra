# High-Relative-Accuracy Floquet Spectra

Minimal public reproducibility repository for the manuscript **"Factor-Native High-Relative-Accuracy Floquet Spectra of Periodically Gated Positive Transport--Reaction Chains."**

## Contents

- `src/tn_floquet.py` — core numerical implementation.
- `scripts/reproduce_all.py` — regenerates the numerical outputs and figures from the public Python implementation.
- `scripts/compute_eigenvectors_sign_variation.py` — high-precision eigenvector/sign-variation analysis.
- `tests/test_numerics.py` — regression tests.
- `data/koev_tntool_bd_benchmark.csv` — benchmark factor data packed in TNTool's `BD(A)` convention.
- `data/koev_tntool_reference.csv` — independent multiprecision reference spectrum.
- `results/koev_tntool_comparison.csv` — measured mode-by-mode TNTool comparison used for the independent validation.
- `requirements.txt` — Python dependencies.

## Reproduction

From the repository root:

    pip install -r requirements.txt
    python scripts/reproduce_all.py
    python -m pytest -q

The reproduction script creates the generated numerical tables and figures locally. They are intentionally not stored in this minimal repository.

A clean verification of this public tree reproduced the reported numerical summaries and then passed all 8 regression tests.

## TNTool validation

Koev's TNTool is an external dependency and is **not redistributed** in this repository. The file `data/koev_tntool_bd_benchmark.csv` contains the benchmark input packed in TNTool's `BD(A)` convention. The official TNTool algorithms are not modified.

The benchmark `BD(A)` file has SHA-256:

    cb046e3d10840b09a6b505977b57b83cbd2c70643f97854f2bdafa4020a8ed78

The stored `results/koev_tntool_comparison.csv` records the independent TNTool validation reported in the manuscript.

## Repository

https://github.com/ozalloum/high-relative-accuracy-floquet-spectra
