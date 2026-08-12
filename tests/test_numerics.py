from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tn_floquet import (  # noqa: E402
    build_dense_matrix,
    cauchon_eigenvalues,
    dense_eigenvalues,
    high_precision_dense_reference,
    high_precision_structured_reference,
    make_graded_parameters,
    make_microreactor_protocol,
    parameters_from_tntool_bd,
    pulse_schedule,
    tntool_bd_matrix,
    tntool_expand_bd,
)


def test_moderate_case_matches_dense() -> None:
    params = make_graded_parameters(
        n=10,
        retention_span_decades=6,
        coupling_strength=0.08,
        asymmetry=1.5,
        target_rho=0.82,
    )
    structured = cauchon_eigenvalues(params)
    dense = dense_eigenvalues(params)
    assert np.max(np.abs(dense.imag)) < 1e-12
    relative = np.abs(dense.real - structured) / structured
    assert np.max(relative) < 1e-10
    assert np.all(structured > 0)


def test_high_dynamic_range_matches_independent_mp_dense() -> None:
    params = make_graded_parameters(
        n=18,
        retention_span_decades=32,
        coupling_strength=0.08,
        asymmetry=1.5,
        target_rho=0.85,
    )
    structured = cauchon_eigenvalues(params)
    reference = high_precision_dense_reference(params, decimal_digits=90)
    relative = np.abs(structured - reference) / reference
    assert np.max(relative) < 1e-12
    assert np.isclose(structured[0], 0.85, rtol=1e-13, atol=1e-15)


def test_two_high_precision_routes_agree() -> None:
    params = make_graded_parameters(
        n=12,
        retention_span_decades=18,
        coupling_strength=0.11,
        asymmetry=1.7,
        target_rho=0.78,
    )
    dense_mp = high_precision_dense_reference(params, decimal_digits=80)
    structured_mp = high_precision_structured_reference(params, decimal_digits=80)
    relative = np.abs(dense_mp - structured_mp) / dense_mp
    assert np.max(relative) < 5e-15


def test_chronological_schedule_reconstructs_monodromy() -> None:
    params = make_graded_parameters(
        n=7,
        retention_span_decades=5,
        coupling_strength=0.09,
        asymmetry=1.4,
        target_rho=0.80,
    )
    propagator = np.eye(params.n)
    for record in pulse_schedule(params):
        direction = record["direction"]
        if direction == "decay":
            factor = np.diag(params.diagonal)
        else:
            factor = np.eye(params.n)
            source = int(record["source_compartment"]) - 1
            target = int(record["target_compartment"]) - 1
            factor[target, source] = float(record["parameter"])
        propagator = factor @ propagator

    expected = build_dense_matrix(params)
    assert np.allclose(propagator, expected, rtol=2e-14, atol=2e-15)


def test_determinant_matches_diagonal_product() -> None:
    params = make_graded_parameters(
        n=9,
        retention_span_decades=8,
        coupling_strength=0.07,
        asymmetry=1.3,
        target_rho=0.84,
    )
    eigenvalues = cauchon_eigenvalues(params)
    assert np.isclose(
        np.prod(eigenvalues),
        np.prod(params.diagonal),
        rtol=5e-13,
        atol=0.0,
    )


def test_tntool_bd_roundtrip_reconstructs_matrix() -> None:
    params = make_graded_parameters(
        n=11,
        retention_span_decades=7,
        coupling_strength=0.07,
        asymmetry=1.6,
        target_rho=0.81,
    )
    bd = tntool_bd_matrix(params)
    recovered = parameters_from_tntool_bd(
        bd,
        target_rho=params.target_rho,
        retention_span_decades=params.retention_span_decades,
        coupling_strength=params.coupling_strength,
        asymmetry=params.asymmetry,
    )
    assert np.array_equal(np.diag(bd), params.diagonal)
    assert np.allclose(
        build_dense_matrix(recovered),
        build_dense_matrix(params),
        rtol=0.0,
        atol=0.0,
    )

    # Independent check against Koev's TNExpand convention.  This catches the
    # within-sweep reversal required for the upper bidiagonal factors.
    expanded = tntool_expand_bd(bd)
    expected = build_dense_matrix(params)
    assert np.linalg.norm(expanded - expected) / np.linalg.norm(expected) < 5e-15
    assert bd[0, -2] == params.upper[2][2]
    assert bd[1, -1] == params.upper[2][1]


def test_dimensional_microreactor_protocol_is_stable_and_plausible() -> None:
    protocol = make_microreactor_protocol()
    params = protocol.parameters
    values = cauchon_eigenvalues(params)
    pulse_rates = []
    for family in (params.lower, params.upper):
        for row in family[1:]:
            assert row is not None
            pulse_rates.extend(value / protocol.pulse_duration_s for value in row.values())
    decay_rates = -np.log(params.diagonal) / protocol.decay_dwell_s
    assert np.isclose(values[0], 0.78, rtol=2e-13)
    assert np.isclose(protocol.cycle_duration_s, 3.4)
    assert min(pulse_rates) >= 1.5
    assert max(pulse_rates) <= 3.3
    assert min(decay_rates) > 0.0
    assert max(decay_rates) < 12.1


def test_high_precision_eigenvector_sign_variation_outputs() -> None:
    import pandas as pd

    diagnostics = pd.read_csv(ROOT / "data" / "eigenvectors" / "sign_variation_diagnostics.csv")
    dense = pd.read_csv(ROOT / "data" / "eigenvectors" / "dense_vs_high_precision_eigenvectors.csv")

    expected = np.arange(len(diagnostics))
    assert len(diagnostics) == 24
    assert np.array_equal(diagnostics["expected_sign_changes"].to_numpy(), expected)
    assert np.array_equal(diagnostics["right_sign_changes"].to_numpy(), expected)
    assert np.array_equal(diagnostics["left_sign_changes"].to_numpy(), expected)
    assert diagnostics["right_matches_expected"].all()
    assert diagnostics["left_matches_expected"].all()

    numpy_failures = dense.loc[~dense["numpy_right_matches_expected"], "mode"].tolist()
    scipy_right_failures = dense.loc[~dense["scipy_right_matches_expected"], "mode"].tolist()
    scipy_left_failures = dense.loc[~dense["scipy_left_matches_expected"], "mode"].tolist()
    assert numpy_failures == [23, 24]
    assert scipy_right_failures == [23, 24]
    assert scipy_left_failures == [23, 24]
