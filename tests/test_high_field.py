"""Non-HFS electronic Paschen-Back regressions, independent of other fixes.

Reference energies use Eterm + muBcm*B*(mL + 2*mS), retaining the existing
approximation's ideal g factors. Orbital E1 references use Wigner 3j symbols
directly and conserve spin; they do not call AtomSpect.dipolestr.
"""
from pathlib import Path
import os
import sys

os.environ.setdefault("SYMPY_GROUND_TYPES", "python")
os.environ.setdefault("MPMATH_NOGMPY", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
from sympy.physics.wigner import wigner_3j
import AtomSpect


@pytest.mark.parametrize("L,S", [(1, 1), (2, .5)])
@pytest.mark.parametrize("field", [0., 1., -2.])
@pytest.mark.parametrize("term", [0., 123.])
@pytest.mark.parametrize("sort", [False, True])
def test_analytic_energies_and_coupled_basis(L, S, field, term, sort):
    j_values = np.arange(abs(L-S), L+S+.1)
    level = AtomSpect.stateprocess(L, S, np.arange(len(j_values))*100.+100., field, sortE=sort)
    energies, vectors = AtomSpect.Zeeman_func(level, field, Highfield=True, Eterm=term)
    expected = [term + AtomSpect.muBcm*field*(ml+2*ms)
                for ml in np.arange(-L, L+1) for ms in np.arange(-S, S+1)]
    np.testing.assert_allclose(energies, np.sort(expected), rtol=0, atol=1e-12)
    vectors = np.asarray(vectors)
    psi = np.asarray(level[0])
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(len(energies)), atol=1e-13)
    # Test the eigen-equation in the uncoupled basis. This catches returning
    # uncoupled vectors directly even if their energies happen to be correct.
    uncoupled = psi.T @ vectors
    row_energy = np.array([term + AtomSpect.muBcm*field*(s[3]+2*s[5]) for s in level[-1]])
    np.testing.assert_allclose(row_energy[:, None]*uncoupled, uncoupled*energies,
                               rtol=0, atol=1e-12)


def test_eigensolver_receives_a_diagonal_hermitian_matrix(monkeypatch):
    level = AtomSpect.stateprocess(1, 1, np.array([100., 200., 300.]), 1.)
    original = np.linalg.eigh
    captured = []

    def check(matrix):
        matrix = np.asarray(matrix)
        np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-14)
        np.testing.assert_array_equal(matrix, np.diag(np.diag(matrix)))
        captured.append(matrix.copy())
        return original(matrix)

    monkeypatch.setattr(np.linalg, "eigh", check)
    AtomSpect.Zeeman_func(level, 1., Highfield=True, Eterm=123.)
    assert len(captured) == 1
    np.testing.assert_allclose(np.trace(captured[0]), 9*123., atol=1e-12)


def test_missing_term_energy_has_a_clear_error():
    level = AtomSpect.stateprocess(1, 1, np.array([100., 200., 300.]), 1.)
    with pytest.raises(ValueError, match="require Eterm"):
        AtomSpect.Zeeman_func(level, 1., Highfield=True)


@pytest.mark.parametrize("sort", [False, True])
def test_high_field_fan_and_solver_agree(sort):
    fields = np.array([-1., 0., 1.])
    level = AtomSpect.stateprocess(1, 1, np.array([100., 200., 300.]), 1., sortE=sort)
    fan = np.sort(np.asarray(AtomSpect.FastZfan(level, fields, Eterm_=123.)[1]), axis=0)
    for i, field in enumerate(fields):
        energies = AtomSpect.Zeeman_func(level, field, Highfield=True, Eterm=123.)[0]
        np.testing.assert_allclose(energies, fan[:, i], rtol=0, atol=1e-12)


def line_totals(strengths, wavelengths):
    """Sum strengths over components that share a wavelength.

    High-field levels are highly degenerate. Within an equal-energy group the
    split of strength between individual columns depends on the chosen
    eigenbasis, so only the total at each wavelength is observable.
    """
    keys = np.round(np.asarray(wavelengths, dtype=float), 9)
    lines = np.unique(keys)
    return lines, np.array([np.sum(np.asarray(strengths)[keys == k]) for k in lines])


def high_transition(field=3., sort=False, geometry=None):
    # Synthetic term energies, in cm^-1: only the common term separation
    # matters in this approximation; these are not measured atomic constants.
    g = AtomSpect.stateprocess(1, 1, np.array([100., 200., 300.]), field, sortE=sort)
    e = AtomSpect.stateprocess(2, 1, np.array([20100., 20200., 20300.]), field, sortE=sort)
    zg = AtomSpect.Zeeman_func(g, field, Highfield=True, Eterm=0.)
    ze = AtomSpect.Zeeman_func(e, field, Highfield=True, Eterm=20000.)
    result = AtomSpect.Zeeman_signal(g, zg, e, ze, **(geometry or {}))
    # Locate each eigenvector in the independently specified uncoupled basis.
    ug = np.asarray(g[0]).T @ np.asarray(zg[1])
    ue = np.asarray(e[0]).T @ np.asarray(ze[1])
    np.testing.assert_allclose(np.max(np.abs(ug), axis=0), 1., atol=1e-13)
    np.testing.assert_allclose(np.max(np.abs(ue), axis=0), 1., atol=1e-13)
    gs = [g[-1][i] for i in np.argmax(np.abs(ug), axis=0)]
    es = [e[-1][i] for i in np.argmax(np.abs(ue), axis=0)]
    return g, zg, e, ze, result, gs, es


@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("geometry", [dict(Bangle=90), dict(Bangle=0),
    dict(Bangle=37, Filter=True, gamma=23, T_Alpha=.65, T_Beta=.35)])
def test_strengths_match_spin_conserving_orbital_e1(sort, geometry):
    _, _, _, _, result, gs, es = high_transition(sort=sort, geometry=geometry)
    expected = []
    forbidden = []
    for upper in es:
        for lower in gs:
            ml_e, ml_g = upper[3], lower[3]
            q = ml_e-ml_g
            allowed = upper[5] == lower[5] and abs(q) <= 1
            forbidden.append(not allowed)
            if allowed:
                angular = float(wigner_3j(2, 1, 1, -ml_e, q, ml_g))**2
                factor = AtomSpect.polarization(q, 0, geometry["Bangle"],
                    geometry.get("gamma", 0), Pol_Filter=geometry.get("Filter", False),
                    TAlpha=geometry.get("T_Alpha", 1), TBeta=geometry.get("T_Beta", 1))[0]
                expected.append(angular * factor**2)
            else:
                expected.append(0.)
    expected = np.asarray(expected)
    expected /= expected.sum()
    lines, got = line_totals(result[0], result[6])
    _, want = line_totals(expected, result[6])
    assert len(lines) > 1
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-14)
    # Lines made only of forbidden components must stay dark.
    _, allowed = line_totals(~np.asarray(forbidden), result[6])
    assert np.max(got[allowed == 0], initial=0) < 1e-20


@pytest.mark.parametrize("sort", [False, True])
def test_main_high_field_outputs_match_direct_calculation(sort):
    deck = dict(s_ground=1, s_excited=1, l_ground=1, l_excited=2,
                E_ground=np.array([100., 200., 300.]),
                E_excited=np.array([20100., 20200., 20300.]),
                Bmag=3., b_angle=90, EtermG=0., EtermE=20000.,
                DoHighSig=True, DoLowSig=True, sortE=sort, amu=4,
                Convfxn="GaussianInstrum", Temp=300, specstep=.005,
                spec_window=[498., 502.], fxnwindow=.1)
    result = AtomSpect.AtomSpect_Main(deck)
    direct = high_transition(sort=sort)[4]
    np.testing.assert_allclose(result["wave_vac_high"], direct[6], rtol=0, atol=1e-12)
    np.testing.assert_allclose(line_totals(result["signal_high"], direct[6])[1],
                               line_totals(direct[0], direct[6])[1], rtol=0, atol=1e-14)
    assert np.all(np.isfinite(result["SpecOutHigh"]))
    assert np.max(result["SpecOutHigh"][1]) > 0
    # Enabling the repaired approximation must leave the exact calculation
    # and independently requested low-field calculation unchanged.
    without_high = deck.copy()
    del without_high["DoHighSig"]
    reference = AtomSpect.AtomSpect_Main(without_high)
    for key in ["signal", "wave_air", "SpecOut", "signal_low", "SpecOutLow"]:
        np.testing.assert_array_equal(result[key], reference[key])
