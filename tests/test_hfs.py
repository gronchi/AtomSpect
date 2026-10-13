"""Regression tests for the hyperfine (HFS) Zeeman Hamiltonian.

The reference Hamiltonian is built independently of AtomSpect, from spin
matrices in the uncoupled |mL, mS, mI> basis:

    H = sum_J P_J [E_J + A_J I.J + B_J Q_J] P_J + muB*B*(gL*Lz + gS*Sz - gI*Iz)

where P_J projects onto the fine-structure level J and Q_J is the standard
electric-quadrupole operator. Only the physical constants (gS, gL, muBcm) are
taken from AtomSpect, so the tests check the matrix assembly, not the
constants.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import AtomSpect as A

HZ_TO_CM = 1 / (100 * A.c_light)


def spin_ops(j):
    m = np.arange(j, -j - 1, -1)
    jz = np.diag(m)
    jp = np.diag(np.sqrt(j * (j + 1) - m[1:] * (m[1:] + 1)), 1)
    jx = (jp + jp.T) / 2
    jy = (jp - jp.T) / 2j
    return jx, jy, jz


def reference_energies(L, S, I, E_J, A_J, B_J, g_I, Bfield):
    """Eigenvalues (cm^-1) of the independently built HFS + Zeeman Hamiltonian."""
    dl, ds, di = int(2 * L + 1), int(2 * S + 1), int(2 * I + 1)
    eye_l, eye_s, eye_i = np.eye(dl), np.eye(ds), np.eye(di)

    def embed(op_l, op_s, op_i):
        return np.kron(np.kron(op_l, op_s), op_i)

    Lc = [embed(o, eye_s, eye_i) for o in spin_ops(L)]
    Sc = [embed(eye_l, o, eye_i) for o in spin_ops(S)]
    Ic = [embed(eye_l, eye_s, o) for o in spin_ops(I)]
    Jc = [l + s for l, s in zip(Lc, Sc)]
    J2 = sum(j @ j for j in Jc)
    IJ = sum(i @ j for i, j in zip(Ic, Jc))
    one = np.eye(dl * ds * di)

    Jvals = np.arange(abs(L - S), L + S + 0.1)
    H = np.zeros_like(one, dtype=complex)
    for J, E, a, b in zip(Jvals, E_J, A_J, B_J):
        P = one.copy()
        for Jo in Jvals:
            if Jo != J:
                P = P @ (J2 - Jo * (Jo + 1) * one) / (J * (J + 1) - Jo * (Jo + 1))
        HJ = E * one + a * HZ_TO_CM * IJ
        if b and I >= 1 and J >= 1:
            Q = (3 * IJ @ IJ + 1.5 * IJ - I * (I + 1) * J * (J + 1) * one) / (
                2 * I * (2 * I - 1) * J * (2 * J - 1))
            HJ = HJ + b * HZ_TO_CM * Q
        H += P @ HJ @ P
    H += Bfield * A.muBcm * (A.g_L * Lc[2] + A.g_S * Sc[2] - g_I * Ic[2])
    return np.linalg.eigvalsh(H)


def atomspect_energies(L, S, I, E_J, A_J, B_J, g_I, Bfield, sortE=False, **kw):
    level = A.stateprocess_HFS(
        L, S, I, {"HFSA": A_J, "HFSB": B_J}, g_I, np.asarray(E_J, float), Bfield,
        sortE=sortE)
    return A.Zeeman_func_HFS(level, Bfield, g_I, **kw)[0]


# 85Rb constants. 5S1/2 from Steck; 5P (J=1/2, 3/2) as in Examples/Rb_ex1/Rb_ex.py.
RB85_I = 2.5
RB85_GI = A.lande_gi(2.5, 1.35298)
RB_GROUND = dict(L=0, S=0.5, E_J=[0.0], A_J=[1011.910813e6], B_J=[0.0])
RB_EXCITED = dict(L=1, S=0.5, E_J=[12578.950, 12816.545],
                  A_J=[120.527e6, 25e6], B_J=[0.0, 25.88e6])


def test_rb85_ground_zero_field_splitting():
    """A(I+1/2) for a J=1/2 level, converted with 1/(100 c)."""
    e = atomspect_energies(I=RB85_I, g_I=RB85_GI, Bfield=0, **RB_GROUND)
    expected = 1011.910813e6 * (RB85_I + 0.5) * HZ_TO_CM
    assert np.ptp(e) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("level", [RB_GROUND, RB_EXCITED], ids=["5S", "5P"])
@pytest.mark.parametrize("Bfield", [0.0, 0.01, 0.1, 1.0])
@pytest.mark.parametrize("sortE", [False, True])
def test_matches_reference_hamiltonian(level, Bfield, sortE):
    got = atomspect_energies(I=RB85_I, g_I=RB85_GI, Bfield=Bfield, sortE=sortE, **level)
    ref = reference_energies(I=RB85_I, g_I=RB85_GI, Bfield=Bfield, **level)
    np.testing.assert_allclose(np.sort(got), ref, rtol=0, atol=1e-9)


def test_breit_rabi_rb85_ground():
    """Analytic Breit-Rabi formula for J=1/2 (H_Z sign convention of AtomSpect)."""
    I, gJ, gI = RB85_I, A.g_S, -RB85_GI
    dE = 1011.910813e6 * (I + 0.5) * HZ_TO_CM
    for Bfield in [0.01, 0.1, 0.5]:
        x = (gJ - gI) * A.muBcm * Bfield / dE
        expected = []
        for m in np.arange(-I - 0.5, I + 0.6):
            base = -dE / (2 * (2 * I + 1)) + gI * A.muBcm * m * Bfield
            if abs(m) == I + 0.5:
                # Stretched states: dE*I/(2I+1) +/- (gJ/2 + I*gI)*muB*B
                expected.append(dE * I / (2 * I + 1)
                                + np.sign(m) * (gJ / 2 + I * gI) * A.muBcm * Bfield)
            else:
                root = dE / 2 * np.sqrt(1 + 4 * m * x / (2 * I + 1) + x**2)
                expected += [base + root, base - root]
        got = atomspect_energies(I=I, g_I=RB85_GI, Bfield=Bfield, **RB_GROUND)
        np.testing.assert_allclose(np.sort(got), np.sort(expected), rtol=0, atol=1e-9)


def test_b_constants_follow_their_j_manifold():
    """Each state's diagonal shift must use the B constant of its own J."""
    I, a_j, b_j = 1.5, [10e6, 20e6], [30e6, 40e6]
    level = A.stateprocess_HFS(1, 0.5, I, {"HFSA": a_j, "HFSB": b_j},
                               A.lande_gi(I, 1.0), np.array([100.0, 200.0]), 0)
    for shift, state in zip(level[2], level[-1]):
        k = 0 if state[2] == 0.5 else 1
        assert shift == pytest.approx(
            A.Delta_E_HFS(I, state[2], state[0], a_j[k], b_j[k]), abs=1e-15)


@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("level", [RB_GROUND, RB_EXCITED], ids=["5S", "5P"])
def test_psihfs_boolean(flag, level):
    """PsiHFS=True and PsiHFS=False must both give the reference Hamiltonian."""
    got = atomspect_energies(I=RB85_I, g_I=RB85_GI, Bfield=0.1, PsiHFS=flag, **level)
    ref = reference_energies(I=RB85_I, g_I=RB85_GI, Bfield=0.1, **level)
    np.testing.assert_allclose(np.sort(got), ref, rtol=0, atol=1e-9)
