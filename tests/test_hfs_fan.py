"""HFS fan regression tests, independent of the repaired fan formula.

Rb85 ground-state I=5/2 and A=1011.910813 MHz come from Steck's
Rubidium 85 D Line Data (https://steck.us/alkalidata/rubidium85numbers.pdf).
Other constants and energies below are synthetic, chosen to distinguish J
manifolds and expose indexing, units, nuclear-sign and quadrupole errors.
The reference uses spin matrices in |mJ,mI>, not AtomSpect's state rows.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import AtomSpect

HZ_TO_CM = 1 / (100 * 299792458)


def spin(j):
    m = np.arange(-j, j + 1)
    z = np.diag(m)
    plus = np.diag(np.sqrt(j*(j+1) - m[:-1]*(m[:-1]+1)), -1)
    return (plus + plus.T)/2, (plus - plus.T)/2j, z


def reference(J, L, S, I, energy, a, b, gi, field):
    """Full fixed-J Hamiltonian; its diagonal is the first-order HPB limit."""
    j_ops = [np.kron(op, np.eye(int(2*I+1))) for op in spin(J)]
    i_ops = [np.kron(np.eye(int(2*J+1)), op) for op in spin(I)]
    ij = sum(np.dot(j, i) for j, i in zip(j_ops, i_ops))
    identity = np.eye(ij.shape[0])
    h = energy*identity + a*HZ_TO_CM*ij
    if I >= 1 and J >= 1:
        q = (3*np.dot(ij, ij) + 1.5*ij - I*(I+1)*J*(J+1)*identity)
        h += b*HZ_TO_CM*q/(2*I*(2*I-1)*J*(2*J-1))
    gj = (0 if J == 0 else
          (AtomSpect.g_L*(J*(J+1)+L*(L+1)-S*(S+1))
           + AtomSpect.g_S*(J*(J+1)-L*(L+1)+S*(S+1)))/(2*J*(J+1)))
    h += AtomSpect.muBcm*field*(gj*j_ops[2] - gi*i_ops[2])
    return np.real(np.diag(h)), np.linalg.eigvalsh(h)


def deck(gi=0.001):
    # Synthetic fine-structure energies in cm^-1; A/B in Hz.
    return dict(s_ground=0.5, l_ground=0, E_ground=np.array([0.0]),
                s_excited=0.5, l_excited=1, E_excited=np.array([100., 300.]),
                I_spin=1.5, mu_I=gi*1.5*AtomSpect.m_prot/AtomSpect.m_elec,
                HFS_G=[100e6], HFS_E=[20e6, 40e6], HFSB_E=[0., 15e6],
                Bmag=np.array([0., 0.2, -0.5]))


@pytest.mark.parametrize('gi', [0., 0.001, -0.001])
@pytest.mark.parametrize('flags', ['GH', 'EH', 'GEHL'])
def test_fan_matches_independent_diagonal(gi, flags):
    inputs = deck(gi)
    got = AtomSpect.ZeemanFan(inputs, flags)
    for key, L, energies, aa, bb in [
        ('G', 0, [0.], [100e6], [0.]),
        ('E', 1, [100., 300.], [20e6, 40e6], [0., 15e6])]:
        if key not in flags:
            continue
        for column, field in enumerate(inputs['Bmag']):
            expected = []
            for J, energy, a, b in zip(np.arange(abs(L-.5), L+.6), energies, aa, bb):
                diagonal, _ = reference(J, L, .5, 1.5, energy, a, b, gi, field)
                expected.extend(diagonal)
            np.testing.assert_allclose(np.sort(got['High'+key][:, column]),
                                       np.sort(expected), rtol=0, atol=1e-12)
        # Requesting approximations must not alter the exact fan.
        exact = AtomSpect.ZeemanFan(inputs, key)
        np.testing.assert_array_equal(got[key], exact[key])


@pytest.mark.parametrize('sortE', [False, True])
def test_constants_follow_j_even_when_states_reordered(sortE):
    level = AtomSpect.stateprocess_HFS(1, .5, 1.5,
            {'HFSA': [20e6, 40e6], 'HFSB': [0., 15e6]}, .001,
            np.array([100., 99.]), .1, sortE=sortE)
    got = AtomSpect.FastZfan(level, -.2, g_I=.001,
                            A_HFS=[20e6, 40e6], B_HFS=[0., 15e6])[1]
    expected = np.concatenate([reference(.5, 1, .5, 1.5, 100., 20e6, 0., .001, -.2)[0],
                               reference(1.5, 1, .5, 1.5, 99., 40e6, 15e6, .001, -.2)[0]])
    np.testing.assert_allclose(np.sort(got), np.sort(expected), rtol=0, atol=1e-12)
    # Check each row too: a correct spectrum with permuted line identities
    # must not silently pass when sortE changes the state-list order.
    by_state = {}
    for J, energy, a, b in [(.5, 100., 20e6, 0.), (1.5, 99., 40e6, 15e6)]:
        diagonal, _ = reference(J, 1, .5, 1.5, energy, a, b, .001, -.2)
        keys = [(J, mj, mi) for mj in np.arange(-J, J+1)
                for mi in np.arange(-1.5, 2.)]
        by_state.update(zip(keys, diagonal))
    np.testing.assert_allclose(got, [by_state[(s[2], s[3], s[5])] for s in level[-1]],
                               rtol=0, atol=1e-12)


def test_rb85_ground_converges_to_full_hamiltonian():
    inputs = deck()
    inputs.update(I_spin=2.5, mu_I=1.35298, HFS_G=[1011.910813e6],
                  Bmag=np.array([1., 10., 100.]))
    gi = AtomSpect.lande_gi(2.5, 1.35298)
    got = AtomSpect.ZeemanFan(inputs, 'GH')
    errors = []
    for column, field in enumerate(inputs['Bmag']):
        _, exact = reference(.5, 0, .5, 2.5, 0., 1011.910813e6, 0., gi, field)
        np.testing.assert_allclose(np.sort(got['G'][:, column]), exact, atol=1e-12, rtol=0)
        errors.append(np.max(np.abs(np.sort(got['HighG'][:, column])-exact)))
    assert errors[1] < errors[0]/8
    assert errors[2] < errors[1]/8
    assert errors[2] < 6e-5  # cm^-1; first-order approximation is not exact.


@pytest.mark.parametrize('I,J,L,S', [(0., 0., 0., 0.), (.5, .5, 0., .5), (1.5, 0., 1., 1.)])
def test_no_quadrupole_singularity(I, J, L, S):
    state = [0., 0., J, J, I, I, L, L, S, S]
    got = AtomSpect.ZeemanhighHFS(state, 10e6, .001, .2, Bhfs=30e6)
    expected, _ = reference(J, L, S, I, 0., 10e6, 30e6, .001, .2)
    assert got == pytest.approx(expected[-1], abs=1e-15)


def test_missing_constants_raise_instead_of_silent_zero():
    level = AtomSpect.stateprocess_HFS(0, .5, 1.5, {'HFSA': [100e6]}, 0., [0.], 0.)
    with pytest.raises(ValueError, match='A_HFS'):
        AtomSpect.FastZfan(level, .2)


def test_scalar_constants_and_zero_nuclear_moment():
    level = AtomSpect.stateprocess_HFS(0, .5, 1.5, {'HFSA': [100e6]}, 0., [0.], 0.)
    low, high = AtomSpect.FastZfan(level, 0., A_HFS=100e6)
    np.testing.assert_allclose(low, level[2], atol=1e-15, rtol=0)
    diagonal, _ = reference(.5, 0, .5, 1.5, 0., 100e6, 0., 0., 0.)
    np.testing.assert_allclose(np.sort(high), np.sort(diagonal), atol=1e-15, rtol=0)


@pytest.mark.parametrize('gi', [0., .001])
def test_non_hfs_fan_basis_does_not_depend_on_nuclear_moment(gi):
    level = AtomSpect.stateprocess(1, .5, np.array([100., 300.]), 0.)
    _, high = AtomSpect.FastZfan(level, .2, Eterm_=200., g_I=gi)
    expected = [200. + AtomSpect.muBcm*.2*(s[3]+2*s[5]) for s in level[-1]]
    np.testing.assert_allclose(high, expected, atol=1e-15, rtol=0)


def test_quadrupole_limit_converges_to_full_fixed_j_hamiltonian():
    # A synthetic single J=3/2 term permits a direct comparison without
    # inter-J mixing. Compare actual matrix eigenvalues, not another formula.
    errors = []
    for field in [.1, 1., 10.]:
        diagonal, exact = reference(1.5, 0, 1.5, 1.5, 0., 40e6, 15e6, .001, field)
        errors.append(np.max(np.abs(np.sort(diagonal)-exact)))
        states = [[0., 0., 1.5, mj, 1.5, mi, 0., 0., 1.5, mj]
                  for mj in np.arange(-1.5, 2.) for mi in np.arange(-1.5, 2.)]
        got = [AtomSpect.ZeemanhighHFS(s, 40e6, .001, field, 15e6) for s in states]
        np.testing.assert_allclose(got, diagonal, rtol=0, atol=1e-14)
    assert errors[1] < errors[0]/8
    assert errors[2] < errors[1]/8
