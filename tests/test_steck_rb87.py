"""Independent Rb-87 benchmarks from Steck revision 2.3.4 (2025-08-08).

https://steck.us/alkalidata/rubidium87numbers.pdf
Tests distinguish mathematical consistency from precision of atomic inputs.
"""
import os

os.environ.setdefault('MPMATH_NOGMPY', '1')
os.environ.setdefault('SYMPY_GROUND_TYPES', 'python')

import numpy as np
import pytest

import AtomSpect as A


def hfs_state(F, m, J, L):
    return [F, m, J, 0, 1.5, 0, L, 0, .5, 0]


@pytest.mark.parametrize('J, expected', [
    (.5, {1: {1: 1/6, 2: 5/6}, 2: {1: 1/2, 2: 1/2}}),
    (1.5, {1: {0: 1/6, 1: 5/12, 2: 5/12},
           2: {1: 1/20, 2: 1/4, 3: 7/10}}),
])
def test_table_8_strengths_for_every_ground_m(J, expected):
    # Work with absorption matrix elements, before observing a direction.
    # Raw amplitudes omit the common orbital norm. Normalize by the sum
    # over all excited F, m for one ground sublevel, as in Eqs. (39)-(42).
    for F, branches in expected.items():
        for m in range(-F, F+1):
            sums = {}
            for Fe in range(int(abs(J-1.5)), int(J+1.5)+1):
                sums[Fe] = sum(A.dipolestr_HFS(
                    hfs_state(F, m, .5, 0), hfs_state(Fe, me, J, 1))**2
                    for me in range(-Fe, Fe+1))
            total = sum(sums.values())
            for Fe, strength in sums.items():
                assert strength/total == pytest.approx(branches.get(Fe, 0), abs=1e-14)


def test_table_9_and_10_component_strengths():
    # For this orbital channel, sum over excited sublevels is 2/3;
    # Table 9/10 list amplitudes relative to Steck's electronic dipole.
    sigma_squared = [1/30, 1/10, 1/5, 1/3, 1/2]
    pi_squared = [1/6, 4/15, 3/10, 4/15, 1/6]
    for m, sigma, pi in zip(range(-2, 3), sigma_squared, pi_squared):
        ground = hfs_state(2, m, .5, 0)
        assert A.dipolestr_HFS(ground, hfs_state(3, m+1, 1.5, 1))**2/(2/3) == pytest.approx(sigma)
        assert A.dipolestr_HFS(ground, hfs_state(3, m, 1.5, 1))**2/(2/3) == pytest.approx(pi)


def test_zero_field_intervals_and_center_of_gravity():
    for J, ahz, bhz, expected_intervals in [
        (.5, 3.417341305452145e9, 0., [6.83468261090429e9]),
        (.5, 407.25e6, 0., [814.5e6]),
        (1.5, 84.7185e6, 12.4965e6, [72.222e6, 156.9405e6, 266.652e6]),
    ]:
        Fs = np.arange(abs(J-1.5), J+1.5+1)
        hz = np.array([A.Delta_E_HFS(1.5, J, F, ahz, bhz) for F in Fs])*100*A.c_light
        np.testing.assert_allclose(np.diff(hz), expected_intervals, rtol=1e-14)
        assert np.dot(2*Fs+1, hz) == pytest.approx(0., abs=1e-5)


@pytest.mark.parametrize('field', [0., 1e-4, .01, .1, 1.])
def test_rb87_legacy_eigensolver_against_breit_rabi(field):
    # Use the same constants on both sides to test assembly, not input accuracy.
    ahz = 3.417341305452145e9
    gi = .0009951414  # Positive moment convention; Steck's gI is negative.
    level = A.stateprocess_HFS(0, .5, 1.5, {'HFSA': [ahz]}, gi, [0.], field)
    got = A.Zeeman_func_HFS(level, field, gi)[0]*100*A.c_light
    d = 2*ahz
    scale = A.muBcm*100*A.c_light*field
    x = (A.g_S+gi)*scale/d
    expected = [d*1.5/4 + sign*(A.g_S/2-1.5*gi)*scale for sign in [-1, 1]]
    for m in [-1, 0, 1]:
        center = -d/8-gi*m*scale
        root = d/2*np.sqrt(1+m*x+x*x)
        expected.extend([center-root, center+root])
    np.testing.assert_allclose(np.sort(got), np.sort(expected), rtol=0, atol=1e-4)


@pytest.mark.parametrize('J, wavelength, tau, dipole, oscillator', [
    (.5, 794.978851156, 27.679e-9, 2.9931, .34231),
    (1.5, 780.241209686, 26.2348e-9, 4.22752, .69577),
])
def test_lifetime_oscillator_and_reduced_dipole_conventions(J, wavelength, tau, dipole, oscillator):
    k = A.AtomicConstants()
    lower = A.AtomicLevel('s', 0, 1, 1, g_J=2.002331070)
    upper = A.AtomicLevel('p', 1e7/wavelength, int(2*J), -1, g_J=1.)
    reference = A.RadiativeReference('p', 's', 'lifetime', tau,
                                    wavelength_vac_nm=wavelength, branching_fraction=1.)
    strength = reference.strength_au(upper, lower, k)
    # Steck Eq. (39) uses a dipole smaller than our standard reduced
    # matrix element by sqrt(2*J_lower+1), here sqrt(2).
    assert np.sqrt(strength/2) == pytest.approx(dipole, rel=2e-5)
    from_f = A.RadiativeReference('p', 's', 'f', oscillator,
                                wavelength_vac_nm=wavelength).strength_au(upper, lower, k)
    assert from_f == pytest.approx(strength, rel=2e-5)
    components = A.calculate_components(A.build_model_a([lower, upper], [reference]), 0.)
    emitting = components.states['parent_level_overlaps'][:, 1] > .99
    np.testing.assert_allclose(components.states['partial_decay_rate_s1'][emitting], 1/tau, rtol=1e-12)
    # Pure q radiation patterns, Eqs. (58)-(59), with absolute normalization.
    for angle in [0., 37., 90.]:
        t = np.deg2rad(angle)
        pattern = np.where(components.q == 0, 3/(8*np.pi)*np.sin(t)**2,
                           3/(16*np.pi)*(1+np.cos(t)**2))
        np.testing.assert_allclose(components.directional_rates(angle), components.A_s1*pattern,
                                   rtol=1e-12, atol=1e-8)
