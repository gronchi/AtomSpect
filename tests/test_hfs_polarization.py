"""HFS labels checked against uncoupled orbital E1 operators.

Rb85 energies and excited A/B values follow Examples/Rb_ex1/Rb_ex.py.
Ground A=1011.910813 MHz is from Steck's Rb85 D Line Data:
https://steck.us/alkalidata/rubidium85numbers.pdf. Fields, viewing angles
and transmission factors are test choices, not fitted measurements.
"""
from pathlib import Path
from functools import lru_cache
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest
from sympy.physics.wigner import wigner_3j

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import AtomSpect


@lru_cache(maxsize=None)
def levels(field, sort=False):
    gi = AtomSpect.lande_gi(2.5, 1.35298)
    ground = AtomSpect.stateprocess_HFS(0, .5, 2.5,
        {'HFSA': [1011.910813e6]}, gi, np.array([0.]), field, sortE=sort)
    excited = AtomSpect.stateprocess_HFS(1, .5, 2.5,
        {'HFSA': [120.527e6, 25.0354e6], 'HFSB': [0., 25.898e6]},
        gi, np.array([12578.95, 12816.545]), field, sortE=sort)
    return ground, AtomSpect.Zeeman_func_HFS(ground, field, gi), excited, AtomSpect.Zeeman_func_HFS(excited, field, gi)


def orbital_operators(ground, excited):
    """Orbital E1 changes mL only; spin and nuclear projections are spectators."""
    operators = []
    for q in [-1, 0, 1]:
        uncoupled = np.zeros((len(excited[-1]), len(ground[-1])))
        for e, se in enumerate(excited[-1]):
            for g, sg in enumerate(ground[-1]):
                if se[9] == sg[9] and se[5] == sg[5]:
                    uncoupled[e, g] = (-1)**(se[6]-se[7])*float(wigner_3j(
                        se[6], 1, sg[6], -se[7], q, sg[7]))
        coupled = np.linalg.multi_dot([excited[0], uncoupled, ground[0].T])
        operators.append(coupled.ravel())
    return np.array(operators).T


def check_reference(args, result, theta=90, gamma=0, filtered=False, alpha=1, beta=1):
    ground, _, excited, _ = args
    transform = result[4]
    amplitudes = np.dot(transform.T, orbital_operators(ground, excited))
    raw = np.abs(amplitudes)**2
    theta, gamma = np.deg2rad([theta, gamma])
    a = alpha*(np.cos(gamma)**2 if filtered else 1)
    b = beta*(np.sin(gamma)**2 if filtered else 1)
    weights = np.array([(a*np.cos(theta)**2+b)/2, a*np.sin(theta)**2,
                        (a*np.cos(theta)**2+b)/2])
    expected = np.dot(raw, weights)
    expected /= expected.sum()
    np.testing.assert_allclose(result[0], expected, rtol=0, atol=2e-12)
    basis_delta = np.array([e[1]-g[1] for e in excited[-1] for g in ground[-1]])
    delta = np.dot(basis_delta, np.abs(transform)**2)
    labels = np.array([r[1] for r in result[2]])
    active = raw.sum(axis=1) > raw.max()*1e-12
    np.testing.assert_allclose(labels[active], delta[active], rtol=0, atol=1e-10)
    assert not np.any(labels[active] == 3)
    for column, q in enumerate([-1, 0, 1]):
        assert np.max(raw[labels != q, column]) < raw.max()*1e-12


@pytest.mark.parametrize('field', [0., 1e-9, .01, .1, .5, -.5])
@pytest.mark.parametrize('sort', [False, True])
def test_labels_and_strengths_against_orbital_e1(field, sort):
    args = levels(field, sort)
    result = AtomSpect.Zeeman_signal_HFS(*args)
    check_reference(args, result)


@pytest.mark.parametrize('theta,gamma,filtered', [(0, 0, False), (45, 30, False),
    (90, 0, True), (90, 90, True), (45, 30, True)])
def test_geometry_filter_and_film(theta, gamma, filtered):
    args = levels(.5)
    result = AtomSpect.Zeeman_signal_HFS(*args, Bangle=theta, gamma=gamma,
                Filter=filtered, T_Alpha=.65, T_Beta=.35)
    check_reference(args, result, theta, gamma, filtered, .65, .35)


def test_permuted_eigenpairs_follow_supplied_columns():
    args = levels(.1)
    original = AtomSpect.Zeeman_signal_HFS(*args)
    rng = np.random.default_rng(431)
    pg, pe = rng.permutation(len(args[1][0])), rng.permutation(len(args[3][0]))
    changed = (args[0], [args[1][0][pg], args[1][1][:, pg]],
               args[2], [args[3][0][pe], args[3][1][:, pe]])
    result = AtomSpect.Zeeman_signal_HFS(*changed)
    order = (pe[:, None]*len(pg)+pg[None, :]).ravel()
    np.testing.assert_allclose(result[0], original[0][order], rtol=0, atol=2e-12)
    np.testing.assert_array_equal(np.array(result[2])[:, 1], np.array(original[2])[order, 1])
    np.testing.assert_array_equal(result[6], original[6][order])
    check_reference(changed, result)


def test_degenerate_mixtures_are_resolved_without_mutating_inputs():
    args = list(levels(0.))
    rng = np.random.default_rng(223)
    for index in [1, 3]:
        energies, vectors = args[index][0].copy(), args[index][1].copy()
        for energy in np.unique(energies):
            columns = np.flatnonzero(energies == energy)
            rotation, _ = np.linalg.qr(rng.normal(size=(len(columns), len(columns))))
            vectors[:, columns] = np.dot(vectors[:, columns], rotation)
        args[index] = [energies, vectors]
    snapshots = [args[i][1].copy() for i in [1, 3]]
    result = AtomSpect.Zeeman_signal_HFS(*args)
    check_reference(args, result)
    for i, snapshot in zip([1, 3], snapshots):
        np.testing.assert_array_equal(args[i][1], snapshot)


def test_complex_eigenvector_phases_do_not_change_strengths():
    args = list(levels(.1))
    original = AtomSpect.Zeeman_signal_HFS(*args)
    for index in [1, 3]:
        energies, vectors = args[index]
        args[index] = [energies, np.asarray(vectors)*np.exp(1j*np.arange(len(energies)))]
    result = AtomSpect.Zeeman_signal_HFS(*args)
    np.testing.assert_allclose(result[0], original[0], rtol=0, atol=2e-12)
    check_reference(args, result)


def test_crossing_scan_has_no_active_forbidden_labels():
    # A small synthetic I=1/2 term makes a dense scan affordable; constants
    # and energies are deliberately synthetic, not tabulated atomic data.
    gi = .001
    ground = AtomSpect.stateprocess_HFS(0, .5, .5, {'HFSA': [100e6]},
                                       gi, np.array([0.]), 0.)
    excited = AtomSpect.stateprocess_HFS(1, .5, .5,
        {'HFSA': [30e6, 40e6], 'HFSB': [0., 0.]}, gi,
        np.array([10000., 10100.]), 0.)
    for field in np.concatenate([np.linspace(-.02, .02, 41), [1e-12, -1e-12]]):
        args = (ground, AtomSpect.Zeeman_func_HFS(ground, field, gi),
                excited, AtomSpect.Zeeman_func_HFS(excited, field, gi))
        result = AtomSpect.Zeeman_signal_HFS(*args)
        check_reference(args, result)


def test_main_partitions_use_eigenpair_labels():
    deck = dict(s_ground=.5, s_excited=.5, l_ground=0, l_excited=1,
        E_ground=np.array([0.]), E_excited=np.array([12578.95, 12816.545]),
        I_spin=2.5, mu_I=1.35298, HFS_G=[1011.910813e6],
        HFS_E=[120.527e6, 25.0354e6], HFSB_E=[0., 25.898e6],
        Bmag=.1, b_angle=90, amu=85, spec_window=[775, 799],
        plot_window=[775, 799], specstep=.01, specres=1)
    result = AtomSpect.AtomSpect_Main(deck)
    direct = AtomSpect.Zeeman_signal_HFS(*levels(.1))
    np.testing.assert_allclose(result['signal'], direct[0], rtol=0, atol=2e-12)
    assert np.sum(result['other_sticks'][1]) == pytest.approx(0., abs=1e-14)
    for key, label in [('pi_sticks', 0), ('sigmaplus_sticks', -1), ('sigmaminus_sticks', 1)]:
        mask = np.array([r[1] for r in direct[2]]) == label
        np.testing.assert_allclose(result[key][1], direct[0][mask], rtol=0, atol=2e-12)
