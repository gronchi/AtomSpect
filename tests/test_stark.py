"""Independent scalar/tensor dc Stark and zero-field compatibility checks."""
import os

os.environ.setdefault('MPMATH_NOGMPY', '1')
os.environ.setdefault('SYMPY_GROUND_TYPES', 'python')

import numpy as np
import pytest

import AtomSpect as A
from AtomSpect.config import InputDeck


def deck(hfs=False, **extra):
    d = dict(s_ground=.5, s_excited=.5, l_ground=0, l_excited=1,
             E_ground=[0.], E_excited=[12578.95098147, 12816.54938993],
             Bmag=.03, b_angle=37., amu=86.90918, Temp=300.,
             Convfxn='GaussianInstrum', spec_window=[779.9, 780.2],
             plot_window=[779.9, 780.2], fxnwindow=.02)
    if hfs:
        d.update(I_spin=1.5, mu_I=2.74, HFS_G=[3.417341305452145e9],
                 HFS_E=[407.25e6, 84.7185e6], HFSB_E=[0., 12.4965e6])
    return dict(d, **extra)


def assert_same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            if key != 'Input':
                assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    else:
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('hfs', [False, True])
def test_zero_field_is_identical_to_legacy_path(hfs):
    assert_same(A.AtomSpect_Main(deck(hfs)), A.AtomSpect_Main(deck(hfs, Efield=0.)))


def test_scalar_transition_shift_and_field_squared():
    # J_lower=0, J_upper=1; with scalar-only shifts every line moves equally.
    base = dict(s_ground=0, s_excited=0, l_ground=0, l_excited=1,
                E_ground=[0.], E_excited=[20000.], Bmag=0., b_angle=90.,
                spec_window=[499., 501.], plot_window=[499., 501.],
                Convfxn='GaussianInstrum', Temp=300., amu=40.)
    ordinary = A.AtomSpect_Main(base.copy())
    for electric in [1000., -1000., 2000.]:
        shifted = A.AtomSpect_Main(dict(base, Efield=electric, alpha0_G=.1, alpha0_E=.3))
        got_hz = (1e7/shifted.wave_vac - 1e7/ordinary.wave_vac)*100*A.c_light
        np.testing.assert_allclose(got_hz, -.5*(.3-.1)*electric**2, atol=.2)
        np.testing.assert_array_equal(shifted.signal, ordinary.signal)


def test_electronic_tensor_eigenvalues_and_trace():
    # Pure J=1 term. Tensor factors are +1,-2,+1 for m=-1,0,1.
    level = A.stateprocess(1, 0, [0.], 0.)
    matrix = A.stark_hamiltonian(level, Efield=100., alpha0=.2, alpha2=-.04)
    np.testing.assert_allclose(np.diag(matrix)*100*A.c_light, [-800., -1400., -800.])
    tensor_only = A.stark_hamiltonian(level, 100., 0., -.04)
    assert np.trace(tensor_only) == pytest.approx(0., abs=1e-20)


@pytest.mark.parametrize('electric', [1., 3000., 30000.])
def test_hfs_tensor_against_independent_spin_hamiltonian(electric):
    # Build fixed-J hyperfine Hamiltonians in |mJ,mI> without CG coefficients.
    I = 1.5
    reference = []
    for J, origin, ahz, bhz, a0, a2 in [
        (.5, 0., 407.25e6, 0., .2, 0.),
        (1.5, 100., 84.7185e6, 12.4965e6, .3, -.04),
    ]:
        def spin(j):
            m = np.arange(-j, j+1)
            z = np.diag(m)
            plus = np.diag(np.sqrt(j*(j+1)-m[:-1]*(m[:-1]+1)), -1)
            return (plus+plus.T)/2, (plus-plus.T)/2j, z
        jops = [np.kron(o, np.eye(4)) for o in spin(J)]
        iops = [np.kron(np.eye(int(2*J+1)), o) for o in spin(I)]
        ij = sum(j@i for j, i in zip(jops, iops))
        identity = np.eye(len(ij))
        h = ahz*ij
        if J >= 1:
            h += bhz*(3*ij@ij+1.5*ij-I*(I+1)*J*(J+1)*identity)/(2*I*(2*I-1)*J*(2*J-1))
        h += -.5*electric**2*a0*identity
        if J >= 1:
            h += -.5*electric**2*a2*(3*jops[2]@jops[2]-J*(J+1)*identity)/(J*(2*J-1))
        reference.extend(np.linalg.eigvalsh(h)/(100*A.c_light)+origin)
    level = A.stateprocess_HFS(1, .5, I, {'HFSA': [407.25e6, 84.7185e6],
                 'HFSB': [0., 12.4965e6]}, 0., [0., 100.], 0.)
    got = A.Zeeman_func_HFS(level, 0., 0., Efield=electric, alpha0=[.2, .3], alpha2=[0., -.04])[0]
    np.testing.assert_allclose(got, np.sort(reference), rtol=0, atol=1e-12)
    if electric > 1:
        stark = A.stark_hamiltonian(level, electric, [.2, .3], [0., -.04])
        assert np.max(np.abs(stark-np.diag(np.diag(stark)))) > 0


def test_weak_hfs_tensor_diagonal_matches_steck_eq31():
    level = A.stateprocess_HFS(1, .5, 1.5, {'HFSA': [1e6, 2e6]}, 0., [0., 100.], 0.)
    actual = np.diag(A.stark_hamiltonian(level, 10., [.2, .3], [0., -.04]))*100*A.c_light
    for shift, s in zip(actual, level[-1]):
        F, m, J, _, I = s[:5]
        scalar, tensor = (.2, 0.) if J == .5 else (.3, -.04)
        factor = 0.
        if J >= 1 and F >= 1:
            X = F*(F+1)+J*(J+1)-I*(I+1)
            factor = ((3*m*m-F*(F+1))*(3*X*(X-1)-4*F*(F+1)*J*(J+1))
                      / ((2*F+3)*(2*F+2)*F*(2*F-1)*J*(2*J-1)))
        assert shift == pytest.approx(-.5*10**2*(scalar+tensor*factor), abs=1e-12)


def test_hfs_stark_matrix_respects_basis_permutation():
    level = A.stateprocess_HFS(1, .5, 1.5, {'HFSA': [1e6, 2e6]}, 0., [0., 100.], 0.)
    matrix = A.stark_hamiltonian(level, 1000., [.2, .3], [0., -.04])
    permutation = np.random.default_rng(12).permutation(len(level[-1]))
    reordered = list(level)
    reordered[-1] = [level[-1][i] for i in permutation]
    result = A.stark_hamiltonian(reordered, 1000., [.2, .3], [0., -.04])
    np.testing.assert_allclose(result, matrix[np.ix_(permutation, permutation)], atol=1e-20)


@pytest.mark.parametrize('hfs', [False, True])
def test_fan_matches_spectrum_solver(hfs):
    d = deck(hfs, Efield=10000., alpha0_G=.0794,
             alpha0_E=[.0794+.122306, .0794+.1340], alpha2_E=[0., -.0406])
    spectrum = A.AtomSpect_Main(d.copy())
    fan = A.ZeemanFan(dict(d, Bmag=[d['Bmag']]), 'G')
    np.testing.assert_allclose(fan['G'][:, 0], spectrum.ZeemanSplits_G, atol=1e-12)
    assert_same(A.ZeemanFan(deck(hfs), 'GE'), A.ZeemanFan(deck(hfs, Efield=0.), 'GE'))


def test_atomic_model_and_radiative_bridge():
    # Use the bridge to exercise Stark shifts in the newer calibrated API.
    d = deck(True, Efield=10000., alpha0_G=.0794,
             alpha0_E=[.201706, .2134], alpha2_E=[0., -.0406])
    d['RadiativeData'] = dict(orbital_strength_au=20., isotope_id='Rb87')
    result = A.AtomSpect_Main(d)
    model_result = result.AtomicComponents
    assert 'dc_stark' in model_result.metadata
    assert 'STARK_NOT_INCLUDED' not in model_result.validity_flags
    assert 'STARK_CROSS_J_NOT_INCLUDED' in model_result.validity_flags
    ordinary = A.AtomSpect_Main(dict(d, Efield=0.))
    assert np.max(np.abs(model_result.nu_Hz-ordinary.AtomicComponents.nu_Hz)) > 1e6
    # Remove constants differences to compare tensor/scalar Hamiltonians.
    lower = A.LSTerm('l', 0, 1, {1: 0.}, 1)
    upper = A.LSTerm('u', 1, 1, {1: 100., 3: 200.}, -1)
    model = A.build_model_b(lower, upper, orbital_strength_au=1.)
    assert A.add_dc_stark(model) is model
    data = A.HyperfineData(3, 1., {x.level_id: (1e6, 0.) for x in model.levels}, 'test')
    polar = {'l:J1': (.1, 0.), 'u:J1': (.2, 0.), 'u:J3': (.3, -.04)}
    first_stark = A.add_hyperfine(A.add_dc_stark(model, 10000., polar), data)
    first_hfs = A.add_dc_stark(A.add_hyperfine(model, data), 10000., polar)
    np.testing.assert_allclose(first_stark.H0, first_hfs.H0, atol=1e-14)
    with pytest.raises(ValueError, match='already'):
        A.add_dc_stark(first_hfs, 1., polar)


def test_atomic_model_scalar_shift_changes_transition_rates_with_frequency():
    levels = [A.AtomicLevel('l', 0., 1, 1, g_J=2.),
              A.AtomicLevel('u', 100., 3, -1, g_J=1.)]
    model = A.build_model_a(levels, [A.RadiativeReference('u', 'l', 'strength_au', 1.)])
    old = A.calculate_components(model, .01)
    new = A.calculate_components(A.add_dc_stark(model, 1000., {'l': (.1, 0.), 'u': (.3, 0.)}), .01)
    np.testing.assert_allclose(new.nu_Hz-old.nu_Hz, -100000., atol=1e-3)
    np.testing.assert_allclose(new.strength_au, old.strength_au, atol=1e-14)
    nonzero = old.A_s1 > 0
    np.testing.assert_allclose(new.A_s1[nonzero]/old.A_s1[nonzero],
                               (new.nu_Hz[nonzero]/old.nu_Hz[nonzero])**3)


@pytest.mark.parametrize('extra, error', [
    ({'Efield': float('nan')}, ValueError),
    ({'Efield': [1., 2.]}, ValueError),
    ({'Efield': 1.}, ValueError),
    ({'Efield': 1., 'alpha0_G': .1, 'alpha0_E': [1., 2., 3.]}, ValueError),
    ({'Efield': 1., 'alpha0_G': .1, 'alpha0_E': .2, 'alpha2_G': .1}, ValueError),
    ({'Efield': 1., 'alpha0_G': .1, 'alpha0_E': .2, 'DoLowSig': True}, NotImplementedError),
])
def test_invalid_stark_inputs_fail_early(extra, error):
    with pytest.raises(error):
        InputDeck.from_dict(deck(**extra))
