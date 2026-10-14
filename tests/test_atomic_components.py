"""Independent invariants and the revised guide's frozen synthetic benchmark."""
import os
os.environ.setdefault('MPMATH_NOGMPY', '1')
os.environ.setdefault('SYMPY_GROUND_TYPES', 'python')

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest
from scipy import constants as sc
from sympy.physics.wigner import wigner_3j, wigner_6j

import AtomSpect as A
from AtomSpect.atomic import _ls_transform


def benchmark():
    k = A.AtomicConstants(g_s=2.00231930436, mu_B_cm=.46686447719298224,
                          C_E1=2.02612688451534e18, source='guide revision 2 frozen synthetic benchmark')
    lower = A.LSTerm('lower', 0, 1, {1: 0.}, 1)
    upper = A.LSTerm('upper', 1, 1, {1: 20000., 3: 20001.}, -1)
    reference = A.RadiativeReference('upper:J1', 'lower:J1', 'A', 1e8, wavelength_vac_nm=500.)
    return A.build_model_b(lower, upper, [reference], constants=k)


def test_frozen_guide_benchmark():
    model = benchmark()
    assert model.metadata['orbital_strength_au'] == pytest.approx(18.508218950448505, rel=1e-14)
    result = A.calculate_components(model, 1.)
    upper_ids = np.unique(result.upper_index[result.upper_two_Mtotal == 1])
    expected = [(0, 1, 19999.64737066, .05873363681, 1.761915906e7),
                (0, -1, 20000.58218241, .27459969652, 8.238710319e7),
                (1, 1, 20000.88468206, .27459969652, 8.239084144e7),
                (1, -1, 20001.81949382, .05873363681, 1.762490043e7)]
    for rank, ml, wavenumber, angular, rate in expected:
        index, = np.flatnonzero((result.upper_index == upper_ids[rank]) & (result.lower_two_Mtotal == ml))
        assert result.nu_Hz[index]/(100*model.constants.c) == pytest.approx(wavenumber, abs=1e-6)
        assert result.strength_au[index]/model.metadata['orbital_strength_au'] == pytest.approx(angular, abs=1e-10)
        assert result.A_s1[index] == pytest.approx(rate, rel=1e-7)
    # Directly check the Hamiltonian's phase-sensitive off-diagonal entry.
    basis = model.basis
    indices = [i for i, b in enumerate(basis) if b['level_id'].startswith('upper') and b['two_mJ'] == 1]
    T = model.T0[np.ix_(indices, indices)]
    g = model.constants.g_s
    np.testing.assert_allclose(T, [[(4-g)/6, -(g-1)*np.sqrt(2)/3],
                                  [-(g-1)*np.sqrt(2)/3, (g+2)/6]], atol=1e-14)


@pytest.mark.parametrize('B', [0., .1, 1., 10.])
@pytest.mark.parametrize('lower_L,upper_L,two_S', [(0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 2, 1), (1, 1, 2), (2, 3, 3)])
def test_strict_ls_per_state_and_global_sums(lower_L, upper_L, two_S, B):
    def energies(L, origin):
        return {j: origin+index for index, j in enumerate(range(abs(2*L-two_S), 2*L+two_S+1, 2))}
    lower = A.LSTerm('l', lower_L, two_S, energies(lower_L, 0), 1)
    upper = A.LSTerm('u', upper_L, two_S, energies(upper_L, 20000), -1)
    model = A.build_model_b(lower, upper, orbital_strength_au=2.5)
    result = A.calculate_components(model, B)
    sums = np.bincount(result.upper_index, weights=result.strength_au, minlength=len(result.states['energy_cm']))
    upper_states = result.states['energy_cm'] > 10000
    np.testing.assert_allclose(sums[upper_states], 2.5/(2*upper_L+1), rtol=1e-12, atol=1e-14)
    assert result.strength_au.sum() == pytest.approx((two_S+1)*2.5, rel=1e-12)
    assert result.diagnostics['relative_norm_error'] < 1e-12
    assert result.diagnostics['max_unitarity_error'] < 1e-12
    assert np.all(np.isin(result.q, [-1, 0, 1]))


def test_coupled_and_independent_uncoupled_dipoles_and_energies():
    model = benchmark()
    lower, upper = (A.LSTerm(**model.metadata[name]) for name in ('lower_term', 'upper_term'))
    lower_indices = [i for i, b in enumerate(model.basis) if b['level_id'].startswith('lower')]
    upper_indices = [i for i, b in enumerate(model.basis) if b['level_id'].startswith('upper')]
    Ul, lower_unc = _ls_transform(lower, [model.basis[i] for i in lower_indices])
    Uu, upper_unc = _ls_transform(upper, [model.basis[i] for i in upper_indices])
    R = np.sqrt(model.metadata['orbital_strength_au'])
    for q in (-1, 0, 1):
        independent = np.zeros((len(upper_unc), len(lower_unc)))
        for i, (ml_u, ms_u) in enumerate(upper_unc):
            for j, (ml_l, ms_l) in enumerate(lower_unc):
                if ms_u == ms_l:
                    independent[i, j] = (-1)**(upper.L-ml_u//2)*float(wigner_3j(upper.L, 1, lower.L, -ml_u/2, q, ml_l/2))*R
        coupled = model.Dq[q][np.ix_(upper_indices, lower_indices)]
        np.testing.assert_allclose(coupled, Uu @ independent @ Ul.T, atol=1e-14)
        Hu = model.H0[np.ix_(upper_indices, upper_indices)] + model.constants.mu_B_cm*model.T0[np.ix_(upper_indices, upper_indices)]
        Hl = model.H0[np.ix_(lower_indices, lower_indices)] + model.constants.mu_B_cm*model.T0[np.ix_(lower_indices, lower_indices)]
        Eu, Cu = np.linalg.eigh(Hu)
        El, Cl = np.linalg.eigh(Hl)
        Euu, Cuu = np.linalg.eigh(Uu.T @ Hu @ Uu)
        Ell, Cll = np.linalg.eigh(Ul.T @ Hl @ Ul)
        np.testing.assert_allclose(Eu, Euu, atol=1e-10)
        np.testing.assert_allclose(El, Ell, atol=1e-10)
        np.testing.assert_allclose(np.abs(Cu.conj().T @ coupled @ Cl)**2,
                                   np.abs(Cuu.conj().T @ independent @ Cll)**2, atol=2e-10)


def test_zero_field_recovery_all_reference_branches_and_frequency_update():
    model = benchmark()
    lower = A.LSTerm(**model.metadata['lower_term'])
    upper = A.LSTerm(**model.metadata['upper_term'])
    references = [A.RadiativeReference(**model.metadata['references'][0]),
                  A.RadiativeReference('upper:J3', 'lower:J1', 'A', 1e8*(20001/20000)**3,
                                       wavelength_vac_nm=1e7/20001)]
    calibrated = A.build_model_b(lower, upper, references, constants=model.constants)
    result = A.calculate_components(calibrated, 0)
    for i, overlap in enumerate(result.states['parent_level_overlaps']):
        if overlap[1] > .99:
            assert result.states['partial_decay_rate_s1'][i] == pytest.approx(1e8, rel=1e-12)
        elif overlap[2] > .99:
            assert result.states['partial_decay_rate_s1'][i] == pytest.approx(1e8*(20001/20000)**3, rel=1e-12)
    bad = A.RadiativeReference('upper:J3', 'lower:J1', 'A', 1.1e8, wavelength_vac_nm=1e7/20001)
    with pytest.raises(ValueError, match='common orbital norm'):
        A.build_model_b(lower, upper, [references[0], bad])
    shifted_upper = A.AtomicLevel('u', 20001, 1, -1, g_J=1.)
    isolated = A.build_model_a([A.AtomicLevel('l', 0, 1, 1, g_J=2.), shifted_upper],
                               [A.RadiativeReference('u', 'l', 'A', 1e8, wavelength_vac_nm=500)])
    components = A.calculate_components(isolated, 0)
    emitting = components.states['partial_decay_rate_s1'] > 0
    np.testing.assert_allclose(components.states['partial_decay_rate_s1'][emitting], 1e8*(20001/20000)**3, rtol=1e-12)


def complex_model():
    levels = [A.AtomicLevel('l', 0., 1, 1, g_J=2.), A.AtomicLevel('a', 20000., 1, -1, g_J=.8),
              A.AtomicLevel('b', 20001., 1, -1, g_J=1.2)]
    return A.build_model_c(levels, {('a', 'l'): 1., ('b', 'l'): .7j}, {('a', 'b'): .3+.2j},
                           phase_convention_id='test shared phases', matrix_dataset_id='synthetic complex matrices')


@pytest.mark.parametrize('model_factory', [benchmark, complex_model])
def test_arbitrary_complex_consistent_basis_rephasing(model_factory):
    model = model_factory()
    phases = np.exp(1j*np.random.default_rng(71).uniform(-np.pi, np.pi, len(model.basis)))
    original = A.calculate_components(model, .7)
    rotated = A.calculate_components(A.rephase_model(model, phases), .7)
    np.testing.assert_allclose(original.nu_Hz, rotated.nu_Hz, rtol=1e-13)
    np.testing.assert_allclose(original.strength_au, rotated.strength_au, atol=1e-13)
    np.testing.assert_allclose(original.directional_rates(37, analyzer=[1/np.sqrt(2), 1j/np.sqrt(2)]),
                               rotated.directional_rates(37, analyzer=[1/np.sqrt(2), 1j/np.sqrt(2)]), rtol=1e-11, atol=1e-10)


def test_unknown_model_c_links_are_not_silent_zeros():
    levels = complex_model().levels
    options = dict(phase_convention_id='explicit', matrix_dataset_id='test')
    with pytest.raises(A.MissingAtomicData, match='magnetic'):
        A.build_model_c(levels, {('a', 'l'): 1., ('b', 'l'): .7j}, {}, **options)
    with pytest.raises(A.MissingAtomicData, match='E1'):
        A.build_model_c(levels, {('a', 'l'): 1.}, {('a', 'b'): 0.}, **options)
    with pytest.raises(ValueError, match='Hermitian'):
        A.build_model_c(levels, {('a', 'l'): 1., ('l', 'a'): 2., ('b', 'l'): 1.}, {('a', 'b'): 0.}, **options)


def test_external_data_conversions_and_si_prefactor():
    k = A.AtomicConstants()
    lower, upper = A.AtomicLevel('l', 0, 1, 1), A.AtomicLevel('u', 20000, 3, -1)
    frequency = k.c/(500e-9)
    gf = 1.2
    results = []
    for kind, value in [('f', gf/2), ('gf', gf), ('loggf', np.log10(gf))]:
        results.append(A.RadiativeReference('u', 'l', kind, value, frequency_Hz=frequency).strength_au(upper, lower, k))
    expected = 1.5*gf/(k.h*frequency/sc.physical_constants['Hartree energy'][0])
    np.testing.assert_allclose(results, expected, rtol=1e-9)
    strength_si = expected*k.strength_unit_SI
    expected_rate = 16*np.pi**3*frequency**3/(3*k.epsilon_0*k.h*k.c**3)*strength_si
    assert A.e1_rate(frequency, strength_si, units='SI', constants=k) == pytest.approx(expected_rate, rel=1e-14)
    r = A.RadiativeReference('u', 'l', 'lifetime', 1e-8, frequency_Hz=frequency, branching_fraction=.2)
    assert r.strength_au(upper, lower, k) == pytest.approx(A.e1_line_strength(frequency, 2e7, 3, constants=k))
    with pytest.raises(ValueError):
        A.RadiativeReference('u', 'l', 'lifetime', 1e-8, frequency_Hz=frequency).strength_au(upper, lower)
    with pytest.raises(A.MissingAtomicData):
        A.RadiativeReference('u', 'l', 'A', 1e8).strength_au(upper, lower)
    with pytest.raises(ValueError, match='multipole'):
        A.RadiativeReference('u', 'l', 'A', 1e8, frequency_Hz=frequency, multipole='M1+E2').strength_au(upper, lower)
    np.testing.assert_allclose(A.einstein_b(frequency, expected_rate), k.c**3*expected_rate/(8*np.pi*k.h*frequency**3))
    Bul, Blu = A.einstein_b(frequency, expected_rate, lower_weight=2, upper_weight=4)
    assert Blu == 2*Bul
    norm, variance = A.fit_ls_norm([2/3, 4/3], np.array([2/3, 4/3])*3., np.diag([.1, .2]))
    assert norm == pytest.approx(3.) and variance > 0


def test_hyperfine_references_use_F_weight_and_electronic_recoupling():
    u, l = A.AtomicLevel('u', 20000, 1, -1), A.AtomicLevel('l', 0, 1, 1)
    factor = 5*3*float(wigner_6j(.5, 2, 1.5, 1, .5, 1))**2
    rate = 1e8*2/5*factor
    hfs = A.RadiativeReference('u', 'l', 'A', rate, wavelength_vac_nm=500,
                              upper_two_F=4, lower_two_F=2, two_I=3)
    electronic = A.RadiativeReference('u', 'l', 'A', 1e8, wavelength_vac_nm=500)
    assert hfs.strength_au(u, l) == pytest.approx(electronic.strength_au(u, l), rel=1e-13)


@pytest.mark.parametrize('B', [0, .1, 1, 10])
def test_hyperfine_extension_energy_interval_and_strength_norm(B):
    model = benchmark()
    hfs = A.HyperfineData(3, 1., {x.level_id: (1e6, 0.) for x in model.levels}, 'synthetic I=3/2')
    enlarged = A.add_hyperfine(model, hfs)
    result = A.calculate_components(enlarged, B)
    assert result.total_projection_definition == 'M_F'
    assert 'HYPERFINE_UNASSESSED' not in result.validity_flags
    assert 'HFS_OFFDIAGONAL_NOT_INCLUDED' in result.validity_flags
    assert result.strength_au.sum() == pytest.approx(4*2*model.metadata['orbital_strength_au'], rel=1e-12)
    if B == 0:
        lower = result.states['energy_cm'][result.states['parent_level_overlaps'][:, 0] > .99]
        assert np.ptp(lower) == pytest.approx(2e6/(100*model.constants.c), rel=1e-11)


def test_geometry_populations_profiles_and_frequency_jacobian():
    result = A.calculate_components(benchmark(), 1)
    for theta in (0., 37., 90.):
        angle = np.deg2rad(theta)
        factors = np.where(result.q == 0, 3/(8*np.pi)*np.sin(angle)**2,
                           3/(16*np.pi)*(1+np.cos(angle)**2))
        np.testing.assert_allclose(result.directional_rates(theta), result.A_s1*factors, rtol=1e-13, atol=1e-9)
    populations = np.ones(len(result.states['energy_cm']))
    np.testing.assert_allclose(result.emission_weights(populations), result.A_s1)
    np.testing.assert_allclose(result.emission_weights(populations, power=True), result.A_s1*result.constants.h*result.nu_Hz)
    with pytest.raises(A.MissingAtomicData):
        result.emission_weights({})
    spectrum = A.synthesize_spectrum(result, np.linspace(499.8, 500.2, 20001), populations,
                                     temperature_K=300, mass_amu=40, width_nm=.002)
    assert spectrum['captured_weight'] == pytest.approx(spectrum['expected_weight'], rel=1e-8)
    # A constant density on a bounded frequency interval has a normalized wavelength transform.
    k = result.constants
    lo, hi = k.c/(501e-9), k.c/(499e-9)
    grid = np.linspace(499e-9, 501e-9, 10001)
    density = A.frequency_profile_to_wavelength(grid, lambda nu: np.ones_like(nu)/(hi-lo))
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    assert integrate(density, grid) == pytest.approx(1, rel=1e-9)


def test_strong_field_spin_conservation_and_residual_fine_structure():
    model = benchmark()
    result = A.calculate_components(model, 1e6)  # formal mathematical limit, not physical field validity
    lower, upper = (A.LSTerm(**model.metadata[name]) for name in ('lower_term', 'upper_term'))
    nl = 2
    Uu, unc = _ls_transform(upper, model.basis[nl:])
    upper_vectors = result.states['eigenvectors'][nl:]
    for column in range(upper_vectors.shape[1]):
        unc_weights = np.abs(Uu.T @ upper_vectors[:, column])**2
        if np.max(unc_weights) > .99:
            ml, ms = unc[int(np.argmax(unc_weights))]
            mask = result.upper_index == column
            # The lower S states have mJ=mS; spin-changing strengths vanish.
            forbidden = mask & (result.lower_two_Mtotal != ms)
            assert np.sum(result.strength_au[forbidden]) < 1e-10
    beta = model.constants.mu_B_cm*1e6
    residual_diagonal = np.diag(Uu.T @ model.H0[nl:, nl:] @ Uu).real
    slopes = np.array([(ml+model.constants.g_s*ms)/2 for ml, ms in unc])
    expected = np.sort(residual_diagonal + beta*slopes)
    got = np.linalg.eigvalsh(model.H0[nl:, nl:] + beta*model.T0[nl:, nl:])
    np.testing.assert_allclose(got, expected, atol=1e-5, rtol=0)


def test_field_tracking_and_isolated_non_e1_rates():
    results = A.calculate_field_grid(benchmark(), [0, .1, 1, 10])
    for result in results:
        assert len(set(result.states['state_id'])) == len(result.states['state_id'])
        np.testing.assert_array_equal(result.upper_state_id, result.states['state_id'][result.upper_index])
    k = A.AtomicConstants()
    u, l = A.AtomicLevel('u', 20000., 4, 1, g_J=1), A.AtomicLevel('l', 0., 0, 1, g_J=0)
    rates = A.isolated_multipole_rates(u, l, 0, 1e5, 20000*100*k.c, multipole='E2')
    for m in np.unique(rates['upper_two_M']):
        assert rates['A_s1'][rates['upper_two_M'] == m].sum() == pytest.approx(1e5)
    assert 'absolute strength' in rates['strength_unit']
    with pytest.raises(ValueError):
        A.isolated_multipole_rates(u, l, 0, 1e5, 20000*100*k.c, multipole='M1+E2')


def test_opt_in_driver_bridge_preserves_every_legacy_numeric_output():
    deck = dict(s_ground=.5, s_excited=.5, l_ground=0, l_excited=1,
                E_ground=[0.], E_excited=[20000., 20001.], Bmag=1, b_angle=37,
                amu=40, Convfxn='GaussianInstrum', Temp=300,
                spec_window=[499, 501], plot_window=[499, 501])
    before = A.AtomSpect_Main(dict(deck))
    calibrated = A.AtomSpect_Main(dict(deck, RadiativeData={'references': [dict(upper_two_J=1, lower_two_J=1,
                                      kind='A', value=1e8, wavelength_vac_nm=500)]}))
    assert set(calibrated) == set(before) | {'AtomicComponents'}
    def compare(a, b):
        if isinstance(a, (list, tuple)):
            for x, y in zip(a, b):
                compare(x, y)
        else:
            np.testing.assert_equal(a, b)
    for key in before:
        if key != 'Input':
            compare(before[key], calibrated[key])
    assert calibrated.AtomicComponents.model_id == 'B'
    assert 'AtomicComponents' in calibrated.describe()


@pytest.mark.parametrize('value', [.5, 1.00000001, -1, True])
def test_new_quantum_number_contract_rejects_rounding(value):
    with pytest.raises(ValueError):
        A.AtomicLevel('invalid', 0., value, 1)


def test_incomplete_terms_and_invalid_physical_inputs_fail_explicitly():
    with pytest.raises(A.MissingAtomicData):
        A.LSTerm('u', 1, 1, {1: 20000.}, -1).levels()
    with pytest.raises(ValueError):
        A.e1_rate(-1., 1.)
    model = benchmark()
    with pytest.raises(A.MissingAtomicData):
        A.add_hyperfine(model, A.HyperfineData(3, 1., {}, 'test'))
    with pytest.raises(ValueError):
        A.rephase_model(model, np.ones(len(model.basis))*2)
