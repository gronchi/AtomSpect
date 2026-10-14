"""Edge cases and independent invariants found during the October 2026 review."""
import os

os.environ.setdefault('MPMATH_NOGMPY', '1')
os.environ.setdefault('SYMPY_GROUND_TYPES', 'python')

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

import AtomSpect as A
from AtomSpect.config import InputDeck
from AtomSpect.plotting.slider import Zslider, Zslider_Pol


def deck(**extra):
    values = dict(s_ground=0, s_excited=0, l_ground=0, l_excited=1,
                  E_ground=[0.], E_excited=[12500.], Bmag=.1, b_angle=90,
                  spec_window=[799, 801], plot_window=[799, 801])
    return dict(values, **extra)


def test_empty_stick_arrays_and_python_lists():
    spec, sticks, bins = A.Convol_Spect([], [], [500, 700], [590, 610], 40, .002)
    np.testing.assert_array_equal(spec, [[590, 610], [0, 0]])
    assert len(sticks[0]) == 0 and bins == []
    spec, _, _ = A.Convol_Spect([600], [1], [500, 700], [599, 601], [40], .002)
    assert np.max(spec[1]) > 0


def test_overlapping_bins_use_each_stick_once_and_grid_is_ordered():
    waves = np.array([600., 600.15, 600.16])
    result = A.make_stickbins(waves, np.arange(3), windsize=.1)
    assert sum(map(len, result['binned_waves'])) == len(waves)
    assert len(result['bin_windows']) == 1
    spectrum, _, _ = A.Convol_Spect(waves, np.ones(3), [599, 601], [599, 601], 40, .002)
    assert np.all(np.diff(spectrum[0]) >= 0)
    expected = sum(A.Lorrentzian(spectrum[0][1:-1]*1e-9, w*1e-9, pixel_size=.002e-9)
                   for w in waves)
    np.testing.assert_allclose(spectrum[1][1:-1], expected)


def test_integer_wavelengths_preserve_fractional_metadata():
    args = ([600], [1.], [599, 601], [599, 601])
    integer = A.Convol_Spect(*args, 1.5, .002, Temperature_in=.5, functiontype='Gaussian')
    floating = A.Convol_Spect(np.array([600.]), *args[1:], 1.5, .002,
                            Temperature_in=.5, functiontype='Gaussian')
    np.testing.assert_allclose(integer[0], floating[0])


def test_coarse_sampling_retains_two_points():
    spec = A.Convol_Sticks(600, [599.99, 600.01], 300, 40, 1, specres=1)
    assert len(spec[0]) == 2


@pytest.mark.parametrize('width,resolution', [(0, 1), (-1, 1), (np.nan, 1), (1, 0)])
def test_invalid_sampling_raises(width, resolution):
    with pytest.raises(ValueError):
        A.Convol_Sticks(600, [599, 601], 300, 40, width, specres=resolution)


def test_multispec_ev_temperature_and_scale_count():
    obj = dict(wave_air=np.array([600]), signal=np.array([1.]), Input={'TempeV': .5, 'amu': 1.5})
    _, strengths, temps, masses = A.MultiSpec([obj], [2])
    np.testing.assert_array_equal(strengths, [2])
    np.testing.assert_array_equal(temps, [5801])
    np.testing.assert_array_equal(masses, [1.5])
    with pytest.raises(ValueError, match='one scale'):
        A.MultiSpec([obj], [])


@pytest.mark.parametrize('slider', [Zslider, Zslider_Pol])
def test_slider_updates_existing_ev_temperature(slider):
    raw = deck(Convfxn='Gaussian', TempeV=20)
    if slider is Zslider:
        result = slider(.1, .1, .002, raw)
    else:
        result = slider(.1, .1, .002, 90, 0, raw)
    assert InputDeck.from_dict(result['Input']).convolution.temperature == pytest.approx(1160.2)


def test_dark_observation_returns_zero_strengths():
    result = A.AtomSpect_Main(deck(TAlpha=0, TBeta=0))
    assert np.all(result.signal == 0)
    assert np.all(result.SpecOut[1] == 0)


def test_numpy_spectrum_save_and_load_with_extension(tmp_path):
    path = tmp_path / 'saved.npy'
    spectrum = np.array([[600., 601.], [2., 4.]])
    A.Savenpy_Dict(path, {'amu': 40}, spectra=spectrum, Normalize_sig=True)
    data, loaded = A.Loadnpy_Dict(path)
    assert data == {'amu': 40}
    np.testing.assert_array_equal(loaded['wavelength'], spectrum[0])
    np.testing.assert_array_equal(loaded['signals'], [.5, 1])
    assert A.Normalize([]).size == 0


def test_single_selected_fan_needs_only_its_term_energy():
    result = A.ZeemanFan(deck(Bmag=np.array([0., 1.]), EtermG=0), 'GH')
    np.testing.assert_array_equal(result['HighG'], [[0., 0.]])
    with pytest.raises(ValueError, match='EtermG'):
        A.ZeemanFan(deck(Bmag=np.array([0., 1.])), 'GH')


def test_hfs_scalar_quadrupole_matches_per_j_constants():
    args = (1, .5, 1.5)
    scalar = A.stateprocess_HFS(*args, {'HFSA': 1e6, 'HFSB': 2e6}, 0, [0, 100], .1)
    vector = A.stateprocess_HFS(*args, {'HFSA': [1e6, 1e6], 'HFSB': [2e6, 2e6]},
                               0, [0, 100], .1)
    np.testing.assert_allclose(scalar[2], vector[2])
    with pytest.raises(ValueError, match='HFSA'):
        A.stateprocess_HFS(*args, {'HFSA': [1e6]}, 0, [0, 100], .1)


def test_wrong_number_of_level_energies_rejected():
    with pytest.raises(ValueError, match='3 finite energies'):
        A.stateprocess(1, 1, [0], .1)
    assert len(A.make_Jmj(np.float64(.5))) == 2
    assert A.lande_gi(0, 0) == 0


def test_weak_hfs_nuclear_sign_matches_exact_hamiltonian_slope():
    # Directly project the exact Zeeman operator onto one coupled |F,mF> state.
    gi = A.lande_gi(1.5, 1)
    level = A.stateprocess_HFS(0, .5, 1.5, {'HFSA': [1e6]}, gi, [0], 0)
    operator = A.HZeeman_HFS(level, gi)['H_Z']
    for index, state in enumerate(level[-1]):
        F, mF, J, _, I, _, L, _, S, _ = state
        assert A.lande_gf(F, I, J, L, S, gi)*mF == pytest.approx(operator[index, index], abs=1e-12)


def test_ground_splits_are_energies_and_approximations_preserve_exact_basis():
    exact = A.AtomSpect_Main(deck())
    np.testing.assert_array_equal(exact.ZeemanSplits_G, exact.EigGround[0])
    raw = dict(s_ground=1, s_excited=1, l_ground=1, l_excited=2,
               E_ground=[100, 200, 300], E_excited=[20100, 20200, 20300],
               Bmag=.5, b_angle=90, DoLowSig=True, sortE=True)
    approximate = A.AtomSpect_Main(raw)
    exact = A.AtomSpect_Main(dict(raw, DoLowSig=False))
    np.testing.assert_array_equal(approximate.statelist_Ground, exact.statelist_Ground)
    np.testing.assert_array_equal(approximate.E0_Ground, exact.E0_Ground)


def test_plot_accepts_numpy_bounds_and_does_not_clip_raw_intensities():
    from matplotlib import pyplot as plt
    axes = A.PlotFunction([[600, 601], [10, 20]], ['C0', 1, '-'],
                          plotwind=np.array([599, 602]), plotpol=False, show=False)
    assert axes[0, 0].get_ylim()[1] > 20
    np.testing.assert_array_equal(axes[0, 0].get_xlim(), [599, 602])
    plt.close(axes[0, 0].figure)


def test_plot_shared_style_draws_all_spectra_and_honors_zero_scale():
    from matplotlib import pyplot as plt
    axes = A.PlotFunction([[[600, 601], [1, 2]], [[600, 601], [3, 4]]],
                          ['C0', 1, '-'], plotpol=False, show=False, marker_size=0,
                          SpectrumPlot=[[600, 601], [2, 4]], ScaleSpectrometer=0)
    assert len(axes[0, 0].lines) == 3
    np.testing.assert_array_equal(axes[0, 0].lines[0].get_ydata(), [0, 0])
    assert axes[0, 0].lines[1].get_markersize() == 0
    plt.close(axes[0, 0].figure)


@pytest.mark.parametrize('extra', [dict(specstep=0), dict(specres=0), dict(fxnwindow=-1),
                                   dict(amu=0), dict(TAlpha=-1), dict(TBeta=np.nan),
                                   dict(Convfxn='Gaussian', Temp=0),
                                   dict(Convfxn='Skewed', Skewness=[0, 1])])
def test_invalid_profile_settings_fail_during_parsing(extra):
    with pytest.raises(ValueError):
        InputDeck.from_dict(deck(**extra))


def test_slider_startup_allows_missing_title_and_numpy_bounds(monkeypatch):
    from matplotlib import pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda *args, **kwargs: None)
    A.MakeSlider(deck(), [[799., 801.], [0., 1.]], plotwind=np.array([799, 801]),
                 do_sticks=False, do_Polplot=False)
    np.testing.assert_array_equal(plt.gcf().axes[0].get_xlim(), [799, 801])
    plt.close('all')


def test_slider_blocks_in_a_script_and_not_in_a_console(monkeypatch):
    """The slider window must stay open when a plain script calls MakeSlider."""
    import sys
    from AtomSpect.plotting import slider

    seen = {}
    monkeypatch.setattr(slider.plt, 'show', lambda block=None: seen.setdefault('block', block))
    assert slider._script_should_block() is True  # pytest is a plain, non-interactive process
    monkeypatch.setattr(sys, 'ps1', '>>> ', raising=False)
    assert slider._script_should_block() is False
    monkeypatch.delattr(sys, 'ps1')

    deck = dict(s_ground=0, s_excited=0, l_ground=0, l_excited=1, E_ground=np.array([0.]),
                E_excited=np.array([12500.]), Bmag=.1, b_angle=90, amu=40, Temp=300,
                Convfxn='Gaussian', spec_window=[799, 801], plot_window=[799, 801], specstep=.01, specres=1)
    slider.MakeSlider(deck, [np.array([799., 801.]), np.array([0., 0.])], [799, 801])
    assert seen['block'] is True


def test_curve_is_not_forced_to_zero_at_a_window_edge_inside_a_line():
    """A broad line crossing the plot-window edge must keep its own value there."""
    waves = np.array([600.0])
    sig = np.array([1.0])
    window = [599.99, 600.20]          # the line centre sits next to the left edge
    (w, s), _, _ = A.Convol_Spect(waves, sig, [599, 601], window, 40, 0.01, functiontype="Lorrentzian",
                          wind_size=0.5, specres=5)
    assert w[0] == pytest.approx(599.99)
    # FWHM 0.01 nm, 0.01 nm from the centre: the Lorentzian is 1/(1+(0.01/0.005)^2) = 0.2 of the peak,
    # and the edge must carry (about) that value rather than 0.
    assert s[0] == pytest.approx(s[1]) and 0.15*np.max(s) < s[0] < 0.4*np.max(s)
    # An edge far from every line is still zero.
    (w2, s2), _, _ = A.Convol_Spect(waves, sig, [599, 601], [598.0, 602.0], 40, 0.01, functiontype="Lorrentzian",
                            wind_size=0.05, specres=5)
    assert s2[0] == 0 and s2[-1] == 0
    assert np.all(np.diff(w2) > 0)


def test_nonblocking_slider_returns_a_responsive_figure(monkeypatch):
    """block=False returns the figure; the kept sliders still drive the curve after the call."""
    import gc
    from AtomSpect.plotting import slider

    monkeypatch.setattr(slider.plt, 'show', lambda *args, **kwargs: None)
    deck = dict(s_ground=0, s_excited=0, l_ground=0, l_excited=1, E_ground=np.array([0.]),
                E_excited=np.array([12500.]), Bmag=.1, b_angle=90, amu=40, Temp=300,
                Convfxn='Gaussian', spec_window=[799, 801], plot_window=[799, 801], specstep=.01, specres=1)
    fig = slider.MakeSlider(deck, [np.array([799., 801.]), np.array([0., 0.])], [799, 801],
                            do_Polplot=0, banglevary=1, block=False)
    assert {'B', 'T', 'Wi', 'angle'} <= set(fig.sliders)
    gc.collect()
    curve = fig.axes[0].get_lines()[1]
    before = np.array(curve.get_ydata(), dtype=float).copy()
    fig.sliders['T'].set_val(25.0)    # much hotter: the normalized line must broaden
    after = np.array(curve.get_ydata(), dtype=float)
    assert np.sum(after > 0.5) > np.sum(before > 0.5)
    slider.plt.close(fig)
