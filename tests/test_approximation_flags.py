"""Approximation switches using a synthetic spinless J=0 -> 1 line.

E=12500 cm^-1 places the line near 800 nm. Energies, field, linewidth and
windows are test choices, not reference atomic data. HFS disabled-path
checks use I=1/2 and zero A/B to isolate flag handling from other HFS fixes.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import AtomSpect


def deck():
    return dict(s_ground=0, s_excited=0, l_ground=0, l_excited=1,
        E_ground=np.array([0.]), E_excited=np.array([12500.]), Bmag=.1,
        b_angle=90, amu=40, plottitle='Synthetic flag test',
        spec_window=[799, 801], plot_window=[799, 801],
        specstep=.01, specres=1, fxnwindow=1.)


@pytest.mark.parametrize('value', [False, 0, None, 'N', 'No', 'False', 'false', '0', ''])
def test_disabled_flags_skip_solver_and_convolution(value, monkeypatch):
    inputs = deck()
    baseline = AtomSpect.AtomSpect_Main(dict(inputs))
    inputs.update(DoLowSig=value, DoHighSig=value)
    original = AtomSpect.Zeeman_func
    def guarded(*args, **kw):
        assert not kw.get('Lowfield') and not kw.get('Highfield')
        return original(*args, **kw)
    monkeypatch.setattr(AtomSpect.spectrum, 'Zeeman_func', guarded)
    result = AtomSpect.AtomSpect_Main(inputs)
    assert set(result) == set(baseline)
    np.testing.assert_array_equal(result['SpecOut'], baseline['SpecOut'])


@pytest.mark.parametrize('low,high', [(True, False), (False, True), (True, True)])
def test_true_flags_produce_only_requested_outputs(low, high):
    inputs = deck()
    inputs.update(DoLowSig=low, DoHighSig=high, EtermG=0., EtermE=12500.)
    result = AtomSpect.AtomSpect_Main(inputs)
    for enabled, suffix in [(low, 'low'), (high, 'high')]:
        assert ('signal_'+suffix in result) == enabled
        assert ('SpecOut'+suffix.title() in result) == enabled
        if enabled:
            assert np.isfinite(result['signal_'+suffix]).all()


@pytest.mark.parametrize('value', ['Y', 'Yes', 'TRUE', 1, np.bool_(True)])
def test_legacy_enabled_forms_match_boolean(value):
    inputs = dict(deck(), EtermG=0., EtermE=12500.)
    reference = AtomSpect.AtomSpect_Main(dict(inputs, DoLowSig=True, DoHighSig=True))
    result = AtomSpect.AtomSpect_Main(dict(inputs, DoLowSig=value, DoHighSig=value))
    np.testing.assert_array_equal(result['SpecOutLow'], reference['SpecOutLow'])
    np.testing.assert_array_equal(result['SpecOutHigh'], reference['SpecOutHigh'])


@pytest.mark.parametrize('value', [False, 'No', 'False'])
def test_hfs_disabled_flags_do_not_access_missing_outputs(value):
    inputs = dict(deck(), I_spin=.5, mu_I=1., HFS_G=[0.], HFS_E=[0.])
    result = AtomSpect.AtomSpect_Main(dict(inputs, DoLowSig=value, DoHighSig=value))
    assert 'SpecOutLow' not in result and 'SpecOutHigh' not in result


@pytest.mark.parametrize('flag', ['DoLowSig', 'DoHighSig'])
def test_hfs_enabled_approximations_fail_explicitly(flag):
    inputs = dict(deck(), I_spin=.5, mu_I=1., HFS_G=[0.], HFS_E=[0.])
    inputs[flag] = True
    with pytest.raises(NotImplementedError, match='HFS approximation spectra'):
        AtomSpect.AtomSpect_Main(inputs)


@pytest.mark.parametrize('value', ['maybe', 2, [True]])
def test_invalid_flag_is_rejected_before_computation(value):
    with pytest.raises(ValueError, match='Approximation flags'):
        AtomSpect.AtomSpect_Main(dict(deck(), DoLowSig=value))


def test_slider_with_false_flags_has_no_uninitialized_lines(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda *args, **kwargs: None)
    inputs = dict(deck(), DoLowSig=False, DoHighSig=False)
    AtomSpect.MakeSlider(inputs, [np.array([799., 801.]), np.array([0., 1.])],
                         [799., 801.], banglevary=0, polvary=0)
    plt.close('all')
