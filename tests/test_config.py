"""InputDeck parsing, defaults and key validation."""
import os
import warnings

os.environ.setdefault("MPMATH_NOGMPY", "1")
os.environ.setdefault("SYMPY_GROUND_TYPES", "python")
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
import AtomSpect
from AtomSpect.config import InputDeck


def deck(**extra):
    base = dict(s_ground=0, s_excited=0, l_ground=0, l_excited=1,
                E_ground=np.array([0.]), E_excited=np.array([12500.]),
                Bmag=.1, b_angle=90)
    base.update(extra)
    return base


def test_missing_required_keys_named():
    d = deck()
    del d['Bmag'], d['b_angle']
    with pytest.raises(KeyError, match="Bmag, b_angle"):
        InputDeck.from_dict(d)


def test_unknown_key_warns_with_suggestion():
    with pytest.warns(UserWarning, match="'Polangle'.*did you mean 'Pol_angle'"):
        InputDeck.from_dict(deck(Polangle=30))


def test_known_keys_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        InputDeck.from_dict(deck(Pol_angle=10, Temp=300, Convfxn='Gaussian', DoLowSig=True))


def test_defaults_and_input_passthrough():
    raw = deck()
    parsed = InputDeck.from_dict(raw)
    assert raw['amu'] == 1 and parsed.raw is raw
    assert (parsed.Polfilter, parsed.Talpha, parsed.Tbeta, parsed.bidir) == (False, 1, 1, False)
    assert list(parsed.spec_window) == [1, 1000]


def test_pol_angle_enables_filter():
    parsed = InputDeck.from_dict(deck(Pol_angle=0))
    assert parsed.Polfilter and parsed.Polangle == 0


def test_hfs_constants_default_b_to_zero():
    parsed = InputDeck.from_dict(deck(I_spin=1.5, mu_I=1., HFS_G=[1e6], HFS_E=[1e6]))
    assert parsed.is_hfs and not parsed.HFS_ConstsG['HFSB'].any()


def test_hfs_rejects_approximation_spectra():
    with pytest.raises(NotImplementedError):
        InputDeck.from_dict(deck(I_spin=1.5, mu_I=1., HFS_G=[1e6], HFS_E=[1e6], DoLowSig=True))


def test_convolution_defaults_without_convfxn():
    s = InputDeck.from_dict(deck(Temp=500, ion_vel=1e3, Skewness=[2, 2])).convolution_settings()
    # The default Lorentzian is temperature independent but honours the other keys.
    assert (s.function, s.temperature, s.ion_vel, s.skewness) == ('Lorrentzian', 300, 1e3, [2, 2])
    assert s.stepsize == 0.002222 and s.window == 0.1 and s.specres == 10


def test_convolution_settings_with_profile():
    s = InputDeck.from_dict(deck(Convfxn='Voigt', TempeV=2., specstep=.01)).convolution_settings()
    assert s.function == 'Voigt' and s.temperature == pytest.approx(2*11602) and s.stepsize == .01
    s = InputDeck.from_dict(deck(Convfxn='Skewed', Skewness=[.5, 1.])).convolution_settings()
    assert s.skewness == [.5, 1.] and s.temperature == 300 and s.stepsize == 0.002
    # A missing temperature fails while the deck is parsed, before any solve.
    for profile in ('Gaussian', 'GaussianInstrum', 'Voigt'):
        with pytest.raises(KeyError, match="requires 'Temp'"):
            InputDeck.from_dict(deck(Convfxn=profile))


def test_main_warns_on_misspelled_key():
    with pytest.warns(UserWarning, match="Unknown input key 'Polfilter'"):
        AtomSpect.AtomSpect_Main(deck(Polfilter=True, spec_window=[799, 801], plot_window=[799, 801],
                                      specstep=.01, specres=1))


def test_wing_truncation_warning():
    from AtomSpect.config import lorentzian_wing_loss
    assert lorentzian_wing_loss(1e9, 0.002) == pytest.approx(0, abs=1e-9)
    assert lorentzian_wing_loss(0.001, 0.002) == pytest.approx(1 - 2/np.pi*np.arctan(1), rel=1e-12)
    with pytest.warns(UserWarning, match="fxnwindow=0.005 nm truncates"):
        InputDeck.from_dict(deck(Convfxn='Lorrentzian', specstep=.002, fxnwindow=.005))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        InputDeck.from_dict(deck(Convfxn='Lorrentzian', specstep=.002, fxnwindow=.5))
        InputDeck.from_dict(deck(Convfxn='Gaussian', Temp=300, specstep=.002, fxnwindow=.005))


def test_radiative_placeholders_are_opt_in():
    flags = dict(spec_window=[799, 801], plot_window=[799, 801], specstep=.01, specres=1)
    plain = AtomSpect.AtomSpect_Main(deck(**flags))
    assert not {'A_Ein', 'A_Ein_polavg', 'sigavgs', 'f_osc', 'S_line'} & set(plain)
    with pytest.warns(UserWarning, match="RadiativePlaceholders"):
        extra = AtomSpect.AtomSpect_Main(deck(RadiativePlaceholders=True, **flags))
    assert {'A_Ein', 'A_Ein_polavg', 'sigavgs', 'f_osc', 'S_line'} <= set(extra)
