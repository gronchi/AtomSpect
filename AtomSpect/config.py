"""Validated view of the input dictionary used by AtomSpect_Main."""
import difflib
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

REQUIRED_KEYS = ('s_ground', 's_excited', 'l_ground', 'l_excited',
                 'E_ground', 'E_excited', 'Bmag', 'b_angle')

OPTIONAL_KEYS = (
    'Pol_angle', 'TAlpha', 'TBeta', 'bidir', 'sortE', 'amu', 'SpectrumData',
    'spec_window', 'plot_window', 'EtermG', 'EtermE', 'I_spin', 'mu_I',
    'HFS_G', 'HFS_E', 'HFSB_G', 'HFSB_E', 'PsiHFS', 'Calc_HFS_Vals',
    'DoLowSig', 'DoHighSig', 'ion_vel', 'Convfxn', 'Skewness', 'Temp', 'TempeV',
    'specstep', 'specres', 'fxnwindow', 'plottitle',
    'RadiativePlaceholders',
)
KNOWN_KEYS = frozenset(REQUIRED_KEYS + OPTIONAL_KEYS)

EV_TO_K = 11602  # Conversion used for the TempeV keyword.
_TEMPERATURE_PROFILES = ('Gaussian', 'GaussianInstrum', 'Voigt')
WING_LOSS_WARNING = 0.02  # Warn when truncating the Lorentzian wings drops more than 2% of a line's area.


def _approximation_enabled(value):
    """Read Boolean flags and legacy yes/no strings consistently."""
    if value is None:
        return False
    if isinstance(value, str):
        answer = value.strip().lower()
        if answer in ('y', 'yes', 'true', '1'):
            return True
        if answer in ('n', 'no', 'false', '0', ''):
            return False
        raise ValueError('Approximation flags require Boolean or yes/no values')
    if np.isscalar(value) and value in (0, 1):
        return bool(value)
    raise ValueError('Approximation flags require Boolean or yes/no values')


@dataclass
class ConvolutionSettings:
    """How the stick spectrum is turned into a curve."""
    function: str
    temperature: float
    skewness: Any
    stepsize: float
    window: float
    ion_vel: Any
    specres: float


def lorentzian_wing_loss(window, stepsize, skewness=(1, 1)):
    """Fraction of a (split) Lorentzian's area lying beyond +-window of its centre.

    The half widths are skewness*stepsize/2 (stepsize is the FWHM); all lengths in nm.
    """
    left, right = (k*stepsize/2 for k in skewness)
    tail = sum(w*(math.pi/2 - math.atan(window/w)) for w in (left, right))
    return tail/((math.pi/2)*(left + right))


def _warn_if_wings_truncated(settings):
    if settings.function in ('Skewed', 'Lorrentzian', 'Voigt'):
        skew = settings.skewness if settings.function == 'Skewed' else (1, 1)
        loss = lorentzian_wing_loss(settings.window, settings.stepsize, skew)
        if loss > WING_LOSS_WARNING:
            warnings.warn(f"fxnwindow={settings.window} nm truncates about {100*loss:.1f}% of each "
                          f"{settings.function} line's area; increase 'fxnwindow' to keep the wings.",
                          UserWarning, stacklevel=4)


@dataclass
class InputDeck:
    """Typed, defaulted view of the input dictionary.

    The original dictionary stays available as `raw` and is what the driver
    returns under data['Input'], so downstream code that reads it is unaffected.
    """
    raw: dict
    S_ground: Any
    S_excited: Any
    L_ground: Any
    L_excited: Any
    E_ground: Any
    E_excited: Any
    Bfield: Any
    Bangle: Any
    do_low: bool = False
    do_high: bool = False
    Polangle: float = 0
    Polfilter: bool = False
    Talpha: float = 1
    Tbeta: float = 1
    bidir: bool = False
    sortE: bool = False
    EtermG: Optional[float] = None
    EtermE: Optional[float] = None
    spec_window: Any = None
    plot_window: Any = None
    is_hfs: bool = False
    I_spin: Any = None
    mu_I: Any = None
    HFS_ConstsG: dict = field(default_factory=dict)
    HFS_ConstsE: dict = field(default_factory=dict)
    PsiHFS: Any = False
    radiative_placeholders: bool = False
    convolution: Optional['ConvolutionSettings'] = None

    @classmethod
    def from_dict(cls, deck):
        """Parse and validate `deck`.

        Missing required keys raise KeyError, malformed approximation flags
        raise ValueError, and unknown keys (usually typos) raise a warning.
        """
        missing = [k for k in REQUIRED_KEYS if k not in deck]
        if missing:
            raise KeyError('Input deck is missing required key(s): ' + ', '.join(missing))
        for key in deck:
            if key not in KNOWN_KEYS:
                close = difflib.get_close_matches(key, KNOWN_KEYS, n=1)
                hint = f" (did you mean '{close[0]}'?)" if close else ''
                warnings.warn(f"Unknown input key '{key}' is ignored{hint}", UserWarning, stacklevel=3)

        do_low = _approximation_enabled(deck.get('DoLowSig', False))
        do_high = _approximation_enabled(deck.get('DoHighSig', False))
        is_hfs = 'I_spin' in deck
        if is_hfs and (do_low or do_high):
            raise NotImplementedError('HFS approximation spectra are not implemented; omit or disable DoLowSig/DoHighSig')

        if 'amu' not in deck:
            deck['amu'] = 1  # Default so that it is not required; visible in data['Input'].

        # Resolve windows: deck value > spectrometer range > default.
        spec_data = deck.get('SpectrumData')
        data_range = [np.min(spec_data[0]), np.max(spec_data[0])] if spec_data is not None else [1, 1000]

        out = cls(
            raw=deck,
            S_ground=deck['s_ground'], S_excited=deck['s_excited'],
            L_ground=deck['l_ground'], L_excited=deck['l_excited'],
            E_ground=deck['E_ground'], E_excited=deck['E_excited'],
            Bfield=deck['Bmag'], Bangle=deck['b_angle'],
            do_low=do_low, do_high=do_high,
            Polangle=deck.get('Pol_angle', 0), Polfilter='Pol_angle' in deck,
            Talpha=deck.get('TAlpha', 1), Tbeta=deck.get('TBeta', 1),
            bidir=deck.get('bidir', False), sortE=deck.get('sortE', False),
            EtermG=deck.get('EtermG'), EtermE=deck.get('EtermE'),
            spec_window=deck.get('spec_window', data_range),
            plot_window=deck.get('plot_window', data_range),
            is_hfs=is_hfs,
        )
        if is_hfs:
            out.I_spin = deck['I_spin']
            out.mu_I = deck['mu_I']
            out.HFS_ConstsG = {'HFSA': deck['HFS_G'],
                               'HFSB': deck['HFSB_G'] if 'HFSB_G' in deck else np.zeros_like(out.E_ground)}
            out.HFS_ConstsE = {'HFSA': deck['HFS_E'],
                               'HFSB': deck['HFSB_E'] if 'HFSB_E' in deck else np.zeros_like(out.E_excited)}
            out.PsiHFS = deck.get('PsiHFS', False)
        out.radiative_placeholders = _approximation_enabled(deck.get('RadiativePlaceholders', False))
        # Resolve the convolution now so that a missing temperature fails before the solve.
        out.convolution = out.convolution_settings()
        settings = out.convolution
        for name, value in (('specstep', settings.stepsize), ('specres', settings.specres),
                            ('fxnwindow', settings.window), ('amu', deck['amu'])):
            if not np.all(np.isfinite(value)) or np.any(np.asarray(value) <= 0):
                raise ValueError(f'{name} must be finite and positive')
        for name, value in (('TAlpha', out.Talpha), ('TBeta', out.Tbeta)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if settings.function in _TEMPERATURE_PROFILES:
            temp = np.asarray(settings.temperature)
            if not np.all(np.isfinite(temp)) or np.any(temp < 0) or (settings.function == 'Gaussian' and np.any(temp == 0)):
                raise ValueError('Temperature must be finite and nonnegative (positive for Gaussian)')
        if settings.function == 'Skewed':
            skew = np.asarray(settings.skewness)
            if skew.shape != (2,) or not np.all(np.isfinite(skew)) or np.any(skew <= 0):
                raise ValueError('Skewness requires two finite positive factors')
        _warn_if_wings_truncated(out.convolution)
        return out

    def convolution_settings(self):
        """Resolve the profile, temperature, widths and step for the convolution.

        Without 'Convfxn' the default Lorentzian is used (it does not depend on
        the temperature). A temperature ('Temp' in K or 'TempeV' in eV) is
        required by the Gaussian, GaussianInstrum and Voigt profiles.
        """
        deck = self.raw
        function = deck.get('Convfxn', 'Lorrentzian')
        temperature = 300
        if function in _TEMPERATURE_PROFILES:
            if 'TempeV' in deck:  # Allow the temperature in eV as well.
                temperature = deck['TempeV']*EV_TO_K
            elif 'Temp' in deck:
                temperature = deck['Temp']
            else:
                raise KeyError(f"Convfxn '{function}' requires 'Temp' (K) or 'TempeV' (eV) in the input deck")
        skewness = deck.get('Skewness', [1, 1]) if function in ('Skewed', 'Lorrentzian') else [1, 1]
        default_step = 0.002 if 'Convfxn' in deck else 0.002222
        return ConvolutionSettings(function=function, temperature=temperature, skewness=skewness,
                                   stepsize=deck.get('specstep', default_step), window=deck.get('fxnwindow', 0.1),
                                   ion_vel=deck.get('ion_vel', 0), specres=deck.get('specres', 10))
