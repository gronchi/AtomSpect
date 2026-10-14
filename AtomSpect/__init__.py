"""Public AtomSpect API for relative Zeeman and hyperfine E1 spectra.

Import AtomSpect_Main, SpectrumResult, and the legacy calculation/plotting
helpers from this module. Implementations live in the corresponding package
modules; see doc/README.md for the manual, examples, and validity limits.
"""


from .constants import *  # noqa: F401,F403
from .units import *  # noqa: F401,F403
from .lineshapes import *  # noqa: F401,F403
from .convolution import *  # noqa: F401,F403
from .io import *  # noqa: F401,F403
from .angular import *  # noqa: F401,F403
from .hfs import *  # noqa: F401,F403
from .radiative import *  # noqa: F401,F403
from .polarization import *  # noqa: F401,F403
from .hamiltonian import *  # noqa: F401,F403
from .levels import *  # noqa: F401,F403
from .transitions import *  # noqa: F401,F403
from .fan import *  # noqa: F401,F403
from .polarization import _eigenpair_polarization as _eigenpair_polarization
from .hamiltonian import (
    _magnetic_eigenvectors as _magnetic_eigenvectors,
    _hfs_magnetic_eigenvectors as _hfs_magnetic_eigenvectors,
)
from .transitions import _zeeman_signal as _zeeman_signal
from .config import (
    InputDeck as InputDeck,
    _approximation_enabled as _approximation_enabled,
)
from .spectrum import AtomSpect_Main as AtomSpect_Main
from .result import SpectrumResult as SpectrumResult
from .atomic_data import (
    AtomicConstants as AtomicConstants,
    AtomicLevel as AtomicLevel,
    RadiativeReference as RadiativeReference,
    LSTerm as LSTerm,
    HyperfineData as HyperfineData,
    MissingAtomicData as MissingAtomicData,
    e1_rate as e1_rate,
    e1_line_strength as e1_line_strength,
    einstein_b as einstein_b,
    fit_ls_norm as fit_ls_norm,
)
from .atomic import (
    AtomicModel as AtomicModel,
    AtomicComponents as AtomicComponents,
    build_model_a as build_model_a,
    build_model_b as build_model_b,
    build_model_c as build_model_c,
    calculate_components as calculate_components,
    calculate_field_grid as calculate_field_grid,
    add_hyperfine as add_hyperfine,
    rephase_model as rephase_model,
    components_from_inputdeck as components_from_inputdeck,
    synthesize_spectrum as synthesize_spectrum,
    frequency_profile_to_wavelength as frequency_profile_to_wavelength,
    isolated_multipole_rates as isolated_multipole_rates,
)


# Plotting needs matplotlib (and tkinter for the slider dialogs). Load it on first use so that
# `import AtomSpect` works on headless machines and stays fast for batch calculations.
_PLOTTING_NAMES = frozenset({
    'plotarray', 'parse_plot_data', 'get_target_axis', 'consolidate_legend', 'PlotFunction',
    'plotZfan', 'Zslider', 'Zslider_Pol', 'enable_manual_input', 'MakeSlider',
})


def __getattr__(name):
    if name == 'plotting' or name in _PLOTTING_NAMES:
        import importlib
        plotting = importlib.import_module('.plotting', __name__)
        value = plotting if name == 'plotting' else getattr(plotting, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _PLOTTING_NAMES | {'plotting'})
