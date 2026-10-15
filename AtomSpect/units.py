"""Energy/wavelength unit conversions and vacuum<->air."""
from .constants import c_light
import numpy as _np


def Hz_to_nm(Ediff):
    """Convert a frequency in Hz to vacuum wavelength in nm (scalar or NumPy array)."""
    return 2.99792458E17 / Ediff


def nm_to_Hz(Ediff):
    """Convert vacuum wavelength in nm to frequency in Hz (scalar or NumPy array)."""
    return 2.99792458E17 / Ediff


def ev_to_MHz(Ediff):
    """Convert an energy in eV to frequency in MHz using the legacy rounded constant."""
    return (1e-6)*Ediff*2.41804e14


def cm_to_nm(E_diff):
    """Convert an energy difference in cm^-1 to vacuum wavelength in nm."""
    return 1e7/E_diff


def cm_to_Hz(E_diff, MHz=False):
    """Convert wavenumber in cm^-1 to frequency in Hz, or MHz when MHz=True."""
    if MHz:
        return 100*c_light*E_diff/1e6  # Return result in MHz.
    else:
        return 100*c_light*E_diff


def cm_to_ev(E_diff):
    """Convert wavenumber in cm^-1 to eV using the legacy rounded constant."""
    return 1.2398e-4*E_diff


def ev_to_cm(E_diff):
    """Convert energy in eV to cm^-1 using the legacy rounded constant."""
    return E_diff/(1.2398e-4)
# %%Wavelength


def _air_refractive_index(wave_vac_nm, pressure_pa, temperature_c,
                          water_vapor_pressure_pa):
    """1993 revised Edlen formula, Steck Rb-87 D Line Data, Eq. (3)."""
    kappa_squared = (1000.0 / wave_vac_nm)**2
    dispersion = (8342.54 + 2406147.0 / (130.0 - kappa_squared)
                  + 15998.0 / (38.9 - kappa_squared))
    pressure_temperature = (
        pressure_pa / 96095.43
        * (1.0 + 1e-8 * (0.601 - 0.00972 * temperature_c) * pressure_pa)
        / (1.0 + 0.0036610 * temperature_c)
    )
    humidity = water_vapor_pressure_pa * (0.037345 - 0.000401 * kappa_squared)
    return 1.0 + (dispersion * pressure_temperature - humidity) * 1e-8


def Vac_to_air(wavein, *, pressure_pa=101325.0, temperature_c=20.0,
               water_vapor_pressure_pa=0.0):
    """Convert vacuum wavelengths in nm to air wavelengths in nm.
    
    Use lambda_air = lambda_vac / n with the 1993 revised Edlen formula in
    Steck, Rubidium 87 D Line Data, Eq. (3):
    https://steck.us/alkalidata/rubidium87numbers.pdf

    Pressure is in Pa, temperature in degrees Celsius, and water vapor
    partial pressure in Pa (not relative humidity). Defaults describe dry
    air at 1 atm and 20 C. Wavelengths may be scalars or arrays.
    The reference quotes a 3-sigma index uncertainty of 3e-8 at 350-650 nm
    under laboratory conditions; avoid the formula's poles elsewhere.
    """
    wave_vac = _np.asarray(wavein, dtype=float)
    return wave_vac / _air_refractive_index(
        wave_vac, pressure_pa, temperature_c, water_vapor_pressure_pa)


def air_to_vac(wavein, *, pressure_pa=101325.0, temperature_c=20.0,
               water_vapor_pressure_pa=0.0):
    """Convert air wavelengths in nm to vacuum wavelengths in nm.

    Invert Vac_to_air by fixed-point iteration, evaluating the refractive
    index at the vacuum wavelength. Atmospheric parameters, defaults, and
    validity limits are the same as Vac_to_air. Inputs may be scalars or
    arrays. Iteration is intended for laboratory air conditions.
    """
    wave_air = _np.asarray(wavein, dtype=float)
    wave_vac = wave_air
    for _ in range(5):
        wave_vac = wave_air * _air_refractive_index(
            wave_vac, pressure_pa, temperature_c, water_vapor_pressure_pa)
    return wave_vac
