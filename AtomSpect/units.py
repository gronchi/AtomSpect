"""Energy/wavelength unit conversions and vacuum<->air."""
from .constants import c_light


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


def Vac_to_air(wavein):
    """Convert vacuum wavelengths in nm to air wavelengths in nm.
    
    Evaluate the implemented Morton refractive-index expression at the
    vacuum wavelength. Inputs may be scalars or NumPy arrays. This empirical
    relation is not validated over arbitrary wavelengths; avoid its poles."""

    # Using Morton2000
    wavevac_angs = wavein*10 #Convert to angstrom to match the paper
    s = (1e4)/wavevac_angs
    n_air = 1 + 8.34254e-5 + (2.406147e-2)/(130 - s**2) + (1.5998e-4)/(38.9 - s**2)
    wavelengths = wavein / n_air

    return wavelengths


def air_to_vac(wavein):
    """Approximately convert air wavelengths in nm to vacuum wavelengths in nm.
    
    Evaluate the same refractive-index expression as Vac_to_air at the air
    wavelength, then multiply. This legacy approximation is not the exact
    numerical inverse of Vac_to_air. Inputs may be scalars or NumPy arrays."""

    # Using Morton2000
    wavevac_angs = wavein*10 #Convert to angstrom to match the paper
    s = (1e4)/wavevac_angs
    n_air = 1 + 8.34254e-5 + (2.406147e-2)/(130 - s**2) + (1.5998e-4)/(38.9 - s**2)
    wavelengths = wavein * n_air

    return wavelengths
