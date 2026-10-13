"""Energy/wavelength unit conversions and vacuum<->air."""
from .constants import c_light


def Hz_to_nm(Ediff):
    # Converts from Hz to nm
    return 2.99792458E17 / Ediff


def nm_to_Hz(Ediff):
    # Converts from nm to Hz
    return 2.99792458E17 / Ediff


def ev_to_MHz(Ediff):
    # Converts from eV to MHz
    return (1e-6)*Ediff*2.41804e14


def cm_to_nm(E_diff):
    # Converts from energy in cm^-1 to wavelength in nm.
    return 1e7/E_diff


def cm_to_Hz(E_diff, MHz=False):
    # Converts from energy in cm^-1 to energy in Hz (E = hf = hc/lambda)
    if MHz:
        return 100*c_light*E_diff/1e6  # Return result in MHz.
    else:
        return 100*c_light*E_diff


def cm_to_ev(E_diff):
    # Convert from cm^-1 to eV
    return 1.2398e-4*E_diff


def ev_to_cm(E_diff):
    # Convert from eV to cm^-1
    return E_diff/(1.2398e-4)
# %%Wavelength


def Vac_to_air(wavein):
    '''
    Converts wavelength from vacuum to air by using the index of refaction.
    
    The calculation is based on Morton2000, so is done for "dry" air and
    1 atmosphere of presure, 14% C and 0.045%CO_2 by volume.
    
    Parameters
    ----------
    wavein : TYPE
        Wavelength in (assumed in nm)

    Returns
    -------
    TYPE
        Wavelength_out in nm

    '''

    # Using Morton2000
    wavevac_angs = wavein*10 #Convert to angstrom to match the paper
    #Define a common factor
    s = (1e4)/wavevac_angs
    #Calculate n_air
    n_air = 1 + 8.34254e-5 + (2.406147e-2)/(130 - s**2) + (1.5998e-4)/(38.9 - s**2)
    #Calculate the new wavelength based on the index of refraction
    wavelengths = wavein / n_air

    return wavelengths


def air_to_vac(wavein):
    '''
    Converts wavelength from vacuum to air, doing the opposite of Vac_to_air
    The calculation is based on Morton2000, so is done for "dry" air and
    1 atmosphere of presure, 14% C and 0.045%CO_2 by volume.
    
    Parameters
    ----------
    wavein : TYPE
        Wavelength in (assumed in nm)
    
    Returns
    -------
    TYPE
        Wavelength_out in nm

    '''

    # Using Morton2000
    wavevac_angs = wavein*10 #Convert to angstrom to match the paper
    #Define a common factor
    s = (1e4)/wavevac_angs
    #Calculate n_air
    n_air = 1 + 8.34254e-5 + (2.406147e-2)/(130 - s**2) + (1.5998e-4)/(38.9 - s**2)
    #Calculate the new wavelength based on the index of refraction
    wavelengths = wavein * n_air

    return wavelengths
