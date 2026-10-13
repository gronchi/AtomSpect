"""Einstein-A style radiative outputs (experimental; see the audit notes)."""
import numpy as np

from .constants import c_light, eps0, hbar, hh, q_charge
from .units import nm_to_Hz


# For conversion factors, 1 = (cm^-1)/(1.2398e-4 eV) = (cm^-1)/(100*c*Hz) = (1.2398e-4 eV)/(100*c*Hz)


def A_Ein(E_in, dipolesig):
    '''
    Calculate the Einstein A coefficient using the dipole moment.
    Bransden QM 
    A_ab = (womega^3 * e*2)/(3\pi eps0hbar*c**3) * |<a|r|b>|^2

    There is also the intensity expression from Bethe and Salpeter - 59.10
    
    I = (4 q^2 omega^4)/(3 c^3) |r|^2 - again pol and spacially averaged.
    
    Compare to 59.11 for Einstein A as
    A = (4 q^2 omega^3)/(3 hbarc^3) |r|^2 - again pol and spacially averaged.
    
    These are in erg/sec units.
    There is definitely interesting physics and math to discover in these relations
    As well as why the expected expressions do not precisely yield the A coefficients.
    What is the missing component, and what is my calulation from the dipolestr actually giving?
    
    
    The e
    Parameters
    ----------
    E_in : float
        Energy in wavelength (nm) for the given line
    dipolesig : float
        Scalar of the dipole sig strength for the given transition.

    Returns
    -------
    tAein : TYPE
        DESCRIPTION.

    '''
 
    # omega = (Ea-Eb)/hbar
    tomega = nm_to_Hz(E_in)/(2*np.pi) #Note also the lack of hbar here, just h.
    #This expression looks close to eq 4.119 in Bransden Atoms (Pg190/182)
    tAein = ((tomega**2 * q_charge**2)/(np.pi*eps0*hh*c_light**3)) * dipolesig  # Not exactly as given by papers, but works?
   
    return tAein


def A_Ein2(E_in, dipolesig):
    '''
    This is done explicitly as written without the change to the factoring.
    Calculate the Einstein A coefficient using the dipole moment.
    Bransden QM 
    A_ab = (womega^3 * e*2)/(3\pi eps0hbar*c**3) * |<a|r|b>|^2

    There is also the intensity expression from Bethe and Salpeter - 59.10
    
    I = (4 q^2 omega^4)/(3 c^3) |r|^2 - again pol and spacially averaged.
    
    Compare to 59.11 for Einstein A as
    A = (4 q^2 omega^3)/(3 hbarc^3) |r|^2 - again pol and spacially averaged.
    
    These are in erg/sec units.
    There is definitely interesting physics and math to discover in these relations
    As well as why the expected expressions do not precisely yield the A coefficients.
    What is the missing component, and what is my calulation from the dipolestr actually giving?
    
    
    The e
    Parameters
    ----------
    E_in : float
        Energy in wavelength (nm) for the given line
    dipolesig : float
        Scalar of the dipole sig strength for the given transition.

    Returns
    -------
    tAein : TYPE
        DESCRIPTION.

    '''
 
    # omega = (Ea-Eb)/hbar
    tomega = nm_to_Hz(E_in)*(2*np.pi) #Note also the
    #This expression looks close to eq 4.119 in Bransden Atoms (Pg190/182)
    tAein = 4*((tomega**3 * q_charge**2)/(np.pi*eps0*hbar*c_light**3)) * dipolesig  # Not exactly as given by papers, but works?
   
    return tAein
