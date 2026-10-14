"""Einstein-A style radiative outputs (experimental; see the audit notes)."""
import numpy as np

from .constants import c_light, eps0, hbar, hh, q_charge
from .units import nm_to_Hz


# For conversion factors, 1 = (cm^-1)/(1.2398e-4 eV) = (cm^-1)/(100*c*Hz) = (1.2398e-4 eV)/(100*c*Hz)


def A_Ein(E_in, dipolesig):
    """Return the legacy radiative placeholder for wavelength E_in in nm.
    
    The implementation multiplies dipolesig by a frequency-squared factor.
    A normalized angular strength omits the radial matrix element, so this
    value is not a physical absolute Einstein A coefficient or rate."""
 
    # omega = (Ea-Eb)/hbar
    tomega = nm_to_Hz(E_in)/(2*np.pi) #Note also the lack of hbar here, just h.
    #This expression looks close to eq 4.119 in Bransden Atoms (Pg190/182)
    tAein = ((tomega**2 * q_charge**2)/(np.pi*eps0*hh*c_light**3)) * dipolesig  # Not exactly as given by papers, but works?
   
    return tAein


def A_Ein2(E_in, dipolesig):
    """Return the alternate legacy frequency-cubed radiative placeholder.
    
    E_in is wavelength in nm; dipolesig is the supplied angular strength.
    The prefactor and missing radial matrix element prevent interpreting
    the result as a validated absolute Einstein coefficient."""
 
    # omega = (Ea-Eb)/hbar
    tomega = nm_to_Hz(E_in)*(2*np.pi) #Note also the
    #This expression looks close to eq 4.119 in Bransden Atoms (Pg190/182)
    tAein = 4*((tomega**3 * q_charge**2)/(np.pi*eps0*hbar*c_light**3)) * dipolesig  # Not exactly as given by papers, but works?
   
    return tAein
