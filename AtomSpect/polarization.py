"""Polarization of E1 components and eigenstate-pair labelling."""
import numpy as np

from .constants import eps0


def polarization(mupper, mlower, Btheta, gamma=0, Pol_Filter=False, TAlpha=1, TBeta=1, pol1 = 1, pol2 = 0):
    '''
    This polarization calculation is generally based on Isler1997 http://dx.doi.org/10.1063/1.872095
    A correction is made with respect to a negative sign in the definition of epsilon plus/minus to be more in line with typical notation.
    This correction removes the issue of an extra negative sign for epsilon_+1-1 when calculated out and the author having the magnitude only.
    There is also a correction in the calculation to properly account for the relative pi/sigma intensity. See documentation for full details.
    Parameters
    ----------
    mupper : magnetic quantum number of upper level. mJ'
    mlower : magnetic quantum number of lower level, mJ
    Btheta : Angle between LoS and maximum magnetic field
    gamma : Angle polarization filter makes with respect to maximizing linear (pi) light transmission.
    Pol_Filter : Boolean, whether a polarization filter was in place for data collection
    T_Alpha : Transmission along one polarization axis. #Should be 1.0 without additional components
    T_Beta : Transmission coefficient along perpendicular polarization axis. Should be 1.0 without additional components

    Returns
    -------
    rad_tmp : Projection of polarization signal amplitude for emmitted light.

    '''

    if Pol_Filter == False:
        
        ealph = np.sqrt(TAlpha) #These are unity unless modified by an additional component
        ebeta = np.sqrt(TBeta) #These are unity unless modified by an additional component
        
    else:

        gammaangle = np.deg2rad(gamma)
        ealph = np.sqrt(TAlpha)*(pol1*np.cos(gammaangle) + pol2*np.sin(gammaangle))
        ebeta = np.sqrt(TBeta)*(pol1*np.sin(gammaangle) + pol2*np.cos(gammaangle))
        
    theta_rad = np.deg2rad(Btheta)  # convert deg to radians
    # print(ealph,ebeta)
    
    epsz = ealph*np.sin(theta_rad)

    e0 = epsz

    # eps0 is for a change in am of 0 (pi)
    eps0 = np.sqrt(e0*e0)

    # eps1 is the for a change in am of 1 (sigma)
    # eps1 = 0.5*np.sqrt(((np.cos(theta_rad)**2) * (ealph**2) + (ebeta**2)))  # While this looks like Isler appendix, we haven't squared it for the intensity calc yet.

    eps1 = np.sqrt(((np.cos(theta_rad)**2) * (ealph**2) + (ebeta**2))*0.5)  # Same as Isler appendix, we haven't squared it for the intensity calc yet.

    # print(polfact)
    # Calculate and check the change in magnetic angular momentum. circularly (sigma) light has a change of +/-1, linear (pi) is a change of 0.
    delta_am = np.array(mupper) - np.array(mlower)
    rad_tmp = [0, 3]
    # Using match case instead of a bunch of if statements.
    match delta_am:
        case 1:
            rad_tmp = [eps1, 1]
        case -1:
            rad_tmp = [eps1, -1]
        case 0:
            rad_tmp = [eps0, 0]
        case _:
            rad_tmp = [0, 3] #This allows tracking of what should be traditionaly forbidden transitions under dipole only
    return rad_tmp


def _eigenpair_polarization(afcf, basis_q, dipoles, lower_m, upper_m,
                          Bangle, gamma, Filter, T_Alpha, T_Beta):
    """Classify definite-M eigenpairs and check their spherical E1 amplitudes.

    q is M_upper - M_lower, matching polarization(). The public sigma-stick
    keys retain their existing emitted-light sign convention. Label 3 denotes
    a pair with no E1 amplitude, never a non-dipole emission component.
    """
    q_values = np.array([-1, 0, 1])
    spherical = np.asarray(dipoles)[:, None]*(np.asarray(basis_q)[:, None] == q_values)
    amplitudes = afcf.T @ spherical
    strengths = np.abs(amplitudes)**2
    delta_m = (upper_m[:, None] - lower_m[None, :]).ravel()
    q_labels = np.rint(delta_m).astype(int)
    if not np.allclose(delta_m, q_labels, rtol=0, atol=1e-8):
        raise ValueError("Polarization labels require integer differences in eigenstate M.")
    # The eigenvectors have definite M, so only the channel matching Delta M
    # can contribute. This guards the classification against basis errors.
    leakage = np.where(q_labels[:, None] == q_values, 0, strengths)
    if np.max(leakage) > 1e-12*np.max(strengths):
        raise ValueError("Spherical E1 amplitudes disagree with eigenstate M.")
    return [polarization(int(q), 0, Bangle, gamma, Pol_Filter=Filter,
                         TAlpha=T_Alpha, TBeta=T_Beta) for q in q_labels]
