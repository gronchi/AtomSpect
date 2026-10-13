# -*- coding: utf-8 -*-
"""
AtomSpect main function definitions.
@author: Leo Nofs
This contains all of the functions for operating the code. This file is NOT run directly.
For details, updates, questions, bug reports, refer to the project github: https://github.com/Lnofs/AtomSpect


"""

import numpy as np
import scipy as scp
import csv
from matplotlib import pyplot as plt
from matplotlib import ticker
import tkinter as tk
from tkinter import simpledialog

from matplotlib.widgets import Button, Slider

from sympy.physics.quantum.cg import CG
from sympy.physics.wigner import wigner_3j as w3j, wigner_6j as w6j

from .constants import *  # noqa: F401,F403
from .units import *  # noqa: F401,F403
from .lineshapes import *  # noqa: F401,F403
from .convolution import *  # noqa: F401,F403
from .io import *  # noqa: F401,F403
from .plotting import *  # noqa: F401,F403


# %%Lande Factors
def lande_gi(I, u_I):
    '''
    Generates the lande g_i factor given the spin (I)and nuclear moment.
    All of these factors can be obtained from Stone (2005)
    https://doi.org/10.1016/j.adt.2005.04.001

    Parameters
    ----------
    I : Nuclear Spin (Integer or Half Integer)
        Nuclear Spin of the atom. 
    I_par : TYPE
        DESCRIPTION.
    u_I : TYPE
        DESCRIPTION.

    Returns
    -------
    thisgi : g_i
        The lande g_i factor for use in the zeeman shift.

    '''
    # Technically, this should be the reduced mass here.
    thisgi = (u_I/I)*(m_elec/m_prot)  # This is gi_prime and is used in zeeman calculations.
    return thisgi


def lande_gj2(J, L, S):
    # Lande gJ as derived in Griffiths and others. Comes from projection of angular momentum and averaging of L and S
    #This has one less step of calculation than lande_gj function but should be equivalent.
    if J == 0:
        return 0
    else:
        return (g_L*(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) + g_S*((J*(J+1) + S*(S+1) - L*(L+1))/(2*J*(J+1))))


def lande_gj(J, L, S):
    # Lande gJ as derived in Griffiths and others. Comes from projection of angular momentum and averaging of L and S
    if J == 0: #There is no lande effect if J=0 due to a singularity (1/J)
        return 0
    else:
        # Use the same electron g factors as the exact Hamiltonian (g_S = 2.002319).
        return lande_gj2(J, L, S)


def lande_gf(F, I, J, L, S, g_I):
    # Lande gF, comes from averaging I, L, and S similar to gJ.
    thisgj = lande_gj(J, L, S)
    #There is no lande effect if F=0, due to a singularity (1/F)
    if F == 0:
        return 0
    else:
        thisgf = thisgj*(F*(F+1) - I*(I+1) + J*(J+1)) / (2*F*(F+1)) + g_I*(F*(F+1) + I*(I+1) - J*(J+1)) / (2*F*(F+1))
        return thisgf


def Delta_E_HFS(I, J, F, A, B):
    #Calculate the shift in energy due to the hyperfine effect, where A and B are the hyperfine constants.
    #These are in units of Hz typically. Output is in units of cm^-1
    #1/(100*c_light) is the conversion factor that takes the Hz and results in cm^-1
    #See SteckRb for a good overview of hyperfine. Kopfermann is a very detailed source as well.
    
    # First need to define a constant K that will be used in the calculations
    K = F*(F+1) - I*(I+1) - J*(J+1)
    #Calculate A_hfs
    E_Ahfs = 0.5*A*K / (100*c_light)
    #Check for the cases when the B_hfs contribution must be zero (I=0,1/2, J=0,1/2, or if B=0 )
    if I == np.abs(0.5) or J == 0.5 or I == 0 or J == 0 or B == 0:
        E_Bhfs = 0
    else:
        #If none of those cases are true, calculate the shift from the quadropole term.
        E_Bhfs = B*((0.75*K*(K+1) - I*(I+1)*J*(J+1))/(2*I*(2*I - 1)*J*(2*J - 1)))/(100*c_light)
    return E_Ahfs + E_Bhfs


# For conversion factors, 1 = (cm^-1)/(1.2398e-4 eV) = (cm^-1)/(100*c*Hz) = (1.2398e-4 eV)/(100*c*Hz)

# %%A_Ein


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

# %%Utils


def make_Jmj(J_am):
    # This function takes an input general angular momentum J and creates a list of [J,mJ] for all corresponding mJ's.
    # J_am can be an array itself and the output will be a flat list of all possible J,mJ values, starting with the lowest [J,mj] and
    # cycling through the mJ's for each J.
    # =============================================================================
    #
    #     'Starting with the lowest J has the potential to cause some issues and there should, instead, be a generalized way for the user to input the J-values ordering'
    #     'The issue would be because the eigval function sorts in ascending order of energies, so I should do likewise, as that may correct for potential issues'
    #     'with energy ordering and states as tracking past diagonalization is not possible.'
    # =============================================================================
    ###
    if type(J_am) == int or type(J_am) == float:
        Jtmp = [[float(J_am), float(x)] for x in np.linspace(-J_am, J_am, int(2*J_am+1))]
    else:
        Jtmp = [[float(jvs), float(x)] for jvs in J_am for x in np.linspace(-jvs, jvs, int(2*jvs+1))]
    return Jtmp


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


def CGCoeff(j1, m1, j2, m2, j3, m3):
    # Calculates Clebsch-Gordan Coefficient and outputs a float instead of sympy result
    thisCG = float(CG(j1, m1, j2, m2, j3, m3).doit())
    return thisCG


def Zeemanlow(gf, mf, B):
    # Lowfield Zeeman approximation. Valid for J or F total AM.
    return (muBcm*gf*mf*B)


def ZeemanHigh(ms, ml, B):
    # Highfield Zeeman approximation, excluding Hyperfine splitting. Energy is still diagonal.
    return muBcm*B*(2*ms + ml)


def ZeemanhighHFS(state_in, Ahfs, g_I, Bmag, Bhfs=0):
    """Hyperfine Paschen-Back shift in cm^-1, with A and B in Hz.

    The basis is |J,mJ;I,mI>: I and J decouple, but L and S remain
    coupled. This approximation requires Zeeman energies above the HFS
    splitting and below the fine-structure splitting. See Steck, Rb85
    D Line Data, Eq. (25), https://steck.us/alkalidata/rubidium85numbers.pdf.
    g_I follows lande_gi's magnetic-moment convention, hence the minus sign.
    """
    J, mJ, I, mI = state_in[2:6]
    L, S = state_in[6], state_in[8]
    hyperfine_hz = Ahfs*mJ*mI
    if I >= 1 and J >= 1:
        quadrupole = ((3*mJ**2 - J*(J+1))*(3*mI**2 - I*(I+1))
                      / (4*J*(2*J-1)*I*(2*I-1)))
        hyperfine_hz += Bhfs*quadrupole
    return (hyperfine_hz/(100*c_light)
            + muBcm*Bmag*(lande_gj2(J, L, S)*mJ - g_I*mI))


def CGw3(j1, m1, j2, m2, j3, m3):
    # Clebsh Gordon Coeff using wigner3j
    return ((-1)**(-j1 + j2 - m3))*np.sqrt(2*j3 + 1)*float(w3j(float(j1), float(j2), float(j3), float(m1), float(m2), -float(m3)))


# %%Zeeman
def dipolestr(LG, LE):
    '''
    Wigner3j = (J'    J    1  )
               (-m_j' m_j  q  )   
    Wigner6j = {S    L'  J'  }
               {1    J    L  }                   

    Parameters
    ----------
    LG, LE are arrays of quantum numbers. Formatted as LG=[J,mJ,L,mL,S,mS]

    Returns
    -------
    tdp3 : float
        Returns the relative dipole strength factor as a multiple of the reduced dipole matrix. Metcalf 4.32

    '''
    thisq = LE[1] - LG[1]
    # if thisq == -1 or thisq== 0 or thisq== 1:

    tw3 = float(w3j(LG[0], 1, LE[0], LG[1], thisq, -LE[1]))
    tdp3 = ((-1)**(LE[2] + LG[4] - LE[1])) * np.sqrt((2*LE[0]+1)*(2*LG[0]+1))*tw3*float(w6j(LE[2], LE[0], LG[4], LG[0], LG[2], 1))

    # else:
    #     tdp3 = 0
    return tdp3


def stateprocess(L_state, S_state, E_level, thisB, sortE=False):
    '''
    Takes input state values and magnetic field to expand the state space. 
    L and S are used to calculate possible J values. all possible mJ values are also created. 
    E_level is an np.array and must match the length of possible J values. Since the values are pulled from NIST, this will likely happen.


    Parameters
    ----------
    L_state : Int or half int
        Orbital angular quantum number. S=0, P=1, D=2, F=3, etc
    S_state : Int or half int
        Spin quantum number, n = 2s+1 for multiplicity, so ^2S_J has (2*s +1) =2, so s=0.5, L=0
    E_level : np.array
        Energy in cm^-1, Lowest J value first.
    thisB : 
        Magnetic field in T

    Returns
    -------
    list
       [w3mat, Jmjs, LS,E_low, E_cm,E_SO_cm_Mat,sortedstates]
    w3mat is the state array of w3 coefficients corresponding to CG coefficients.
    Jmjs are the possible Jmjs #Deprecated
    LS is the L,mL,S,mS states. #Deprecated
    E_low is a list of energies corresponding to the lowfield zeeman approx
    E_cm is the sorted energies in cm^-1
    E_SO_cm_Mat is the sorted (diagonal) matrix of energies with repeated values for degeneracy of mJ
    sortedstates is [J,mJ,L,mL,S,mS] states corresponding to the sorted energies in the matrix. Off diagonals are excluded to this point.

    '''

    # Create an array that contains [L,ml,S,ms] for the given L,S values. Starts with highest ml,mS values and itereates through ms first.
    LS0 = [[float(L_state), float(x), float(S_state), float(y)] for x in np.linspace(L_state, -L_state, int(2*L_state+1), dtype=float) for y in np.linspace(S_state, -S_state, int(2*S_state+1), dtype=float)]
    # LS0 = [[L_state, x, S_state, y] for x in np.linspace(L_state, -L_state, int(2*L_state+1),dtype=float) for y in np.linspace(S_state, -S_state, int(2*S_state+1),dtype=float) ]
    # print(LS0)
    # Now create the possible J values
    Jvals = np.arange(np.abs(L_state-S_state), (L_state+S_state+.1))  # ,dtype=float)

    # Use those J values to make all possible mJ values and starts with Lowest J, and lowest mJ
    Jmjs0 = make_Jmj(Jvals)

    # Combine both lists together to be a single large [J,mJ,L,mL,S,mS] list. The dimensions should match for LS0 and Jmjs0.
    bigstatelist = [[x[0], x[1], y[0], y[1], y[2], y[3]] for x, y in zip(Jmjs0, LS0)]

    # Expand the input energies which were given for each J to correspond to the number of mJs
    E_cm0 = [[E_level[x]]*int(2*Jvals[x] + 1) for x in range(len(E_level))]
    # Flatten the results out
    E_cm0_flat = [item for row in E_cm0 for item in row]
    E_cm0_Mat = np.diagflat(E_cm0_flat)
    # This step uses a low field approximation in order to sort the state-list based on the given magnetic field such that the relative state function order is known.
    # This is used for signal strength calculation as otherwise we may not know what order the functions are after they get diagonalized. This step might not be needed.
    LowE = []

    if sortE == True:
        biglist = []
        for i, x in enumerate(E_cm0_flat):
            thislandegJ = lande_gj(bigstatelist[i][0], bigstatelist[i][2], bigstatelist[i][4])  # caluclate lande gj factor for a given state
            if thisB == 0:
                thisZeemanEnergy = x  # This is to prevent issues with attempting to sort for B=0
            else:
                thisZeemanEnergy = Zeemanlow(thislandegJ, bigstatelist[i][1], thisB) + x  # This is the low zeeman approximation total energy
            biglist.append([thisZeemanEnergy, x, Jmjs0[i], LS0[i], bigstatelist[i]])

        sortedlist = sorted(biglist)  # Soted in ascending energy
        LowE = [x[0] for x in sortedlist]
        sortedstates = [x[-1] for x in sortedlist]
        E_cm_flat = [x[1] for x in sortedlist]  # This is the sorted energies, and is in cm^-1
        # Jmjs = [x[2] for x in sortedlist]
        # LS = [x[3] for x in sortedlist]
        E_SO_cm_Mat = np.diagflat([item for item in E_cm_flat])  # Despite the name, these energies are actually in cm-1

    # We now use wigner 3j to get an equivalent decoupling as per how CG coefficients work.
    # The result is a 2d where the columns are the |J,mJ> states and the rows determine what the corresponding |L,mL,S,mS> state was.
    # Eventually would like a way to convert this data to a more human readable form
        dim = len(Jmjs0)
        w3mat = np.zeros((dim, dim))
        # [J,mJ,L,mL,S,mS]
        for i, states1 in enumerate(sortedstates):

            for j, states2 in enumerate(sortedstates):
                # This conditional is saying that J != J', as otherwise the different J levels will cause duplicate CG's.
                if states1[2] != states2[2]:
                    CG1 = 0
                else:
                    # The CG coefficient couples J,mJ to L,mL, S mS
                    CG1 = CGw3(states2[2], states2[3], states2[4], states2[5], states1[0], states1[1])
                w3mat[i, j] = CG1
        return [w3mat, E_SO_cm_Mat, LowE, E_cm_flat, sortedstates]

    else:
        dim = len(Jmjs0)
        w3mat = np.zeros((dim, dim))
        # [J,mJ,L,mL,S,mS]
        for i, states1 in enumerate(bigstatelist):

            for j, states2 in enumerate(bigstatelist):
                # This conditional is saying that J != J', as otherwise the different J levels will cause duplicate CG's.
                if states1[2] != states2[2]:
                    CG1 = 0
                else:
                    # The CG coefficient couples J,mJ to L,mL, S mS
                    CG1 = CGw3(states2[2], states2[3], states2[4], states2[5], states1[0], states1[1])
                w3mat[i, j] = CG1
        return [w3mat, E_cm0_Mat, LowE, E_cm0_flat, bigstatelist]

def _magnetic_eigenvectors(level, zeeman):
    """Return definite-M eigenvectors matched to the energy ranks of zeeman[0].

    The axial Hamiltonian does not couple different M, so it is rebuilt from
    the eigensystem and diagonalized separately within each M block. This
    avoids tolerance-based resolution of mixed-M vectors near level crossings.
    Equal-energy columns are assigned a reconstructed basis; their original
    individual identities are not preserved. The supplied arrays are unchanged.
    """
    vectors = np.asarray(zeeman[1])
    energies = np.asarray(zeeman[0])
    magnetic_m = np.asarray([state[1] for state in level[-1]])
    hamiltonian = (vectors*energies) @ vectors.conj().T
    block_energies, block_vectors = [], []
    for m in np.unique(magnetic_m):
        idx = np.flatnonzero(magnetic_m == m)
        w, u = np.linalg.eigh(hamiltonian[np.ix_(idx, idx)])
        full = np.zeros((len(magnetic_m), len(idx)), dtype=u.dtype)
        full[idx, :] = u
        block_energies.append(w)
        block_vectors.append(full)
    block_energies = np.concatenate(block_energies)
    block_vectors = np.concatenate(block_vectors, axis=1)
    # Assign the k-th lowest block eigenvector to the column holding the k-th
    # lowest energy, so the result follows the column order of zeeman[0].
    resolved = np.empty_like(block_vectors)
    resolved[:, np.argsort(energies, kind='stable')] = block_vectors[:, np.argsort(block_energies, kind='stable')]
    return resolved


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


def Zeeman_signal(LevelG, LZG, LevelE, LZE, Bangle=90, gamma=0, Filter=False, T_Alpha = 1, T_Beta = 1):
    """Return relative E1 strengths and eigenpair-indexed polarization labels.

    The returned basis-pair list and Signalcoeffa remain basis-indexed; rad,
    wavelengths and strengths refer to columns of the transformed operator.
    Degenerate input columns may be replaced by a definite-M basis, so their
    original individual transition identities are not preserved.
    """
    return _zeeman_signal(LevelG, LZG, LevelE, LZE, Bangle, gamma, Filter, T_Alpha, T_Beta)


def _zeeman_signal(LevelG, LZG, LevelE, LZE, Bangle=90, gamma=0, Filter=False,
                   T_Alpha=1, T_Beta=1, label_eigenpairs=True):
    # print(T_Alpha,T_Beta)
    # Need to make a list of all possible combinations such that each upper state is repeated by the dim of the lower level, and a second list where the opposite is true.
    # ie, each lower level is repeated by the count of upper level states. This makes two lower x upper length arrays that will be useful for wigner calcs.
    bjlist = [[x, y] for y in LevelE[-1] for x in LevelG[-1]]  # This list will be [[bigstatesG], [bigstatesE]]

    # need to define the polarization effect from the filter.
    basis_rad = []
    for i in range(len(bjlist)):


        basis_rad.append(polarization(bjlist[i][1][1], bjlist[i][0][1], Bangle, gamma, Pol_Filter=Filter, TAlpha = T_Alpha, TBeta = T_Beta))

    # Calculate the signal strength coefficients as the product of radiation polarization term and dipole strength.
    # This is essentially a row vector.
    Signalcoeff = []
    dipoles = []
    for i in range(len(bjlist)):
        if basis_rad[i][1] == 3:
            dipoles.append(0)
            Signalcoeff.append(0)
        else:
            dipoles.append(dipolestr(bjlist[i][1], bjlist[i][0]))
            Signalcoeff.append(basis_rad[i][0]*dipoles[-1])
    Signalcoeffa = np.array(Signalcoeff)

    if label_eigenpairs:
        vectors_e = _magnetic_eigenvectors(LevelE, LZE)
        vectors_g = _magnetic_eigenvectors(LevelG, LZG)
        upper_m = np.asarray([state[1] for state in LevelE[-1]]) @ np.abs(vectors_e)**2
        lower_m = np.asarray([state[1] for state in LevelG[-1]]) @ np.abs(vectors_g)**2
        afcf = np.kron(vectors_e, vectors_g)
        rad = _eigenpair_polarization(afcf, [item[1] for item in basis_rad], dipoles,
                                    lower_m, upper_m, Bangle, gamma, Filter, T_Alpha, T_Beta)
    else:
        # Approximation spectra consume only strengths and wavelengths; their
        # eigenvectors do not necessarily use the coupled M basis above.
        afcf = np.kron(np.asarray(LZE[1]), np.asarray(LZG[1]))
        rad = basis_rad

    signal_out = []
    for i in range(len(bjlist)):
        signal_out.append((np.sum(afcf[:, i]*Signalcoeffa)**2))
    tsigout = np.array(signal_out)
    sigsum = np.sum(tsigout)

    #But it doesn't seem like this actually changes the final spectra post normalization.
    sigout = np.array([x/sigsum for x in tsigout])  # Make it so that sum of intensities is 1.
    #This means we have an oscillator strength here.
    # Conver the output to wavelength in nm then account for shift from vacuum to air.
    wave_vac = np.array([1e7/(y-x) for y in LZE[0] for x in LZG[0]])
    wave_air = Vac_to_air(wave_vac)

    return [sigout, wave_air, rad, bjlist, afcf, Signalcoeffa, wave_vac,sigsum,tsigout]

def Zeeman_sig_norad(LevelG, LZG, LevelE, LZE):

    # Need to make a list of all possible combinations such that each upper state is repeated by the dim of the lower level, and a second list where the opposite is true.
    # ie, each lower level is repeated by the count of upper level states. This makes two lower x upper length arrays that will be useful for wigner calcs.
    bjlist = [[x, y] for y in LevelE[-1] for x in LevelG[-1]]  # This list will be [[bigstatesG], [bigstatesE]]

    # need to define the polarization effect from the filter.

    # Calculate the signal strength coefficients as the product of radiation polarization term and dipole strength.
    # This is essentially a row vector.
    Signalcoeff = []
    for i in range(len(bjlist)):
        Signalcoeff.append(dipolestr(bjlist[i][0], bjlist[i][1]))
    Signalcoeffa = np.array(Signalcoeff)

    afcf = np.kron(np.array(LZE[1]), np.array(LZG[1]))  # This uses the tensor product after the magnetic field is taken into account.

    # Calculate the signal which is the sum of the square of the product of afcf, which si reduced dipole signal (including field effects) and the signal coeff. Isler1997
    signal_out = []
    for i in range(len(bjlist)):
        signal_out.append((np.sum(afcf[:, i]*Signalcoeffa)**2))
    tsigout = np.array(signal_out)
    sigsum = np.sum(tsigout)
    sigout = np.array([x/sigsum for x in tsigout])  # Make it so taht sum of intensities is 1.

    return sigout


def Zeeman_func(Level_in, Binput, Lowfield=False, Highfield=False, Eterm=None):
    '''
    Calculates the Zeeman shift hamiltonian and combines it with the Eo energy before diagonalizing to get the eig vec/vals.

    Parameters
    ----------
    Level_in : TYPE
        DESCRIPTION.
    Binput : TYPE
        DESCRIPTION.
    Eterm : float, optional
        Term energy in cm^-1, required when Highfield=True. Zero is valid.
        The non-HFS high-field approximation neglects fine-structure splitting
        and retains ZeemanHigh's g_L=1, g_S=2 convention.

    Returns
    -------
    list
        Eigenvalues and eigenvector columns in the coupled |J,mJ> basis.
        High-field energies are diagonalized in |mL,mS>, then the vectors are
        transformed back to the coupled basis used by Zeeman_signal.

    '''
    # Level=[CGmat, Eo_mat,E0_flat,bigstatelist]
    Ldim = len(Level_in[-1])  # Dimension of the atomic level: Length of the # of possible states.
    # Spi is wavefunction basis matrix. Comprised of CG coefficients.
    Psi = Level_in[0]

    PsiL = np.zeros_like(Psi)  # Matrix for storing mL contribution of H_Zeeman
    PsiS = np.zeros_like(Psi)  # Matrix for storing mS contribution of H_Zeeman

    # bigstatelist [J,mJ,L,mL,S,mS]
    # Create diagonal elements for the m values for each qm. Doing this is forcing that <n|m> = delta_nm, so 1 if n=m, but 0 else.
    tmLMat = np.diagflat([x[3] for x in Level_in[-1]])
    tmSMat = np.diagflat([x[5] for x in Level_in[-1]])

    # This is equivalent to the <n|Jz|m> calculation to make the pertubation matrix of zeeman energies
    for i in range(Ldim):
        tPsiL = Psi[i]
        for j in range(Ldim):
            PsiL[i, j] = np.linalg.multi_dot([tPsiL, tmLMat, Psi[j]])
            PsiS[i, j] = np.linalg.multi_dot([tPsiL, tmSMat, Psi[j]])

    # Combine the energy matricies for mL and mS

    # Diagonalize and calculate the eigenvalues and eigenvectors
    # Check for High and Low field approximations
    if Lowfield == False and Highfield == False:  # If both flags False - Do intermediate regime
        H_Zeeman = g_S*PsiS + g_L*PsiL

        scaledMat = np.asmatrix(Level_in[1] + Binput*H_Zeeman*(muBcm))

    elif Highfield == True and not Lowfield:  # Check if only High field
        if Eterm is None:
            raise ValueError("Non-HFS high-field calculations require Eterm in cm^-1 (zero is valid).")
        H_ZeemanH = np.array([ZeemanHigh(x[5], x[3], Binput) for x in Level_in[-1]])
        # mL/mS occupy the columns of Psi. Both the term energy and Zeeman
        # shifts are diagonal in this uncoupled basis, not broadcast vectors.
        scaledMat = np.asmatrix(np.diag(Eterm + H_ZeemanH))

    elif Lowfield == True and not Highfield:  # Check if only Low field
        gjmat = np.array([lande_gj(x[0], x[2], x[4]) for x in Level_in[-1]])
        tmJMat = np.diagflat([x[1] for x in Level_in[-1]])

        H_ZeemanL = gjmat*tmJMat

        scaledMat = np.asmatrix(Level_in[1] + Binput*H_ZeemanL*(muBcm))
    else:
        print('Something went wrong in Zeeman_func flag checking')
        # scaledMat = np.asmatrix(Level_in[1] )

    eigZ, eigvecZ = np.linalg.eigh(scaledMat)  # , driver = "evx") #Using driver="evx" seems to change some of the signs of the resulting matricies. In the 0-field case, the eigenvectors should all be +1 I believe.

    if Highfield and not Lowfield:
        eigvecZ = np.dot(Psi, eigvecZ)

    return [eigZ, eigvecZ]


def HZeeman(Level_in):
    '''
    Calculates the Zeeman Energy from the Hamiltonian using perturbation theory of <psi' |H'|psi> for H' = H_Zeeman for each m (ml, ms)
    This just outputs the Zeeman Hamiltonian, assuming B = 1T. This Zeeman component can be scaled based on the magnetic field strength.


    Parameters
    ----------
    Level_in : List
        This is the output of the StateProcess function of the format Level=[w3mat, Eo_mat,E0_flat,bigstatelist]


    Returns
    -------
    list
        [E0, H_Zeeman], where E0 is the zero-field energies and H_Zeeman is the Zeeman energies at 1T.

    '''
    # Level=[CGmat, Eo_mat,E0_flat,bigstatelist]
    Ldim = len(Level_in[-1])  # Dimension of the atomic level: Length of the # of possible states.
    # Spi is wavefunction basis matrix. COmprised of CG coefficients.
    Psi = Level_in[0]

    # PsiR = Level_in[0].T

    PsiL = np.zeros_like(Psi)
    PsiS = np.zeros_like(Psi)

    # bigstatelist [J,mJ,L,mL,S,mS]
    # Create diagonal elements for the m values for each qm. Doing this is forcing that <n|m> = delta_nm, so 1 if n=m, but 0 else.
    tmLMat = np.diagflat([x[3] for x in Level_in[-1]])
    tmSMat = np.diagflat([x[5] for x in Level_in[-1]])

    # This is equivalent to the <n|Jz|m> calculation to make the pertubation matrix of zeeman energies

    for i in range(Ldim):
        tPsiL = Psi[i]
        for j in range(Ldim):
            PsiL[i, j] = np.linalg.multi_dot([tPsiL, tmLMat, Psi[j]])
            PsiS[i, j] = np.linalg.multi_dot([tPsiL, tmSMat, Psi[j]])
    # #Combine the energy matricies for mL and mS
    H_Zeeman = g_S*PsiS + g_L*PsiL  # Combine all components

    E0 = Level_in[1]

    return [E0, H_Zeeman]


# %%ZeemanHFS

def stateprocess_HFS(L_state, S_state, I_state, HFS_Const, g_I, E_level, thisB, sortE=False):

    # Create an array that contains [L,ml,S,ms] for the given L,S values. Starts with lowest ml,mS values and itereates through ms first.
    # LS0 = [[L_state, x, S_state, y] for x in np.linspace(-L_state, L_state, int(2*L_state+1)) for y in np.linspace(-S_state, S_state, int(2*S_state+1)) ]
    # Create an array that contains [L,ml,S,ms,I,mI] for the given L,S,I values. Starts with Highest mL,mS values and itereates through mI first, then mS, then mL
    # Extra type-casting is required due to new numpy.float64 objects interaction with sympy wigner functions
    LSI0 = [[float(L_state), float(x), float(S_state), float(y), float(I_state), float(z)] for x in np.linspace(-L_state, L_state, int(2*L_state+1)) for y in np.linspace(-S_state, S_state, int(2*S_state+1))for z in np.linspace(I_state, -I_state, int(2*I_state+1))]

    # Now create the V values, starting with the lowest J state first.

    # Make all Js corresponding to L,S relation
    Jvals = np.arange(np.abs(L_state-S_state), (L_state+S_state+.1))

    # Make all mJs for the Js
    Jmjs0 = make_Jmj(Jvals)

    # Make a list of J,mJ,I,mI states
    JI0 = [[float(x[0]), float(x[1]), float(I_state), float(y)] for x in Jmjs0 for y in np.linspace(-I_state, I_state, int(2*I_state+1))]

    # Combine J and Is to make possible Fs
    Fvals = [np.arange(np.abs(-x + I_state), (x + I_state+.1)) for x in Jvals]
    # Flatten the list to be one big list
    Fvals = [float(item) for row in Fvals for item in row]
    # Make the possible mFs for all Fs
    Fmfs0 = make_Jmj(Fvals)

    E_cm0 = [[E_level[x]]*int(2*Jvals[x] + 1)*int(2*I_state + 1) for x in range(len(E_level))]
    E0_cm_flat = [item for row in E_cm0 for item in row]

    HFS_A_In = HFS_Const['HFSA']
    try:
        HFSConst = [[HFS_A_In[x]]*int(2*Jvals[x] + 1)*int(2*I_state + 1) for x in range(len(E_level))]
        HFS_Flat = [item for row in HFSConst for item in row]
    except:
        # This handles the case of single length HFS inputs
        HFSConst = [[HFS_A_In]*int(2*Jvals[x] + 1)*int(2*I_state + 1) for x in range(len(E_level))]
        HFS_Flat = [item for row in HFSConst for item in row]

        # HFS_Flat = HFSConst
    # print(HFSConst)

    if 'HFSB' in HFS_Const:
        try:
            HFSConstB = [[HFS_Const['HFSB'][x]]*int(2*Jvals[x] + 1)*int(2*I_state + 1) for x in range(len(E_level))]
            HFS_FlatB = [item for row in HFSConstB for item in row]
        except:
            HFSConstB = [list(HFS_Const['HFSB'])*int(2*Jvals[x] + 1)*int(2*I_state + 1) for x in range(len(E_level))]
            HFS_FlatB = [item for row in HFSConstB for item in row]
    else:
        HFS_FlatB = np.zeros_like(E0_cm_flat)
    # Duplicate energies based on degeneracy and flatten the results

    # Duplicate HFS constants to match F levels based on degeneracy and flatten the results
    # HFSConst = [[HFS_Const[0][x]]*int(2*Jvals[x] +1)*int(2*I_state + 1) for x in range(len(E_level))]
    # Flatten the shape of the Energy array

    # Make a large list of [F,mF,J,mJ,I,mI,L,mL,S,mS]
    bigstatelist = [[x[0], x[1], y[0], y[1], y[2], y[3], z[0], z[1], z[2], z[3]] for x, y, z, in zip(Fmfs0, JI0, LSI0)]

    # This step uses a low field approximation in order to sort the state-list based on the given magnetic field such that the relative state function order is known.
    # This is used for signal strength calculation as otherwise we may not know what order the functions are after they get diagonalized. This step might not be needed.

    if sortE:
        biglist = []
        for i, x in enumerate(E0_cm_flat):
            # thislandegF = lande_gf(bigstatelist[i][0],I_state, bigstatelist[i][2], L_state,S_state,g_I) #Calculate lande gf factor for the given transition
            thislandegF = lande_gf(bigstatelist[i][0], bigstatelist[i][4], bigstatelist[i][2], bigstatelist[i][6], bigstatelist[i][8], g_I)  # Calculate lande gf factor for the given transition

            # Lowfield zeeman approximation with hfs, which should be mf and gf.
            thisZeemanEnergy = Zeemanlow(thislandegF, bigstatelist[i][1], thisB) + x + Delta_E_HFS(I_state, bigstatelist[i][2], bigstatelist[i][0], HFS_Flat[i], HFS_FlatB[i])

            # thisZeemanEnergy = Zeemanlow(thislandegJ,bigstatelist[i][3],thisB) + muBcm*g_I*thisB*bigstatelist[i][5] + x + Delta_E_HFS(I_state,bigstatelist[i][2],bigstatelist[i][0],HFS_Flat[i],HFS_FlatB[i])
            biglist.append([thisZeemanEnergy, x, bigstatelist[i], HFS_Flat[i], HFS_FlatB[i]])
        # Sort the results and pull sorted values
        sortedlist = sorted(biglist)
        sortedstates = [x[2] for x in sortedlist]
        sortedHFC = [x[3] for x in sortedlist]  # Sorted hyperfine constants
        sortedHFCB = [x[4] for x in sortedlist]  # Sorted hyperfine constants

        E_cm = [x[1] for x in sortedlist]
        E_low = [x[0] for x in sortedlist]

        # Calculate Hyperfine energies with sorted states and values.
        E_HFSFlat = [Delta_E_HFS(states1[4], states1[2], states1[0], sortedHFC[i], sortedHFCB[i]) for i, states1 in enumerate(sortedstates)]  # Hyperfine + E0 (spin orbital, Nist)

        # E_HFSFlat = [E_cm[i] + Delta_E_HFS(states1[4],states1[2],states1[0], sortedHFC[i], sortedHFCB[i])    for i,states1 in enumerate(sortedstates)] #Hyperfine + E0 (spin orbital, Nist)
        # E_HFSA = [Delta_E_HFS(states1[4],states1[2],states1[0], sortedHFC[i],0)    for i,states1 in enumerate(sortedstates)] #Just Hyperfine energy
        # E_HFSB = [Delta_E_HFS(states1[4],states1[2],states1[0], 0, B=sortedHFCB[i])    for i,states1 in enumerate(sortedstates)] #Just Hyperfine energy

        # E0_cm_Mat = np.diagflat([item for item in E_HFSFlat]) #Make the diagonal energy matrix
        E0_cm_Mat = np.diagflat([item for item in E_cm])  # Make the diagonal energy matrix

        # We now use wigner 3j to get an equivalent decoupling as per how CG coefficients work.
        # There are two coefficients as we go from |F,mF> -> |J,mJ>|I,mI> then another set for |J,mJ>|I,mI> -> |L,mL>|S,mS>|I,mI>. Keeping track of constants is slightly tricky.
        # There can be issues with overla
        # Eventually would like a way to convert this data to a more human readable form
        dimhfs = len(Fmfs0)
        w3cg1 = np.zeros((dimhfs, dimhfs))
        w3cg2 = np.zeros((dimhfs, dimhfs))
        # Done in a single set of iterations. The check for states1[2] != states2[2] is saying that J != J' to calculate, otherwise it's zero.
        for i, states1 in enumerate(sortedstates):

            for j, states2 in enumerate(sortedstates):
                # This conditional is saying that J != J', as otherwise the different J levels will cause duplicate CG's.
                if states1[2] != states2[2]:
                    CG1 = 0
                else:
                    # Make a large list of [F,mF,J,mJ,I,mI,L,mL,S,mS]

                    # The CG coefficient couples F,mF to J,mJ;I,mI for CG1
                    CG1 = CGw3(states2[2], states2[3], states2[4], states2[5], states1[0], states1[1])
                # This conditional ensures we don't allow nuclear spin to flip or change between states. states[5] is mI.
                if states1[5] != states2[5]:
                    CG2 = 0
                    # CG2 = CGw3(states2[6],states2[7],states2[8],states2[9],states1[2],states1[3])
                    # w3cg2[i,j] = CG2

                else:
                    CG2 = CGw3(states2[6], states2[7], states2[8], states2[9], states1[2], states1[3])

                    w3cg2[i, j] = CG2
                w3cg1[i, j] = CG1

        w3cgprod = np.matmul(w3cg1, w3cg2)

        return [w3cgprod, E0_cm_Mat, E_HFSFlat, E_low, sortedstates]
    else:

        # E_cm = [x[1] for x in biglist]

        # Calculate Hyperfine energies with sorted states and values.
        E_HFSFlat = [Delta_E_HFS(states1[4], states1[2], states1[0], HFS_Flat[i], HFS_FlatB[i]) for i, states1 in enumerate(bigstatelist)]  # Hyperfine + E0 (spin orbital, Nist)

        #Holdovers from work with HFS energy contributions. Left for future possible use.
        # E0_HFSFlat = [E_cm[i] + Delta_E_HFS(states1[4],states1[2],states1[0], HFS_Flat[i], HFS_FlatB[i])    for i,states1 in enumerate(bigstatelist)] #Hyperfine + E0 (spin orbital, Nist)
        # E_HFSA = [Delta_E_HFS(states1[4],states1[2],states1[0], sortedHFC[i],0)    for i,states1 in enumerate(sortedstates)] #Just Hyperfine energy
        # E_HFSB = [Delta_E_HFS(states1[4],states1[2],states1[0], 0, B=sortedHFCB[i])    for i,states1 in enumerate(sortedstates)] #Just Hyperfine energy

        E0_cm_Mat = np.diagflat([item for item in E0_cm_flat])  # Make the diagonal energy matrix

        # We now use wigner 3j to get an equivalent decoupling as per how CG coefficients work.
        # There are two coefficients as we go from |F,mF> -> |J,mJ>|I,mI> then another set for |J,mJ>|I,mI> -> |L,mL>|S,mS>|I,mI>. Keeping track of constants is slightly tricky.
        # There can be issues with overla
        # Eventually would like a way to convert this data to a more human readable form
        dimhfs = len(Fmfs0)
        w3cg1 = np.zeros((dimhfs, dimhfs))
        w3cg2 = np.zeros((dimhfs, dimhfs))
        # Done in a single set of iterations. The check for states1[2] != states2[2] is saying that J != J' to calculate, otherwise it's zero.
        for i, states1 in enumerate(bigstatelist):

            for j, states2 in enumerate(bigstatelist):
                # This conditional is saying that J != J', as otherwise the different J levels will cause duplicate CG's.
                if states1[2] != states2[2]:
                    CG1 = 0

                else:
                    # Make a large list of [F,mF,J,mJ,I,mI,L,mL,S,mS]

                    # The CG coefficient couples F,mF to J,mJ;I,mI for CG1
                    CG1 = CGw3(states2[2], states2[3], states2[4], states2[5], states1[0], states1[1])
                # This conditional ensures we don't allow nuclear spin to flip or change between states. states[5] is mI.
                if states1[5] != states2[5]:
                    CG2 = 0
                    # CG2 = CGw3(states2[6],states2[7],states2[8],states2[9],states1[2],states1[3])
                    # w3cg2[i,j] = CG2

                else:
                    CG2 = CGw3(states2[6], states2[7], states2[8], states2[9], states1[2], states1[3])

                    w3cg2[i, j] = CG2
                w3cg1[i, j] = CG1

        w3cgprod = np.matmul(w3cg1, w3cg2)
        return [w3cgprod, E0_cm_Mat, E_HFSFlat, HFS_Flat, bigstatelist]


def dipolestr_HFS(LG, LE):
    '''
    Wigner3j =  (F     1    F'  )
                (-m_f  q   -m_F )   
    Wigner6j1 = {L'    J'  S  }
                {J    L    1  } 
    Wigner6j2 = {J'   F'   I  }
                {F    J    1  }                   
    Per Metcalf 4.33 

    Input state objects are [F,mF,J,mJ,I,mI,L,mL,S,mS]               
    Parameters
    ----------

    Returns
    -------
    tdp3 : float
        This is the u_eg dipole scalar for a specific transition between states (m-resolved)
        For use with nuclear spin only.

    '''
    # States are from bigsortedlist = [F,mF,J,mJ,I,mI,L,mL,S,mS]
    # [0,1 ,2,3 ,4,5 ,6,7 ,8,9 ]

    thisq = LE[1] - LG[1]

    # This is Metcalf 4.33
    # Calculate the wigner 3j and two wigner 6j.
    tw3 = float(w3j(LG[0], 1, LE[0], LG[1], thisq, -LE[1]))
    tw61 = float(w6j(LE[6], LE[2], LG[8], LG[2], LG[6], 1))
    tw62 = float(w6j(LE[2], LE[0], LG[4], LG[0], LG[2], 1))

    tphase = (-1)**(1 + LE[6] + LG[8] + LG[2] + LE[2] + LG[4] - LE[1])
    troot = np.sqrt((2*LG[0] + 1)*(2*LE[0]+1) * (2*LG[2] + 1)*(2*LE[2] + 1))

    tdp3 = tphase * troot*tw3*tw61*tw62

    return tdp3


def _hfs_magnetic_eigenvectors(level, zeeman):
    """Resolve definite-mF columns, preserving supplied energy-rank order.

    Rebuild the axial Hamiltonian and diagonalize its separate mF blocks.
    Equal-energy input columns receive a reconstructed basis; their individual
    identities are not preserved. The input eigensystem is not modified.
    """
    energies, vectors = np.asarray(zeeman[0]), np.asarray(zeeman[1])
    magnetic_m = np.asarray([state[1] for state in level[-1]])
    # Remove the common energy offset to avoid cancellation when small HFS
    # splittings sit on a large optical energy. Fix summation order as well.
    columns = np.argsort(energies, kind='stable')
    ordered = vectors[:, columns]
    hamiltonian = np.dot(ordered*(energies[columns]-np.min(energies)), ordered.conj().T)
    block_energies, block_vectors, block_m = [], [], []
    for m in np.unique(magnetic_m):
        indices = np.flatnonzero(magnetic_m == m)
        w, u = np.linalg.eigh(hamiltonian[np.ix_(indices, indices)])
        full = np.zeros((len(magnetic_m), len(indices)), dtype=u.dtype)
        full[indices, :] = u
        block_energies.extend(w)
        block_vectors.append(full)
        block_m.extend([m]*len(indices))
    ranks = np.argsort(block_energies, kind='stable')
    resolved = np.empty_like(vectors)
    resolved[:, columns] = np.concatenate(block_vectors, axis=1)[:, ranks]
    resolved_m = np.empty_like(magnetic_m)
    resolved_m[columns] = np.asarray(block_m)[ranks]
    return resolved, resolved_m


def Zeeman_signal_HFS(LevelG, LZG, LevelE, LZE, Bangle=90, gamma=0, Filter=False,T_Alpha = 1, T_Beta = 1):
    """Return relative E1 strengths with eigenpair-indexed polarization labels.

    rad, strengths and wavelengths share upper-then-lower eigenpair order.
    bflist and Signalcoeffa retain their basis-pair order. Degenerate columns
    may receive a reconstructed definite-mF basis rather than retaining their
    individual identities. Sigma stick keys retain the emitted-light convention.
    """

    # Need to make a list of all possible combinations such that each upper state is repeated by the dim of the lower level, and a second list where the opposite is true.
    # ie, each lower level is repeated by the count of upper level states. This makes two lower x upper length arrays that will be useful for wigner calcs.
    bflist = [[x, y] for y in LevelE[-1] for x in LevelG[-1]]  # This list will be [[bigstatesG], [bigstatesE]]

    # need to define the polarization effect from the filter.
    basis_rad = []
    for i in range(len(bflist)):
        basis_rad.append(polarization(bflist[i][1][1], bflist[i][0][1], Bangle, gamma, Pol_Filter=Filter,TAlpha = T_Alpha, TBeta = T_Beta))
        # Rad output is [polfactor,'polarization'] with polarization being 'sigma-','pi','sigma+' accordingly
    # Calculate signal strength coefficient as product of dipole strength and polarization component
    Signalcoeff = []
    dipoles = []
    for i in range(len(bflist)):
        dipoles.append(dipolestr_HFS(bflist[i][0], bflist[i][1]))
        Signalcoeff.append(basis_rad[i][0]*dipoles[-1])
    Signalcoeffa = np.array(Signalcoeff)

    vectors_e, upper_m = _hfs_magnetic_eigenvectors(LevelE, LZE)
    vectors_g, lower_m = _hfs_magnetic_eigenvectors(LevelG, LZG)
    afcf = np.kron(vectors_e.conj(), vectors_g)
    delta_m = (upper_m[:, None] - lower_m[None, :]).ravel()
    # q is mF_upper - mF_lower, as in polarization() and dipolestr_HFS().
    rad = [polarization(float(q), 0, Bangle, gamma, Pol_Filter=Filter,
                        TAlpha=T_Alpha, TBeta=T_Beta) for q in delta_m]
    basis_q = np.asarray([pair[1][1] - pair[0][1] for pair in bflist])
    spherical = np.asarray(dipoles)[:, None]*(basis_q[:, None] == [-1, 0, 1])
    strengths_q = np.abs(np.dot(afcf.T, spherical))**2
    leakage = np.where(delta_m[:, None] == [-1, 0, 1], 0, strengths_q)
    if np.max(leakage) > 1e-12*np.max(strengths_q):
        raise ValueError('Spherical HFS E1 amplitudes disagree with eigenstate mF')

    # Calculate the signal. This is like the dot product of a column of the afcf reduced dipole state matrix with the row vector comprised of dipole strengths(with polarization)
    signal_out = []
    for i in range(len(bflist)):
        signal_out.append(abs(np.sum(afcf[:, i]*Signalcoeffa))**2)
    tsigout = np.array(signal_out)
    sigsum = np.sum(tsigout)
    sigout = np.array([x/sigsum for x in tsigout])  # Make it so taht sum of intensities is 1.

    # Convert the result from cm^-1 to nm and include shift to air from vacuum
    wave_vac = np.array([1e7/(y-x) for y in LZE[0] for x in LZG[0]])
    wave_air = Vac_to_air(wave_vac)

    return [sigout, wave_air, rad, bflist, afcf, Signalcoeffa, wave_vac]


def Zeeman_func_HFS(Level_in, Binput, gI, PsiHFS=True):
    '''
    Calcualte the Zeeman shift including hyperfine effects.

    Parameters
    ----------
    Level_in : TYPE
        DESCRIPTION.
    Binput : TYPE
        DESCRIPTION.
    gI : TYPE
        DESCRIPTION.

    Returns
    -------
    list containing the eigenvalues and eigenvectores after matrix diagonalization
        [eigZ,eigvecZ]


    '''
    Ldim = len(Level_in[-1])  # Dimension of the atomic level: Length of the # of possible states.
    Psi = Level_in[0]
    PsiR = Level_in[0].T
    PsiL = np.zeros_like(Psi)
    PsiS = np.zeros_like(Psi)
    PsiI = np.zeros_like(Psi)
    PsiH = np.zeros_like(Psi)

    # Create diagonal elements for the m values for each qm.
    tmLMat = np.diagflat([x[7] for x in Level_in[-1]])
    tmSMat = np.diagflat([x[9] for x in Level_in[-1]])
    tmIMat = np.diagflat([x[5] for x in Level_in[-1]])

    # H_HFS = AIJcos(IJ) = AIJ*(F(F+1) - I(I+1) - J(J+1))/(IJ)

    # tH_HFS = np.diagflat([x for x in Level_in[3]])

    tH_HFS = np.diagflat([x for x in Level_in[2]])
    # tH_HFS = np.diagflat([x for x in Level_in[3]])

    # This is equivalent to the <n|Jz|m> calculation to make the pertubation matrix of zeeman energies
    # The += doesn't seem to change anything, but is kept as my original code used it
    # HFS should be treated in the same way as Zeeman, with the matrix multiplication of the wave fxns.

    for i in range(Ldim):
        tPsiL = Psi[i]
        for j in range(Ldim):

            PsiL[i, j] = np.linalg.multi_dot([tPsiL, tmLMat, Psi[j]])
            PsiS[i, j] = np.linalg.multi_dot([tPsiL, tmSMat, Psi[j]])
            PsiI[i, j] = np.linalg.multi_dot([tPsiL, tmIMat, Psi[j]])
            # PsiH[i,j] = np.linalg.multi_dot([tPsiL, tH_HFS, Psi[j]])

            # PsiH[i,j] = PsiR[i]@(tH_HFS@PsiR[j])

            PsiH[i, j] = np.linalg.multi_dot([PsiR[i], tH_HFS, PsiR[j]])

    H_Zeeman = g_S*PsiS + g_L*PsiL - gI*PsiI  # Combine all components

    # Combine the energy matrices
    # The HFS shifts are diagonal in the |F,mF> basis, the same basis as Level_in[1] and H_Zeeman
    if PsiHFS:
        # print('PsiH')
        scaledMat = np.asmatrix(Level_in[1] + tH_HFS + Binput*H_Zeeman*(muBcm))
    else:
        scaledMat = np.asmatrix(Level_in[1] + np.diagflat(Level_in[2]) + Binput*H_Zeeman*(muBcm))

    # scaledMat = np.asmatrix(Level_in[1] +Level_in[2]+Binput*H_Zeeman*(muBcm) )
    # Diagonalize and calculate eigenvalues and eigenvectors.
    eigZ, eigvecZ = np.linalg.eigh(scaledMat)  # , driver = "evx")
    # return [eigZ,eigvecZ,[tmLMat,tmSMat,tmIMat,g_S*PsiS, g_L*PsiL,  gI*PsiI]]

    return [eigZ, eigvecZ]


def HZeeman_HFS(Level_in, gI):
    '''
    Calcualte the Zeeman shift including hyperfine effects.

    Parameters
    ----------
    Level_in : TYPE
        DESCRIPTION.
    Binput : TYPE
        DESCRIPTION.
    gI : TYPE
        DESCRIPTION.

    Returns
    -------
    list containing the eigenvalues and eigenvectores after matrix diagonalization
        [eigZ,eigvecZ]


    '''
    # stateprocess outputs [w3mat, E_0, E_HFS,HHFS,statelist]
    Ldim = len(Level_in[-1])  # Dimension of the atomic level: Length of the # of possible states.
    Psi = Level_in[0]

    # PsiR = Level_in[0].T
    PsiL = np.zeros_like(Psi)
    PsiS = np.zeros_like(Psi)
    PsiI = np.zeros_like(Psi)
    PsiH = np.zeros_like(Psi)

    # Create diagonal elements for the m values for each qm.
    tmLMat = np.diagflat([x[7] for x in Level_in[-1]])
    tmSMat = np.diagflat([x[9] for x in Level_in[-1]])
    tmIMat = np.diagflat([x[5] for x in Level_in[-1]])

    # H_HFS = AIJcos(IJ) = AIJ*(F(F+1) - I(I+1) - J(J+1))/(IJ)

    # tH_HFS = np.diagflat([x for x in Level_in[3]])

    # tH_HFS = np.diagflat([2*x for x in Level_in[2]])
    tH_HFS = np.diagflat([x for x in Level_in[2]])

    # This is equivalent to the <n|Jz|m> calculation to make the pertubation matrix of zeeman energies
    # The += doesn't seem to change anything, but is kept as my original code used it
    # HFS should be treated in the same way as Zeeman, with the matrix multiplication of the wave fxns.
    for i in range(Ldim):
        tPsiL = Psi[i]

        for j in range(Ldim):
            PsiL[i, j] = np.linalg.multi_dot([tPsiL, tmLMat, Psi[j]])
            PsiS[i, j] = np.linalg.multi_dot([tPsiL, tmSMat, Psi[j]])
            PsiI[i, j] = np.linalg.multi_dot([tPsiL, tmIMat, Psi[j]])
            # PsiH[i,j] = np.linalg.multi_dot([tPsiL, tH_HFS, Psi[j]])

            # PsiH[i,j] = np.linalg.multi_dot([PsiR[i], tH_HFS, PsiR[j]])

    H_Zeeman = g_S*PsiS + g_L*PsiL - gI*PsiI  # Combine all components
    PsiH = tH_HFS

    E0 = Level_in[1]
    AHFS0 = np.diagflat([x for x in Level_in[3]])

    EHF0 = Level_in[2]

    out = {}
    out['E0'] = E0
    out['H_Z'] = H_Zeeman
    out['PsiHFS'] = PsiH
    out['AHFS0'] = AHFS0
    out['EHF0'] = EHF0
    return out


# %%ZeemanFan

#        stateprocces HFS return [w3cgprod,E0_cm_Mat,E_HFSFlat,E_low,bigstatelist]


def FastZfan(States_in, Bfield, Eterm_=0, g_I=0, A_HFS=None, B_HFS=0):
    
    # This is a supporting function which is for streamlining the diagonalization process which makes it much faster
    # Only the required calculation steps are taken for each magnetic field as the E0 and wavefunctions for zero-field don't change.
    # So make the wavefunctions once only before diagonalization.
    # Make a large list of [F,mF,J,mJ,I,mI,L,mL,S,mS] HFS or [J,mJ,L,mL,S,mS] no HFS
    ZfansLow = []
    ZfansHigh = []
    is_hfs = len(States_in[-1][0]) == 10
    if is_hfs:
        if A_HFS is None:
            raise ValueError('HFS fan approximations require A_HFS in Hz')
        j_values = sorted({state[2] for state in States_in[-1]})
        def constants_by_j(values):
            values = np.asarray(values, dtype=float).reshape(-1)
            if values.size == 1:
                values = np.repeat(values, len(j_values))
            if values.size != len(j_values):
                raise ValueError('HFS fan constants need one value per J, lowest J first')
            return dict(zip(j_values, values))
        a_by_j = constants_by_j(A_HFS)
        b_by_j = constants_by_j(B_HFS)
    for i, states in enumerate(States_in[-1]):
        if not is_hfs:
            thislandegF = lande_gj(states[0], states[2], states[4])
            thisZeemanEnergyL = Zeemanlow(thislandegF, states[1], Bfield) + States_in[1][i][i]
            thisZeemanEnergyH = ZeemanHigh(states[5], states[3], Bfield) + Eterm_

        else:

            thislandegF = lande_gf(states[0], states[4], states[2], states[6], states[8], g_I)  # Calculate lande gf factor for the given transition
            thisZeemanEnergyL = Zeemanlow(thislandegF, states[1], Bfield) + States_in[2][i] + States_in[1][i][i]
            # Do not add the F-dependent zero-field HFS shift again: the
            # Paschen-Back expression already contains its diagonal HFS term.
            thisZeemanEnergyH = (ZeemanhighHFS(states, a_by_j[states[2]], g_I,
                                             Bfield, b_by_j[states[2]])
                                + States_in[1][i][i])

        ZfansLow.append(thisZeemanEnergyL)
        ZfansHigh.append(thisZeemanEnergyH)

    # print(Zfans[-1])
    return [ZfansLow, ZfansHigh]


def ZeemanFan(Inputdeck, flags=''):
    # Take the Inputdeck, generate states, then calculate energy for each value. Transpose and put it out.
    # Take the required values from the input deck.
    '''
    flags : String
        Can be any combination of G,E,L,H. The default is ''.
        G: Fan plot Ground
        E: Fan Plot of Excited
        L: Include Lowfield Approx
        H: Include Highfield Approx. With HFS, this is the hyperfine
           Paschen-Back approximation (J remains a good quantum number).
    '''

    S_ground = Inputdeck['s_ground']
    S_excited = Inputdeck['s_excited']
    L_ground = Inputdeck['l_ground']
    L_excited = Inputdeck['l_excited']
    E_ground = Inputdeck['E_ground']
    E_excited = Inputdeck['E_excited']
    Bfield_ext = Inputdeck['Bmag']
    # Bext_angle = Inputdeck['b_angle']
    try:
        EtermE = Inputdeck['EtermE']
        EtermG = Inputdeck['EtermG']
    except:
        EtermE = None
        EtermG = None
    # Generate empty lists for storing values in later. Ground, excited, low and exact. Look to add high approximation too.
    # GZ_fan = []
    # EZ_fan = []

    # Check if HFS is being included.
    if 'I_spin' in Inputdeck:
        HFSConstsG = {}
        HFSConstsG['HFSA'] = Inputdeck['HFS_G']
        HFSConstsE = {}
        HFSConstsE['HFSA'] = Inputdeck['HFS_E']

        if 'HFSB_G' in Inputdeck:  # Check if either of the B HFS constants are in the input deck, otherwise make a copy of the A coefficients as zeros.
            HFSConstsG['HFSB'] = Inputdeck['HFSB_G']
        else:
            HFSConstsG['HFSB'] = np.zeros_like(E_ground)

        if 'HFSB_E' in Inputdeck:  # Check if either of the B HFS constants are in the input deck, otherwise make a copy of the A coefficients as zeros.
            HFSConstsE['HFSB'] = Inputdeck['HFSB_E']
        else:
            HFSConstsE['HFSB'] = np.zeros_like(E_excited)

        I_spin = Inputdeck['I_spin']
        mu_I = Inputdeck['mu_I']
        thisg_I = lande_gi(I_spin, mu_I)

        output = {}
        if 'G' in flags:

            Ground_Level_HFS = stateprocess_HFS(L_ground, S_ground, I_spin, HFSConstsG, thisg_I, E_ground, 0)
            Ground_Zeeman_HFS = HZeeman_HFS(Ground_Level_HFS, thisg_I)

            EigG = []
            for Bvals in Bfield_ext:

                # tmat = np.asmatrix(Ground_Zeeman_HFS[0] +  Ground_Zeeman_HFS[2] +  Bvals*Ground_Zeeman_HFS[1]*muBcm)
                if 'PsiHFS' in Inputdeck:
                    tmat = np.asmatrix(Ground_Zeeman_HFS['E0'] + Ground_Zeeman_HFS['PsiHFS'] + Bvals*Ground_Zeeman_HFS['H_Z']*muBcm)
                else:
                    G_HFS0 = np.diagflat([x for x in Ground_Zeeman_HFS['EHF0']])
                    tmat = np.asmatrix(Ground_Zeeman_HFS['E0'] + G_HFS0 + Bvals*Ground_Zeeman_HFS['H_Z']*muBcm)

                EigG.append(np.linalg.eigh(tmat)[0])
            output['G'] = np.array(EigG).T

            if 'L' in flags:
                LowG = []
                for Bvals in Bfield_ext:
                    LowG.append(FastZfan(Ground_Level_HFS, Bvals, g_I=thisg_I,
                                        A_HFS=HFSConstsG['HFSA'], B_HFS=HFSConstsG['HFSB'])[0])

                output['LowG'] = np.array(LowG).T
            if 'H' in flags:
                HighG = []
                for Bvals in Bfield_ext:
                    HighG.append(FastZfan(Ground_Level_HFS, Bvals, g_I=thisg_I,
                                         A_HFS=HFSConstsG['HFSA'], B_HFS=HFSConstsG['HFSB'])[1])

                output['HighG'] = np.array(HighG).T
        if 'E' in flags:
            Excited_Level_HFS = stateprocess_HFS(L_excited, S_excited,  I_spin, HFSConstsE, thisg_I, E_excited, 0)
            Excited_Zeeman_HFS = HZeeman_HFS(Excited_Level_HFS, thisg_I)

            EigE = []
            for Bvals in Bfield_ext:

                if 'PsiHFS' in Inputdeck:
                    tmat = np.asmatrix(Excited_Zeeman_HFS['E0'] + Excited_Zeeman_HFS['PsiHFS'] + Bvals*Excited_Zeeman_HFS['H_Z']*muBcm)
                else:
                    E_HFS0 = np.diagflat([x for x in Excited_Zeeman_HFS['EHF0']])
                    tmat = np.asmatrix(Excited_Zeeman_HFS['E0'] + E_HFS0 + Bvals*Excited_Zeeman_HFS['H_Z']*muBcm)

                EigE.append(np.linalg.eigh(tmat)[0])
            output['E'] = np.array(EigE).T
            if 'L' in flags:
                LowE = []
                for Bvals in Bfield_ext:
                    LowE.append(FastZfan(Excited_Level_HFS, Bvals, g_I=thisg_I,
                                        A_HFS=HFSConstsE['HFSA'], B_HFS=HFSConstsE['HFSB'])[0])

                output['LowE'] = np.array(LowE).T
            if 'H' in flags:
                HighE = []
                for Bvals in Bfield_ext:
                    HighE.append(FastZfan(Excited_Level_HFS, Bvals, g_I=thisg_I,
                                         A_HFS=HFSConstsE['HFSA'], B_HFS=HFSConstsE['HFSB'])[1])

                output['HighE'] = np.array(HighE).T

    else:
        output = {}
        if 'G' in flags:
            Ground_Level = stateprocess(L_ground, S_ground, E_ground, 0)
            Ground_Zeeman = HZeeman(Ground_Level)

            EigG = []
            for Bvals in Bfield_ext:

                tmat = np.asmatrix(Ground_Zeeman[0] + Bvals*Ground_Zeeman[1]*muBcm)

                EigG.append(np.linalg.eigh(tmat)[0])
            output['G'] = np.array(EigG).T

            if 'L' in flags:
                LowG = []
                for Bvals in Bfield_ext:
                    LowG.append(FastZfan(Ground_Level, Bvals)[0])

                output['LowG'] = np.array(LowG).T
            if 'H' in flags:
                HighG = []
                for Bvals in Bfield_ext:
                    HighG.append(FastZfan(Ground_Level, Bvals,Eterm_=EtermG)[1])

                output['HighG'] = np.array(HighG).T
        if 'E' in flags:
            Excited_Level = stateprocess(L_excited, S_excited, E_excited, 0)
            Excited_Zeeman = HZeeman(Excited_Level)

            EigE = []
            for Bvals in Bfield_ext:

                tmat = np.asmatrix(Excited_Zeeman[0] + Bvals*Excited_Zeeman[1]*muBcm)

                EigE.append(np.linalg.eigh(tmat)[0])
            output['E'] = np.array(EigE).T

            if 'L' in flags:
                LowE = []
                for Bvals in Bfield_ext:
                    LowE.append(FastZfan(Excited_Level, Bvals)[0])

                output['LowE'] = np.array(LowE).T
            if 'H' in flags:
                HighE = []
                for Bvals in Bfield_ext:
                    HighE.append(FastZfan(Excited_Level, Bvals,Eterm_=EtermE)[1])

                output['HighE'] = np.array(HighE).T
    return output


# %%Main


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


def AtomSpect_Main(Inputdeck):
    '''
    This is the main zeeman function. THe input is a dictionary with many optional keywords for more modular functionaly. These optional keywords are checked for to branch the function.


    Parameters
    ----------
    Inputdeck :     InputdeckExample = {
                     's_ground': 3  , #Spin multiplicity for ground state ^(2s +1)L_J , int or half int
                     's_excited': 3 , #Spin multiplicity for excited state ^(2s +1)L_J, int or half int
                     'l_ground': 0  , #Orbital Angular Momentum of ground state, int or half int
                     'l_excited': 1 , #Orbital Angular Momentum of excited state, int or half int
                     'E_ground': np.array([2951.29 ]) , #np.array of lower state, Lowest J first, cm^-1
                     'E_excited': np.array([26229.77,27488.11,27889.68]) ,  #np.array of upper state, Lowest J first, cm^-1
                     'Bmag': .6 , #Magnetic Field  [T]
                     'b_angle': 60 , #Angle between LoS and Bmax
                     'Pol_angle': 0 , #Angle polarizing filter makes with max linear transmission
                     'amu' : 183 , #Weight in AMU
                     'Convfxn': 'Gaussian' , #Optional: 'Gaussian', 'GaussianInstrum'', 'Skewed' , 'Lorrentzian', 'Voigt'.
                     'Skewness' : [0.62,0.99] #Optional, [Left,Right] constant for the skewness of the lorrentzian function. Between 0 and 1. If both are 1, it is a normal lorrentzian.
                     'Temp' : 300, #Temperature in K. Used for Gaussian, Gaussian Instrum, Voigt convolutions
                     'TempeV' : 20 , #Temperature in eV. Can be passed instead of Temp in K. Converted as needed internally.
                     'specstep' : 0.002222 , #stepsize of linefunction [nm]
                     'specres' : 10 , #How many points are evaluated per spectrometer pixel window. Higher means smoother curve.
                     'fxnwindow': 0.5 , #How far from the central peak the convolution will be calculated. Also related to how stick binning works. 
                     'plottitle': 'Excample Title' , #Title for plotting (optional).                      
                     'SpectrumData' : [[Wavelength],[Intensity]] , #List of arrays for wavelength (nm) and intensity (arb)
                     'spec_window' : [425,435], #Min and max for convolution window (nm)
                     'plot_window' : [425,435], #Min and max for plot window range (nm)
                     'HFS_G' : [505.5e6] , #Optional: Hyperfine A constant for the lower level, need one entry per J level, lowest J first [Hz]
                     'HFS_E': [0, 496.2e6, 440.5e6] , #Optional: Hyperfine A constants for upper level, need one entery per J level, lowest J first [Hz]
                     'I_spin' : 0.5, #Optional: nuclear spin, I. Tabulated by Stone et al.
                     'mu_I' :  .11778476 , #Optional: Nuclear dipole moment as tabulated by Stone.              
                     'HFSB_G' : [505.5e6] , #OPtional: Hyperfine B constant for the lower level, need one entry per J level, lowest J first [Hz]
                     'HFSB_E': [0, 496.2e6, 440.5e6] , #Optional: Hyperfine B constants for upper level, need one entery per J level, lowest J first [Hz]
                     'Calc_HFS_Vals': [False,False], #Optional. Passed regarding wether HFS constants A and B should be approximated for values of 0 above. NOT YET IMPLEMENTED
                     'ion_vel' : Float or [Float1,Float2] , #Velocity of particles, mono and bidirectional allowed. A single value can be passed and used for bidirectional splitting if bidir flag is set.
                     'bidir' : True , #Bool as to whether ion motion generates bidirection doppler shift
                     'PsiHFS' : True , #Bool, testing flag left in for completeness
                     'DoLowSig' : True , #Bool for whether Lowfield approximation and signal are included and plotted
                     'DoHighSig' : True , #REQUIRES TERM ENERGY!Bool for whether Highfield approximation and signal are included and plotted
                     'EtermG': 169086.9076, #Term energy of lower state in cm^-1 - Required for DoHighSig
                     'EtermE' : 191444.47832914, #Term energy of upper state in cm^-1 - Required for DoHighSig
                     'sortE' : True , #Bool as to whether the wavefunctions will attempt to be sorted based on expected resulting energy before diagonalization. This significantly slows down operations. 
                      }

    Returns
    -------
    data : Dictionary
        DESCRIPTION.

    '''

    # Define local variables from the inputdeck
    do_low = _approximation_enabled(Inputdeck.get('DoLowSig', False))
    do_high = _approximation_enabled(Inputdeck.get('DoHighSig', False))
    if 'I_spin' in Inputdeck and (do_low or do_high):
        raise NotImplementedError('HFS approximation spectra are not implemented; omit or disable DoLowSig/DoHighSig')

    S_ground = Inputdeck['s_ground']
    S_excited = Inputdeck['s_excited']
    L_ground = Inputdeck['l_ground']
    L_excited = Inputdeck['l_excited']
    E_ground = Inputdeck['E_ground']
    E_excited = Inputdeck['E_excited']
    Bfield_ext = Inputdeck['Bmag']
    Bext_angle = Inputdeck['b_angle']


    # Set polarization filter angle to 0 and define no polarization filter.
    Polangle = 0
    Polfilter = False
    if 'Pol_angle' in Inputdeck:  # Check if the inputdeck has a pol_angle and adjust parameters accordingly.
        Polangle = Inputdeck['Pol_angle']
        Polfilter = True


    if 'TAlpha' in Inputdeck:
        Talpha = Inputdeck['TAlpha']
    else:
        Talpha = 1
    
    if 'TBeta' in Inputdeck:
        Tbeta = Inputdeck['TBeta']
    else:
        Tbeta = 1
    
    if 'bidir' in Inputdeck:
        bidirect = Inputdeck['bidir']
    else:
        bidirect = False

    if 'sortE' in Inputdeck:
        sortE = Inputdeck['sortE']
    else:
        sortE = False

    if 'amu' in Inputdeck:
        pass
    else:
        Inputdeck['amu'] = 1  # Setting a default value so it's not required to be input.
     

    # 1. Establish the "Spectrometer" fallback range
    spec_data = Inputdeck.get('SpectrumData')
    # If data exists, use its bounds; otherwise, use the global default
    data_range = [np.min(spec_data[0]), np.max(spec_data[0])] if spec_data is not None else [1, 1000]
    
    # 2. Resolve windows: Dictionary Value > Spectrometer range > default
    specwindow = Inputdeck.get('spec_window', data_range)
    plotwindow = Inputdeck.get('plot_window', data_range)
    
    
    if 'EtermG' in Inputdeck:
        EtermG = Inputdeck['EtermG']
    else:
        EtermG = None
        
    if 'EtermE' in Inputdeck:
        EtermE = Inputdeck['EtermE']
    else:
        EtermE = None    
        
        
    if 'I_spin' in Inputdeck:  # Check if the input deck has nuclear spin as to calculate HFS splitting and zeeman effect or not.
        HFS_ConstsG = {}
        HFS_ConstsG['HFSA'] = Inputdeck['HFS_G']
        HFS_ConstsE = {}
        HFS_ConstsE['HFSA'] = Inputdeck['HFS_E']
        if 'HFSB_G' in Inputdeck:  # Check if either of the B HFS constants are in the input deck, otherwise make a copy of the A coefficients as zeros.
            HFS_ConstsG['HFSB'] = Inputdeck['HFSB_G']
        else:
            HFS_ConstsG['HFSB'] = np.zeros_like(E_ground)

        if 'HFSB_E' in Inputdeck:  # Check if either of the B HFS constants are in the input deck, otherwise make a copy of the A coefficients as zeros.
            HFS_ConstsE['HFSB'] = Inputdeck['HFSB_E']
        else:
            HFS_ConstsE['HFSB'] = np.zeros_like(E_excited)

        if 'PsiHFS' in Inputdeck:
            PsiHFS = Inputdeck['PsiHFS']
        else:
            PsiHFS = False
            

        # Use spin, spin parity, and nuclear dipole moment to calculate lande g_i nuclear factor. Might be off by a factor of 1836?
        I_spin = Inputdeck['I_spin']
        mu_I = Inputdeck['mu_I']
        thisg_I = lande_gi(I_spin, mu_I)

        # Process the wavefunction matrix and energies. Preliminary sorting is done so that the final state order should be roughly as expected.
        Ground_Level_HFS = stateprocess_HFS(L_ground, S_ground, I_spin, HFS_ConstsG, thisg_I, E_ground, Bfield_ext)
        Excited_Level_HFS = stateprocess_HFS(L_excited, S_excited, I_spin, HFS_ConstsE, thisg_I, E_excited, Bfield_ext)

        # Take the sorted states, the state matrix, and energy matrix and calculate the zeeman splitting (with HFS)
        Ground_Zeeman_HFS = Zeeman_func_HFS(Ground_Level_HFS, Bfield_ext, thisg_I, PsiHFS)
        Excited_Zeeman_HFS = Zeeman_func_HFS(Excited_Level_HFS, Bfield_ext, thisg_I, PsiHFS)
        # Use the eigenvectors and values from the diagonalization of the hamiltonian to calculate the relative signal strength. This function uses a combination of Isler1997 and Metcalf 4.32/4.33
        Z_sig_HFS = Zeeman_signal_HFS(Ground_Level_HFS, Ground_Zeeman_HFS, Excited_Level_HFS, Excited_Zeeman_HFS, Bangle=Bext_angle, gamma=Polangle, Filter=Polfilter,T_Alpha = Talpha, T_Beta = Tbeta)

        # Output a bunch of data
        data = {}
        data['signal'] = Z_sig_HFS[0]
        data['wave_vac'] = Z_sig_HFS[6]
        data['wave_air'] = Z_sig_HFS[1]
        data['HFSG_Energy'] = Ground_Level_HFS[2]
        data['HFSE_Energy'] = Excited_Level_HFS[2]
        data['statelist_Ground'] = Ground_Level_HFS[-1]
        data['statelist_Excited'] = Excited_Level_HFS[-1]
        data['RedDipoleOp'] = Z_sig_HFS[-3]  # This is the tensor/kronecker product of the upper and lower state matrices after the field has been included. afcf in calculations. Might not be what is expected.
        data['w3coeffs_ground'] = Ground_Level_HFS[0]
        data['w3coeffs_excited'] = Excited_Level_HFS[0]
        data['ZeemanSplits_G'] = Ground_Zeeman_HFS[-1]
        data['Red_Dipole'] = Z_sig_HFS[4]

        data['rad'] = Z_sig_HFS[2]  # Given out so that it can be used to plot specific polarizations.

    # If there is no nuclear spin in the input deck, do a similar calculation but no hpyerfine splitting ins included nor nuclear zeeman splitting.
    else:
        # Process the wavefunction matrix and energies. Preliminary sorting is done so that the final state order should be roughly as expected.
        Ground_Level = stateprocess(L_ground, S_ground, E_ground, Bfield_ext)
        Excited_Level = stateprocess(L_excited, S_excited, E_excited, Bfield_ext)
        data = {}
        # Preliminary workflow for attempting stark shift calculation. Presently doesn't provide reasonable results.

        # Use ground and excited states to calculate zeeman splitting and diangonalize the energy hamiltonian to obtain eigenvalues and vectors.
        Ground_Zeeman = Zeeman_func(Ground_Level, Bfield_ext)
        Excited_Zeeman = Zeeman_func(Excited_Level, Bfield_ext)
        # Use the new eigenvectors as the basis to calculate the relative signal strengths for the transitions
        Z_sig = Zeeman_signal(Ground_Level, Ground_Zeeman, Excited_Level, Excited_Zeeman, Bangle=Bext_angle, gamma=Polangle, Filter=Polfilter,T_Alpha = Talpha, T_Beta = Tbeta)
        Z_sig_norad = Zeeman_sig_norad(Ground_Level, Ground_Zeeman, Excited_Level, Excited_Zeeman)

        if do_low:
            Ground_Level = stateprocess(L_ground, S_ground, E_ground, Bfield_ext, sortE=sortE)
            Excited_Level = stateprocess(L_excited, S_excited, E_excited, Bfield_ext, sortE=sortE)
            Ground_Zeeman_Low = Zeeman_func(Ground_Level, Bfield_ext, Lowfield=True)
            Excited_Zeeman_Low = Zeeman_func(Excited_Level, Bfield_ext, Lowfield=True)
            Z_sig_Low = Zeeman_signal(Ground_Level, Ground_Zeeman_Low, Excited_Level, Excited_Zeeman_Low, Bangle=Bext_angle, gamma=Polangle, Filter=Polfilter,T_Alpha = Talpha, T_Beta = Tbeta)
            # Z_sig_Low = Zeemansig_Low(Ground_Level,Excited_Level,Bangle=Bext_angle,gamma=Polangle, Filter=Polfilter)
            data['signal_low'] = Z_sig_Low[0]
            data['wave_vac_low'] = Z_sig_Low[6]
            data['wave_air_low'] = Z_sig_Low[1]
            data['EigGround_low'] = Ground_Zeeman_Low
            data['EigExcited_low'] = Excited_Zeeman_Low
        if do_high:
            Ground_Level = stateprocess(L_ground, S_ground, E_ground, Bfield_ext, sortE=sortE)
            Excited_Level = stateprocess(L_excited, S_excited, E_excited, Bfield_ext, sortE=sortE)
            Ground_Zeeman_High = Zeeman_func(Ground_Level, Bfield_ext, Highfield=True, Eterm=EtermG)
            Excited_Zeeman_High = Zeeman_func(Excited_Level, Bfield_ext, Highfield=True,Eterm=EtermE)
            Z_sig_High = _zeeman_signal(Ground_Level, Ground_Zeeman_High, Excited_Level, Excited_Zeeman_High, Bangle=Bext_angle, gamma=Polangle, Filter=Polfilter,T_Alpha = Talpha, T_Beta = Tbeta, label_eigenpairs=False)
            # Z_sig_Low = Zeemansig_Low(Ground_Level,Excited_Level,Bangle=Bext_angle,gamma=Polangle, Filter=Polfilter)

            data['signal_high'] = Z_sig_High[0]
            data['wave_vac_high'] = Z_sig_High[6]
            data['wave_air_high'] = Z_sig_High[1]
            data['EigGround_high'] = Ground_Zeeman_High
            data['EigExcited_high'] = Excited_Zeeman_High
        data['EigGround'] = Ground_Zeeman
        data['EigExcited'] = Excited_Zeeman
        data['ZeemanSplits_G'] = Ground_Zeeman[-1]


        data['sigsum'] = Z_sig[7]
        data['SigNotNorm'] = Z_sig[8]
        data['bigstatelist'] = Z_sig[3]
        data['signal'] = Z_sig[0]
        data['wave_vac'] = Z_sig[6]
        data['wave_air'] = Z_sig[1]
        data['statelist_Ground'] = Ground_Level[-1]
        data['statelist_Excited'] = Excited_Level[-1]

        # Energy level output with lowfield approximation of zeeman splitting output.
        data['E0_Ground'] = Ground_Level[3]
        data['E0_Excited'] = Excited_Level[3]

        data['rad'] = Z_sig[2]
        data['Zsig_norad'] = Z_sig_norad
        # A_Ein isn't polarization normalized,
        data['A_Ein'] = [A_Ein(data['wave_air'][i], dips)/(2*data['bigstatelist'][i][1][0] + 1) for i, dips in enumerate(data['signal'])]
        data['A_Ein_polavg'] = [2*A_Ein(data['wave_air'][i], dips)/(3*(2*data['bigstatelist'][i][1][0] + 1)) for i, dips in enumerate(data['Zsig_norad'])]
        
        data['sigavgs'] = [sigs/(2*data['bigstatelist'][i][1][0] +1) for i,sigs in enumerate(data['signal']) ]

        data['f_osc'] = np.sum(data['signal'])*m_elec/(4*hbar*(np.pi**3))*((2*L_ground+1)/(2*L_excited+1))
        data['S_line'] = np.sqrt(np.sum([(3 /(16*q_charge*np.pi**4*nm_to_Hz(x)))*data['signal'][i] for i,x in enumerate(data['wave_vac'])]))*((L_ground+1)/(2*L_excited+1))
        data['afcf'] = Z_sig[4]

    # There are several instrument functions available for convolution. Lorrentzian, Skewed Lorrentzian, Doppler, and DopplerInstrum, which is a combination of a doppler and instrument (not a Pseudo Voigt though.)

    # These conditionals can very much be cleaned up and simplified in the future.

    if 'ion_vel' in Inputdeck:
        ion_vel = Inputdeck['ion_vel']

    else:
        ion_vel = 0
    if 'Convfxn' in Inputdeck:
        Convtype = Inputdeck['Convfxn']
        convtemp = 300
        SkewedConstants = [1, 1]
        if 'specstep' in Inputdeck:
            stepsize = Inputdeck['specstep']
        else:
            stepsize = 0.002 #Set a default step size 
        if Inputdeck['Convfxn'] == 'Gaussian' or Inputdeck['Convfxn'] == 'GaussianInstrum':
            #Requires Input Temperature for non Lorrentzian lineshapes
            if 'TempeV' in Inputdeck: #Allow for temp to be input in eV as well.
                convtemp = Inputdeck['TempeV']*11602 
            else:               
                convtemp = Inputdeck['Temp']
                
            SkewedConstants = [1, 1]

        if Inputdeck['Convfxn'] == 'Skewed' or Inputdeck['Convfxn'] == 'Lorrentzian':

            if 'Skewness' in Inputdeck:
                SkewedConstants = Inputdeck['Skewness']
            else:
                SkewedConstants = [1, 1]
        if Inputdeck['Convfxn'] == 'Voigt':
            if 'TempeV' in Inputdeck: #Allow for temp to be input in eV as well.
                convtemp = Inputdeck['TempeV']*11602 
            else:               
                convtemp = Inputdeck['Temp']

        if 'fxnwindow' in Inputdeck:
            fxnwindow = Inputdeck['fxnwindow']
        else:
            fxnwindow = 0.1  # This is how far the bin window will extend beyond the edge of the window. May need to be extended if large broadening.
    else:
        # Setting the convolution window for the function

        # This is the default configuration using a normal Lorrentzian lineshape.
        Convtype = 'Lorrentzian'
        SkewedConstants = [1, 1]
        convtemp = 300
        ion_vel = 0
        if 'fxnwindow' in Inputdeck:
            fxnwindow = Inputdeck['fxnwindow']
        else:
            fxnwindow = 0.1  # This is how far the bin window will extend beyond the edge of the window. May need to be extended if large broadening.
        if 'specstep' in Inputdeck:

            stepsize = Inputdeck['specstep']
        else:
            stepsize = 0.002222  # This is the resolution of the spectrometer pixels in nm.

    if 'specres' in Inputdeck:
        specres = Inputdeck['specres']
    else:
        specres = 10
        
    #Make the convolution based on the sticks.
    ConvolvedSpect, redsticks, binwinds = Convol_Spect(data['wave_air'], data['signal'],
                                                       specwindow, plotwindow, Inputdeck['amu'], stepsize,
                                                       Temperature_in=convtemp, functiontype=Convtype,
                                                       wind_size=fxnwindow, Skewness_consts=SkewedConstants, ionvel=ion_vel, specres=specres, bidir=bidirect)
    
    data['SpecOut'] = ConvolvedSpect
    data['reduced_sticks'] = redsticks
    data['bin_windows'] = binwinds

    pilist = np.array([x[1] for x in data['rad']]) == 0
    sigmaplus = np.array([x[1] for x in data['rad']]) == -1  # The sign is flipped.
    sigmaminus = np.array([x[1] for x in data['rad']]) == 1  # If you start from the bottom, you have the change the sign here
    otherlists = np.array([x[1] for x in data['rad']]) == 3

    # data['radlists'] = np.array([pilist,sigmaplus,sigmaminus])

    # data['radlists'] = np.array([pilist,sigmaplus,sigmaminus])
    data['pi_sticks'] = [data['wave_air'][pilist], data['signal'][pilist]]
    data['sigmaplus_sticks'] = [data['wave_air'][sigmaplus], data['signal'][sigmaplus]]
    data['sigmaminus_sticks'] = [data['wave_air'][sigmaminus], data['signal'][sigmaminus]]
    data['other_sticks'] = [data['wave_air'][otherlists], data['signal'][otherlists]]

    if do_low:
        ConvolvedSpectLow, redsticksLow, binwindsLow = Convol_Spect(data['wave_air_low'], data['signal_low'],
                                                                    specwindow, plotwindow,  Inputdeck['amu'], stepsize,
                                                                    Temperature_in=convtemp, functiontype=Convtype,
                                                                    wind_size=fxnwindow, Skewness_consts=SkewedConstants, ionvel=ion_vel, specres=specres, bidir=bidirect)

        data['SpecOutLow'] = np.array(ConvolvedSpectLow) #Ugly, but should fix an issue with plotting it?
        data['reduced_sticksLow'] = redsticksLow
        # data['bin_windows'] = binwindsLow

    if do_high:
        ConvolvedSpectHigh, redsticksHigh, binwindsHigh = Convol_Spect(data['wave_air_high'], data['signal_high'],
                                                                    specwindow, plotwindow,  Inputdeck['amu'], stepsize,
                                                                    Temperature_in=convtemp, functiontype=Convtype,
                                                                    wind_size=fxnwindow, Skewness_consts=SkewedConstants, ionvel=ion_vel, specres=specres, bidir=bidirect)

        data['SpecOutHigh'] = np.array(ConvolvedSpectHigh) #Ugly, but it should fix an issue with plotting it?
        data['reduced_sticksHigh'] = redsticksHigh
        # data['bin_windows'] = binwindsLow
    data['Input'] = Inputdeck
    return data

