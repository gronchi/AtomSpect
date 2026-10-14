"""Angular-momentum helpers: Lande factors, Clebsch-Gordan/3j wrappers, E1 dipole strengths."""
import numpy as np
from sympy.physics.quantum.cg import CG
from sympy.physics.wigner import wigner_3j as w3j
from sympy.physics.wigner import wigner_6j as w6j

from .constants import g_L, g_S, m_elec, m_prot


def lande_gi(I, u_I):
    """Return the nuclear g factor in Bohr-magneton units.

    I is the nuclear spin and u_I the magnetic moment in nuclear magnetons.
    The Hamiltonian uses -g_I I_z. A spin-zero nucleus requires u_I=0."""
    # Technically, this should be the reduced mass here.
    if I == 0:
        if u_I != 0:
            raise ValueError('A spin-zero nucleus must have zero magnetic moment')
        return 0.0
    thisgi = (u_I/I)*(m_elec/m_prot)
    return thisgi


def lande_gj2(J, L, S):
    # Lande gJ as derived in Griffiths and others. Comes from projection of angular momentum and averaging of L and S
    #This has one less step of calculation than lande_gj function but should be equivalent.
    """Return the electronic Lande g_J using the package g_L and g_S constants."""
    if J == 0:
        return 0
    else:
        return (g_L*(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) + g_S*((J*(J+1) + S*(S+1) - L*(L+1))/(2*J*(J+1))))


def lande_gj(J, L, S):
    # Lande gJ as derived in Griffiths and others. Comes from projection of angular momentum and averaging of L and S
    """Return the electronic Lande g_J for quantum numbers J, L, S; zero for J=0."""
    if J == 0: #There is no lande effect if J=0 due to a singularity (1/J)
        return 0
    else:
        # Use the same electron g factors as the exact Hamiltonian (g_S = 2.002319).
        return lande_gj2(J, L, S)


def lande_gf(F, I, J, L, S, g_I):
    # Lande gF, comes from averaging I, L, and S similar to gJ.
    """Return the weak-field hyperfine g_F; zero for F=0.

    F, I, J, L and S are angular momenta. g_I is returned by lande_gi;
    the nuclear contribution carries the same minus sign as the exact Hamiltonian."""
    thisgj = lande_gj(J, L, S)
    #There is no lande effect if F=0, due to a singularity (1/F)
    if F == 0:
        return 0
    else:
        thisgf = thisgj*(F*(F+1) - I*(I+1) + J*(J+1)) / (2*F*(F+1)) - g_I*(F*(F+1) + I*(I+1) - J*(J+1)) / (2*F*(F+1))
        return thisgf


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
    """Return [J, mJ] pairs for a scalar or iterable J, with ascending mJ within each J."""
    if np.isscalar(J_am):
        Jtmp = [[float(J_am), float(x)] for x in np.linspace(-J_am, J_am, int(2*J_am+1))]
    else:
        Jtmp = [[float(jvs), float(x)] for jvs in J_am for x in np.linspace(-jvs, jvs, int(2*jvs+1))]
    return Jtmp


def CGCoeff(j1, m1, j2, m2, j3, m3):
    # Calculates Clebsch-Gordan Coefficient and outputs a float instead of sympy result
    """Return the real Clebsch-Gordan coefficient <j1,m1;j2,m2|j3,m3> as a float."""
    thisCG = float(CG(j1, m1, j2, m2, j3, m3).doit())
    return thisCG


def CGw3(j1, m1, j2, m2, j3, m3):
    # Clebsh Gordon Coeff using wigner3j
    """Return <j1,m1;j2,m2|j3,m3> as a float using a Wigner 3j symbol."""
    return ((-1)**(-j1 + j2 - m3))*np.sqrt(2*j3 + 1)*float(w3j(float(j1), float(j2), float(j3), float(m1), float(m2), -float(m3)))


def dipolestr(LG, LE):
    """Return the signed angular E1 amplitude for a fine-structure basis pair.

    LG and LE each contain [J,mJ,L,mL,S,mS]. The radial reduced matrix
    element is omitted; square the amplitude to obtain an angular strength."""
    thisq = LE[1] - LG[1]
    # if thisq == -1 or thisq== 0 or thisq== 1:

    tw3 = float(w3j(LG[0], 1, LE[0], LG[1], thisq, -LE[1]))
    tdp3 = ((-1)**(LE[2] + LG[4] - LE[1])) * np.sqrt((2*LE[0]+1)*(2*LG[0]+1))*tw3*float(w6j(LE[2], LE[0], LG[4], LG[0], LG[2], 1))

    # else:
    #     tdp3 = 0
    return tdp3


def dipolestr_HFS(LG, LE):
    """Return the signed angular E1 amplitude for a hyperfine basis pair.

    LG and LE each contain [F,mF,J,mJ,I,mI,L,mL,S,mS]. The radial reduced
    matrix element is omitted. The expression uses one 3j and two 6j symbols."""
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
