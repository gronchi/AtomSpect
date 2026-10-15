"""Angular-momentum helpers: Lande factors, Clebsch-Gordan/3j wrappers, E1 dipole strengths."""
from functools import lru_cache

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
    """Return the electronic Lande g_J using the package g_L and g_S constants."""
    if J == 0:
        return 0
    else:
        return (g_L*(J*(J+1) - S*(S+1) + L*(L+1))/(2*J*(J+1)) + g_S*((J*(J+1) + S*(S+1) - L*(L+1))/(2*J*(J+1))))


def lande_gj(J, L, S):
    """Return the electronic Lande g_J for quantum numbers J, L, S; zero for J=0."""
    if J == 0: #There is no lande effect if J=0 due to a singularity (1/J)
        return 0
    else:
        # Use the same electron g factors as the exact Hamiltonian (g_S = 2.002319).
        return lande_gj2(J, L, S)


def lande_gf(F, I, J, L, S, g_I):
    """Return the weak-field hyperfine g_F; zero for F=0.

    F, I, J, L and S are angular momenta. g_I is returned by lande_gi;
    the nuclear contribution carries the same minus sign as the exact Hamiltonian."""
    thisgj = lande_gj(J, L, S)
    if F == 0:
        return 0
    else:
        thisgf = thisgj*(F*(F+1) - I*(I+1) + J*(J+1)) / (2*F*(F+1)) - g_I*(F*(F+1) + I*(I+1) - J*(J+1)) / (2*F*(F+1))
        return thisgf


def make_Jmj(J_am):
    """Return [J, mJ] pairs for a scalar or iterable J, with ascending mJ within each J."""
    if np.isscalar(J_am):
        Jtmp = [[float(J_am), float(x)] for x in np.linspace(-J_am, J_am, int(2*J_am+1))]
    else:
        Jtmp = [[float(jvs), float(x)] for jvs in J_am for x in np.linspace(-jvs, jvs, int(2*jvs+1))]
    return Jtmp


def _triangle(a, b, c):
    """True when a, b, c can couple: |a-b| <= c <= a+b with an integer sum."""
    return abs(a - b) <= c <= a + b and float(a + b + c).is_integer()


@lru_cache(maxsize=None)
def _wigner3j(j1, j2, j3, m1, m2, m3):
    if (m1 + m2 + m3 != 0 or not _triangle(j1, j2, j3)
            or abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3):
        return 0.0
    return float(w3j(j1, j2, j3, m1, m2, m3))


@lru_cache(maxsize=None)
def _wigner6j(j1, j2, j3, j4, j5, j6):
    if not (_triangle(j1, j2, j3) and _triangle(j1, j5, j6)
            and _triangle(j4, j2, j6) and _triangle(j4, j5, j3)):
        return 0.0
    return float(w6j(j1, j2, j3, j4, j5, j6))


def wigner3j(j1, j2, j3, m1, m2, m3):
    """Wigner 3j symbol as a float, exact (SymPy) and cached; zero when a selection rule fails."""
    return _wigner3j(*(float(x) for x in (j1, j2, j3, m1, m2, m3)))


def wigner6j(j1, j2, j3, j4, j5, j6):
    """Wigner 6j symbol as a float, exact (SymPy) and cached; zero when a triangle rule fails."""
    return _wigner6j(*(float(x) for x in (j1, j2, j3, j4, j5, j6)))


def CGCoeff(j1, m1, j2, m2, j3, m3):
    """Return the real Clebsch-Gordan coefficient <j1,m1;j2,m2|j3,m3> as a float."""
    thisCG = float(CG(j1, m1, j2, m2, j3, m3).doit())
    return thisCG


def CGw3(j1, m1, j2, m2, j3, m3):
    """Return <j1,m1;j2,m2|j3,m3> as a float using a Wigner 3j symbol."""
    return (-1)**(-j1 + j2 - m3)*np.sqrt(2*j3 + 1)*wigner3j(j1, j2, j3, m1, m2, -m3)


def dipolestr(LG, LE):
    """Return the signed angular E1 amplitude for a fine-structure basis pair.

    LG and LE each contain [J,mJ,L,mL,S,mS]. The radial reduced matrix
    element is omitted; square the amplitude to obtain an angular strength."""
    q = LE[1] - LG[1]
    phase = (-1)**(LE[2] + LG[4] - LE[1])
    return (phase*np.sqrt((2*LE[0] + 1)*(2*LG[0] + 1))
            * wigner3j(LG[0], 1, LE[0], LG[1], q, -LE[1])
            * wigner6j(LE[2], LE[0], LG[4], LG[0], LG[2], 1))


def dipolestr_HFS(LG, LE):
    """Return the signed angular E1 amplitude for a hyperfine basis pair.

    LG and LE each contain [F,mF,J,mJ,I,mI,L,mL,S,mS]. The radial reduced
    matrix element is omitted. The expression uses one 3j and two 6j symbols."""
    # Metcalf 4.33
    q = LE[1] - LG[1]
    phase = (-1)**(1 + LE[6] + LG[8] + LG[2] + LE[2] + LG[4] - LE[1])
    root = np.sqrt((2*LG[0] + 1)*(2*LE[0] + 1)*(2*LG[2] + 1)*(2*LE[2] + 1))
    return (phase*root*wigner3j(LG[0], 1, LE[0], LG[1], q, -LE[1])
            * wigner6j(LE[6], LE[2], LG[8], LG[2], LG[6], 1)
            * wigner6j(LE[2], LE[0], LG[4], LG[0], LG[2], 1))
