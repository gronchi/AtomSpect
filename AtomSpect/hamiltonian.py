"""Zeeman and hyperfine Hamiltonians, eigen-solvers and eigenvector helpers."""
import numpy as np

from .angular import lande_gj, lande_gj2
from .constants import c_light, g_L, g_S, muBcm


def _zeeman_operator(Psi, states, l_index, s_index, i_index=None, gI=0.0):
    """Zeeman operator (in units of muB*B) in the basis of the coupled states.

    The m quantum numbers are diagonal in the uncoupled basis, so each
    projection is Psi diag(m) Psi^T, with Psi the matrix of CG coefficients.
    `*_index` give the position of mL, mS (and mI) in each state vector.
    """
    def projected(index):
        m = np.array([state[index] for state in states], dtype=float)
        return (Psi*m) @ Psi.T

    H_Zeeman = g_S*projected(s_index) + g_L*projected(l_index)
    if i_index is not None:
        H_Zeeman = H_Zeeman - gI*projected(i_index)
    return H_Zeeman


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


def _definite_m_eigenvectors(level, zeeman):
    """Resolve definite-M eigenvector columns, preserving the energy-rank order.

    Returns (vectors, magnetic_m): the axial Hamiltonian does not couple
    different M (mJ without hyperfine structure, mF with it), so it is rebuilt
    from the eigensystem and diagonalized separately within each M block. This
    avoids tolerance-based resolution of mixed-M vectors near level crossings,
    and magnetic_m holds the exact M of every returned column. Equal-energy
    input columns receive a reconstructed basis; their individual identities
    are not preserved. The input eigensystem is not modified.
    """
    energies, vectors = np.asarray(zeeman[0]), np.asarray(zeeman[1])
    magnetic_m = np.asarray([state[1] for state in level[-1]])
    # Remove the common energy offset to avoid cancellation when small
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


# Names kept for the existing callers and tests.
_magnetic_eigenvectors = _definite_m_eigenvectors
_hfs_magnetic_eigenvectors = _definite_m_eigenvectors


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
    # Level=[CGmat, Eo_mat,E0_flat,bigstatelist]; bigstatelist entries are [J,mJ,L,mL,S,mS].
    Psi = Level_in[0]  # Wavefunction basis matrix, comprised of CG coefficients.

    # Check for High and Low field approximations
    if Lowfield and Highfield:
        raise ValueError('Zeeman_func accepts at most one of Lowfield and Highfield.')
    if not Lowfield and not Highfield:  # If both flags False - Do intermediate regime
        H_Zeeman = _zeeman_operator(Psi, Level_in[-1], l_index=3, s_index=5)
        scaledMat = Level_in[1] + Binput*H_Zeeman*muBcm

    elif Highfield:
        if Eterm is None:
            raise ValueError("Non-HFS high-field calculations require Eterm in cm^-1 (zero is valid).")
        H_ZeemanH = np.array([ZeemanHigh(x[5], x[3], Binput) for x in Level_in[-1]])
        # mL/mS occupy the columns of Psi. Both the term energy and Zeeman
        # shifts are diagonal in this uncoupled basis, not broadcast vectors.
        scaledMat = np.diag(Eterm + H_ZeemanH)

    else:  # Low field
        gjmat = np.array([lande_gj(x[0], x[2], x[4]) for x in Level_in[-1]])
        tmJMat = np.diagflat([x[1] for x in Level_in[-1]])
        scaledMat = Level_in[1] + Binput*(gjmat*tmJMat)*muBcm

    eigZ, eigvecZ = np.linalg.eigh(scaledMat)  # Using driver="evx" seems to change some of the signs of the resulting matrices.

    if Highfield:
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
    # Level=[CGmat, Eo_mat,E0_flat,bigstatelist]; bigstatelist entries are [J,mJ,L,mL,S,mS].
    H_Zeeman = _zeeman_operator(Level_in[0], Level_in[-1], l_index=3, s_index=5)
    return [Level_in[1], H_Zeeman]


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
    # Level=[w3mat, E0 matrix, HFS energies, ...]; states are [F,mF,J,mJ,I,mI,L,mL,S,mS].
    H_Zeeman = _zeeman_operator(Level_in[0], Level_in[-1], l_index=7, s_index=9, i_index=5, gI=gI)

    # The HFS shifts are diagonal in the |F,mF> basis, the same basis as Level_in[1] and H_Zeeman.
    # PsiHFS is accepted for backwards compatibility; both of its settings gave this Hamiltonian.
    scaledMat = Level_in[1] + np.diagflat(Level_in[2]) + Binput*H_Zeeman*muBcm

    eigZ, eigvecZ = np.linalg.eigh(scaledMat)
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
    H_Zeeman = _zeeman_operator(Level_in[0], Level_in[-1], l_index=7, s_index=9, i_index=5, gI=gI)

    out = {}
    out['E0'] = Level_in[1]
    out['H_Z'] = H_Zeeman
    out['PsiHFS'] = np.diagflat(Level_in[2])
    out['AHFS0'] = np.diagflat(Level_in[3])
    out['EHF0'] = Level_in[2]
    return out
