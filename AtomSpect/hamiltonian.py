"""Zeeman and hyperfine Hamiltonians, eigen-solvers and eigenvector helpers."""
import numpy as np

from .angular import lande_gj, lande_gj2
from .constants import c_light, g_L, g_S, muBcm


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
