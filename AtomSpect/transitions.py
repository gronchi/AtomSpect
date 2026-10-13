"""Relative E1 strengths, wavelengths and polarization labels for Zeeman transitions."""
import numpy as np

from .angular import dipolestr, dipolestr_HFS
from .hamiltonian import _hfs_magnetic_eigenvectors, _magnetic_eigenvectors
from .polarization import _eigenpair_polarization, polarization
from .units import Vac_to_air


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
