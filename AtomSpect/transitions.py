"""Relative E1 strengths, wavelengths and polarization labels for Zeeman transitions."""
import numpy as np

from .angular import dipolestr, dipolestr_HFS
from .hamiltonian import _definite_m_eigenvectors
from .polarization import _eigenpair_polarization, polarization
from .units import Vac_to_air


def _transition_pairs(LevelG, LevelE):
    """Every (lower, upper) basis-state pair, with the lower state varying fastest.

    This is the basis order of kron(upper eigenvectors, lower eigenvectors).
    """
    return [[x, y] for y in LevelE[-1] for x in LevelG[-1]]


def _wavelengths(LZG, LZE):
    """Vacuum and air wavelengths (nm) for the eigenpairs, upper-then-lower order."""
    wave_vac = np.array([1e7/(y-x) for y in LZE[0] for x in LZG[0]])
    return wave_vac, Vac_to_air(wave_vac)


def _signal_core(LevelG, LZG, LevelE, LZE, dipole, Bangle, gamma, Filter, T_Alpha, T_Beta,
                 label_eigenpairs=True):
    """Shared E1 signal pipeline for the fine-structure and hyperfine cases.

    `dipole(lower_state, upper_state)` returns the reduced E1 factor of a
    basis-state pair. Returns [sigout, wave_air, rad, pairs, afcf, Signalcoeffa,
    wave_vac, sigsum, tsigout]; rad, strengths and wavelengths share upper-then-
    lower eigenpair order, pairs and Signalcoeffa keep the basis-pair order.
    """
    pairs = _transition_pairs(LevelG, LevelE)

    # Polarization of every basis pair (label 3: no E1 component).
    basis_rad = [polarization(upper[1], lower[1], Bangle, gamma, Pol_Filter=Filter,
                              TAlpha=T_Alpha, TBeta=T_Beta) for lower, upper in pairs]

    # Signal strength coefficients: radiation polarization term times dipole strength.
    dipoles = [0 if rad[1] == 3 else dipole(lower, upper) for (lower, upper), rad in zip(pairs, basis_rad)]
    Signalcoeffa = np.array([rad[0]*d for rad, d in zip(basis_rad, dipoles)])

    if label_eigenpairs:
        vectors_e, upper_m = _definite_m_eigenvectors(LevelE, LZE)
        vectors_g, lower_m = _definite_m_eigenvectors(LevelG, LZG)
        afcf = np.kron(vectors_e.conj(), vectors_g)
        rad = _eigenpair_polarization(afcf, [item[1] for item in basis_rad], dipoles,
                                      lower_m, upper_m, Bangle, gamma, Filter, T_Alpha, T_Beta)
    else:
        # Approximation spectra consume only strengths and wavelengths; their
        # eigenvectors do not necessarily use the coupled M basis above.
        afcf = np.kron(np.asarray(LZE[1]), np.asarray(LZG[1]))
        rad = basis_rad

    # Signal of each eigenpair: |column of the reduced dipole (afcf) matrix . coefficients|^2 (Isler1997).
    tsigout = np.abs(Signalcoeffa @ afcf)**2
    sigsum = np.sum(tsigout)
    sigout = tsigout/sigsum if sigsum > 0 else np.zeros_like(tsigout)

    wave_vac, wave_air = _wavelengths(LZG, LZE)
    return [sigout, wave_air, rad, pairs, afcf, Signalcoeffa, wave_vac, sigsum, tsigout]


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
    # Basis states are [J,mJ,L,mL,S,mS]. The dipole factor takes (upper, lower) here.
    return _signal_core(LevelG, LZG, LevelE, LZE, lambda lower, upper: dipolestr(upper, lower),
                        Bangle, gamma, Filter, T_Alpha, T_Beta, label_eigenpairs)


def Zeeman_sig_norad(LevelG, LZG, LevelE, LZE):
    """Relative E1 strengths without any polarization weighting."""
    pairs = _transition_pairs(LevelG, LevelE)
    Signalcoeffa = np.array([dipolestr(lower, upper) for lower, upper in pairs])
    afcf = np.kron(np.array(LZE[1]), np.array(LZG[1]))  # Tensor product after the magnetic field is taken into account.
    tsigout = np.sum(afcf*Signalcoeffa[:, None], axis=0)**2
    total = np.sum(tsigout)
    return tsigout/total if total > 0 else np.zeros_like(tsigout)


def Zeeman_signal_HFS(LevelG, LZG, LevelE, LZE, Bangle=90, gamma=0, Filter=False,T_Alpha = 1, T_Beta = 1):
    """Return relative E1 strengths with eigenpair-indexed polarization labels.

    rad, strengths and wavelengths share upper-then-lower eigenpair order.
    bflist and Signalcoeffa retain their basis-pair order. Degenerate columns
    may receive a reconstructed definite-mF basis rather than retaining their
    individual identities. Sigma stick keys retain the emitted-light convention.

    Basis states are [F,mF,J,mJ,I,mI,L,mL,S,mS]. Returns
    [sigout, wave_air, rad, bflist, afcf, Signalcoeffa, wave_vac].
    """
    return _signal_core(LevelG, LZG, LevelE, LZE, dipolestr_HFS,
                        Bangle, gamma, Filter, T_Alpha, T_Beta)[:7]
