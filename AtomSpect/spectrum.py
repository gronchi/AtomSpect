"""The AtomSpect_Main driver: build levels, solve, form sticks, convolve."""
import warnings

import numpy as np

from .angular import lande_gi
from .config import InputDeck, _approximation_enabled  # noqa: F401
from .constants import hbar, m_elec, q_charge
from .convolution import Convol_Spect
from .hamiltonian import Zeeman_func, Zeeman_func_HFS
from .levels import stateprocess, stateprocess_HFS
from .radiative import A_Ein
from .result import SpectrumResult
from .transitions import _zeeman_signal, Zeeman_sig_norad, Zeeman_signal, Zeeman_signal_HFS
from .units import nm_to_Hz


def _solve_hfs(deck):
    """Exact hyperfine Zeeman levels and E1 sticks (nuclear spin present)."""
    # Use spin, spin parity, and nuclear dipole moment to calculate lande g_i nuclear factor.
    thisg_I = lande_gi(deck.I_spin, deck.mu_I)

    # Process the wavefunction matrix and energies. Preliminary sorting is done so that the final state order should be roughly as expected.
    Ground_Level_HFS = stateprocess_HFS(deck.L_ground, deck.S_ground, deck.I_spin, deck.HFS_ConstsG, thisg_I, deck.E_ground, deck.Bfield)
    Excited_Level_HFS = stateprocess_HFS(deck.L_excited, deck.S_excited, deck.I_spin, deck.HFS_ConstsE, thisg_I, deck.E_excited, deck.Bfield)

    # Take the sorted states, the state matrix, and energy matrix and calculate the zeeman splitting (with HFS)
    Ground_Zeeman_HFS = Zeeman_func_HFS(Ground_Level_HFS, deck.Bfield, thisg_I, deck.PsiHFS)
    Excited_Zeeman_HFS = Zeeman_func_HFS(Excited_Level_HFS, deck.Bfield, thisg_I, deck.PsiHFS)
    # Use the eigenvectors and values from the diagonalization of the hamiltonian to calculate the relative signal strength. This function uses a combination of Isler1997 and Metcalf 4.32/4.33
    Z_sig_HFS = Zeeman_signal_HFS(Ground_Level_HFS, Ground_Zeeman_HFS, Excited_Level_HFS, Excited_Zeeman_HFS,
                                  Bangle=deck.Bangle, gamma=deck.Polangle, Filter=deck.Polfilter,
                                  T_Alpha=deck.Talpha, T_Beta=deck.Tbeta)

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
    data['ZeemanSplits_G'] = Ground_Zeeman_HFS[0]
    data['Red_Dipole'] = Z_sig_HFS[4]

    data['rad'] = Z_sig_HFS[2]  # Given out so that it can be used to plot specific polarizations.
    return data


def _approximation_signals(deck, data):
    """Optional low-field and high-field approximation sticks (non-HFS only).

    Returns the level lists built last (or None, None when no approximation is
    requested). The driver retains its exact basis metadata independently.
    """
    sig_kwargs = dict(Bangle=deck.Bangle, gamma=deck.Polangle, Filter=deck.Polfilter,
                      T_Alpha=deck.Talpha, T_Beta=deck.Tbeta)
    Ground_Level = Excited_Level = None
    if deck.do_low:
        Ground_Level = stateprocess(deck.L_ground, deck.S_ground, deck.E_ground, deck.Bfield, sortE=deck.sortE)
        Excited_Level = stateprocess(deck.L_excited, deck.S_excited, deck.E_excited, deck.Bfield, sortE=deck.sortE)
        Ground_Zeeman_Low = Zeeman_func(Ground_Level, deck.Bfield, Lowfield=True)
        Excited_Zeeman_Low = Zeeman_func(Excited_Level, deck.Bfield, Lowfield=True)
        Z_sig_Low = Zeeman_signal(Ground_Level, Ground_Zeeman_Low, Excited_Level, Excited_Zeeman_Low, **sig_kwargs)
        data['signal_low'] = Z_sig_Low[0]
        data['wave_vac_low'] = Z_sig_Low[6]
        data['wave_air_low'] = Z_sig_Low[1]
        data['EigGround_low'] = Ground_Zeeman_Low
        data['EigExcited_low'] = Excited_Zeeman_Low
    if deck.do_high:
        Ground_Level = stateprocess(deck.L_ground, deck.S_ground, deck.E_ground, deck.Bfield, sortE=deck.sortE)
        Excited_Level = stateprocess(deck.L_excited, deck.S_excited, deck.E_excited, deck.Bfield, sortE=deck.sortE)
        Ground_Zeeman_High = Zeeman_func(Ground_Level, deck.Bfield, Highfield=True, Eterm=deck.EtermG)
        Excited_Zeeman_High = Zeeman_func(Excited_Level, deck.Bfield, Highfield=True, Eterm=deck.EtermE)
        Z_sig_High = _zeeman_signal(Ground_Level, Ground_Zeeman_High, Excited_Level, Excited_Zeeman_High,
                                    label_eigenpairs=False, **sig_kwargs)
        data['signal_high'] = Z_sig_High[0]
        data['wave_vac_high'] = Z_sig_High[6]
        data['wave_air_high'] = Z_sig_High[1]
        data['EigGround_high'] = Ground_Zeeman_High
        data['EigExcited_high'] = Excited_Zeeman_High
    return Ground_Level, Excited_Level


def _add_radiative_placeholders(deck, data):
    """Opt-in A_Ein, f_osc and S_line style outputs; not physical absolute values."""
    warnings.warn("RadiativePlaceholders: A_Ein, A_Ein_polavg, sigavgs, f_osc and S_line are not valid "
                  "absolute radiative quantities (see the audit); they are kept only for comparison.",
                  UserWarning, stacklevel=4)
    # A_Ein isn't polarization normalized,
    data['A_Ein'] = [A_Ein(data['wave_air'][i], dips)/(2*data['bigstatelist'][i][1][0] + 1) for i, dips in enumerate(data['signal'])]
    data['A_Ein_polavg'] = [2*A_Ein(data['wave_air'][i], dips)/(3*(2*data['bigstatelist'][i][1][0] + 1)) for i, dips in enumerate(data['Zsig_norad'])]

    data['sigavgs'] = [sigs/(2*data['bigstatelist'][i][1][0] + 1) for i, sigs in enumerate(data['signal'])]

    data['f_osc'] = np.sum(data['signal'])*m_elec/(4*hbar*(np.pi**3))*((2*deck.L_ground+1)/(2*deck.L_excited+1))
    data['S_line'] = np.sqrt(np.sum([(3 /(16*q_charge*np.pi**4*nm_to_Hz(x)))*data['signal'][i] for i, x in enumerate(data['wave_vac'])]))*((deck.L_ground+1)/(2*deck.L_excited+1))


def _solve_fine(deck):
    """Exact Zeeman levels and E1 sticks without hyperfine structure."""
    # Process the wavefunction matrix and energies. Preliminary sorting is done so that the final state order should be roughly as expected.
    Ground_Level = stateprocess(deck.L_ground, deck.S_ground, deck.E_ground, deck.Bfield)
    Excited_Level = stateprocess(deck.L_excited, deck.S_excited, deck.E_excited, deck.Bfield)
    data = {}

    # Use ground and excited states to calculate zeeman splitting and diangonalize the energy hamiltonian to obtain eigenvalues and vectors.
    Ground_Zeeman = Zeeman_func(Ground_Level, deck.Bfield)
    Excited_Zeeman = Zeeman_func(Excited_Level, deck.Bfield)
    # Use the new eigenvectors as the basis to calculate the relative signal strengths for the transitions
    Z_sig = Zeeman_signal(Ground_Level, Ground_Zeeman, Excited_Level, Excited_Zeeman, Bangle=deck.Bangle,
                          gamma=deck.Polangle, Filter=deck.Polfilter, T_Alpha=deck.Talpha, T_Beta=deck.Tbeta)
    Z_sig_norad = Zeeman_sig_norad(Ground_Level, Ground_Zeeman, Excited_Level, Excited_Zeeman)

    _approximation_signals(deck, data)

    data['EigGround'] = Ground_Zeeman
    data['EigExcited'] = Excited_Zeeman
    data['ZeemanSplits_G'] = Ground_Zeeman[0]

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
    if deck.radiative_placeholders:
        _add_radiative_placeholders(deck, data)
    data['afcf'] = Z_sig[4]
    return data


def _convolve(deck, data, settings):
    """Convolve the sticks into spectra and split sticks by polarization."""
    def spectrum(waves, signals):
        return Convol_Spect(waves, signals, deck.spec_window, deck.plot_window, deck.raw['amu'], settings.stepsize,
                            Temperature_in=settings.temperature, functiontype=settings.function,
                            wind_size=settings.window, Skewness_consts=settings.skewness,
                            ionvel=settings.ion_vel, specres=settings.specres, bidir=deck.bidir)

    ConvolvedSpect, redsticks, binwinds = spectrum(data['wave_air'], data['signal'])
    data['SpecOut'] = ConvolvedSpect
    data['reduced_sticks'] = redsticks
    data['bin_windows'] = binwinds

    q_values = np.array([x[1] for x in data['rad']])
    pilist = q_values == 0
    sigmaplus = q_values == -1  # The sign is flipped.
    sigmaminus = q_values == 1  # If you start from the bottom, you have the change the sign here
    otherlists = q_values == 3

    data['pi_sticks'] = [data['wave_air'][pilist], data['signal'][pilist]]
    data['sigmaplus_sticks'] = [data['wave_air'][sigmaplus], data['signal'][sigmaplus]]
    data['sigmaminus_sticks'] = [data['wave_air'][sigmaminus], data['signal'][sigmaminus]]
    data['other_sticks'] = [data['wave_air'][otherlists], data['signal'][otherlists]]

    if deck.do_low:
        ConvolvedSpectLow, redsticksLow, _ = spectrum(data['wave_air_low'], data['signal_low'])
        data['SpecOutLow'] = np.array(ConvolvedSpectLow)  # Ugly, but should fix an issue with plotting it?
        data['reduced_sticksLow'] = redsticksLow

    if deck.do_high:
        ConvolvedSpectHigh, redsticksHigh, _ = spectrum(data['wave_air_high'], data['signal_high'])
        data['SpecOutHigh'] = np.array(ConvolvedSpectHigh)
        data['reduced_sticksHigh'] = redsticksHigh


def AtomSpect_Main(Inputdeck):
    """Compute relative E1 sticks and a broadened spectrum for two LS terms.

    Parameters
    ----------
    Inputdeck : dict
        Required: s_ground/s_excited are S (not 2S+1), l_ground/l_excited
        are L, E_ground/E_excited contain one energy per J in cm^-1,
        lowest J first, Bmag is a scalar field in T, and b_angle is the
        angle between field and line of sight in degrees.

        Pol_angle (degrees) enables a linear analyzer even when zero;
        TAlpha/TBeta are intensity transmissions (default 1). amu is the
        mass number (default 1). spec_window and plot_window are nm bounds,
        defaulting to the SpectrumData wavelength range or [1,1000].
        SpectrumData is [wavelengths_nm, signals].

        Convfxn selects Gaussian, GaussianInstrum, Voigt, Skewed, or the
        symmetric Lorrentzian (also the default and unknown-name fallback).
        Temp is in K; TempeV takes precedence, at 11602 K/eV. Temperature
        is required only by the Gaussian and Voigt profiles. specstep is
        the instrument FWHM and sampling pixel width in nm (0.002 when
        Convfxn is supplied, otherwise 0.002222); specres defaults to 10
        samples per pixel. fxnwindow pads bins by 0.1 nm. Skewness=[1,1]
        scales half widths only for Skewed. ion_vel is in m/s (default 0);
        bidir=False, or True to average +/-v or [+v1,-v2].

        I_spin enables HFS and requires mu_I (nuclear magnetons) and
        HFS_G/HFS_E (A constants in Hz, one per J or scalar). HFSB_G/E
        are optional B constants in Hz (default zero). PsiHFS is accepted
        for compatibility and does not change the Hamiltonian.
        Calc_HFS_Vals is reserved and ignored; constants are not estimated.

        DoLowSig/DoHighSig default False and accept Booleans or yes/no
        strings. Non-HFS DoHighSig requires EtermG/EtermE in cm^-1; zero
        is valid. HFS approximation spectra raise NotImplementedError.
        sortE orders approximation bases (default False). plottitle is
        plotting metadata. RadiativePlaceholders=False; enabling it emits
        unvalidated legacy radiative quantities with a warning.

    Returns
    -------
    SpectrumResult
        Dictionary with attribute access. signal, wave_air and wave_vac
        share upper-then-lower eigenpair order. Strengths sum to one when
        transmission is nonzero, or remain zero for a dark observation.
        SpecOut is [wavelength_nm, intensity_per_m]; integrate against
        metres. rad contains [polarization_amplitude, q_label] per pair;
        the sigma stick keys use the emitted-light sign convention.
        describe() explains all present keys. Input references the supplied
        dictionary, which receives an amu default if absent.

    Notes
    -----
    Only relative strengths are physical: radial matrix elements and level
    populations are absent. Finite bins truncate profiles; large Doppler
    shifts may leave the evaluation windows. Degenerate transitions have
    meaningful summed strengths, but no unique individual identity."""

    deck = InputDeck.from_dict(Inputdeck)
    data = _solve_hfs(deck) if deck.is_hfs else _solve_fine(deck)
    _convolve(deck, data, deck.convolution)
    data = SpectrumResult(data)
    data['Input'] = Inputdeck
    return data
