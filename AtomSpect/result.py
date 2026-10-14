"""Result container returned by AtomSpect_Main."""

# What each key of the result holds. Keys that only exist in some modes are marked.
KEY_DESCRIPTIONS = {
    'AtomicComponents': 'Opt-in physical Model B component table from RadiativeData; separate from legacy normalized spectra.',
    'signal': 'Relative E1 strength per eigenstate pair (sums to 1, or zero for a dark observation).',
    'wave_air': 'Air wavelength of every pair [nm].',
    'wave_vac': 'Vacuum wavelength of every pair [nm].',
    'rad': 'Per pair [polarization factor, q label]; q = M_upper - M_lower, 3 = no E1 component.',
    'SpecOut': 'Convolved exact spectrum, [wavelength nm, intensity per metre] over plot_window.',
    'reduced_sticks': 'Sticks kept for plotting, [wavelength, signal] within plot_window.',
    'bin_windows': 'Wavelength windows in which the convolution was evaluated.',
    'pi_sticks': '[wavelength, signal] of the pi components.',
    'sigmaplus_sticks': '[wavelength, signal] of the sigma+ components (emitted-light sign).',
    'sigmaminus_sticks': '[wavelength, signal] of the sigma- components (emitted-light sign).',
    'other_sticks': '[wavelength, signal] of pairs without an E1 component (label 3).',
    'statelist_Ground': 'Basis states of the lower level.',
    'statelist_Excited': 'Basis states of the upper level.',
    'ZeemanSplits_G': 'Lower-level eigenvalues [cm^-1].',
    'Input': 'The input dictionary, including defaults filled in by the driver.',
    # Without hyperfine structure only
    'EigGround': 'Lower-level [eigenvalues, eigenvectors] (no hyperfine structure).',
    'EigExcited': 'Upper-level [eigenvalues, eigenvectors] (no hyperfine structure).',
    'bigstatelist': 'Basis-state pairs [lower, upper] in kron order (no hyperfine structure).',
    'sigsum': 'Sum of the unnormalized strengths before normalization.',
    'SigNotNorm': 'Unnormalized strengths.',
    'E0_Ground': 'Zero-field energies of the lower level [cm^-1].',
    'E0_Excited': 'Zero-field energies of the upper level [cm^-1].',
    'Zsig_norad': 'Strengths without polarization weighting.',
    'afcf': 'Kronecker product of the upper and lower eigenvector matrices.',
    'A_Ein': 'Experimental placeholder; not an absolute Einstein coefficient (opt-in: RadiativePlaceholders; see the audit).',
    'A_Ein_polavg': 'Experimental placeholder; not an absolute Einstein coefficient (opt-in: RadiativePlaceholders; see the audit).',
    'sigavgs': 'Signal divided by 2J_upper + 1 (basis-indexed) (opt-in: RadiativePlaceholders; see the audit).',
    'f_osc': 'Experimental placeholder; dimensionally invalid (opt-in: RadiativePlaceholders; see the audit).',
    'S_line': 'Experimental placeholder; dimensionally invalid (opt-in: RadiativePlaceholders; see the audit).',
    # Optional approximation spectra (DoLowSig / DoHighSig)
    'signal_low': 'Low-field approximation strengths (DoLowSig).',
    'wave_air_low': 'Low-field approximation air wavelengths (DoLowSig).',
    'wave_vac_low': 'Low-field approximation vacuum wavelengths (DoLowSig).',
    'EigGround_low': 'Low-field approximation lower-level eigensystem (DoLowSig).',
    'EigExcited_low': 'Low-field approximation upper-level eigensystem (DoLowSig).',
    'SpecOutLow': 'Convolved low-field approximation spectrum (DoLowSig).',
    'reduced_sticksLow': 'Low-field approximation sticks within plot_window (DoLowSig).',
    'signal_high': 'High-field approximation strengths (DoHighSig).',
    'wave_air_high': 'High-field approximation air wavelengths (DoHighSig).',
    'wave_vac_high': 'High-field approximation vacuum wavelengths (DoHighSig).',
    'EigGround_high': 'High-field approximation lower-level eigensystem (DoHighSig).',
    'EigExcited_high': 'High-field approximation upper-level eigensystem (DoHighSig).',
    'SpecOutHigh': 'Convolved high-field approximation spectrum (DoHighSig).',
    'reduced_sticksHigh': 'High-field approximation sticks within plot_window (DoHighSig).',
    # With hyperfine structure only
    'HFSG_Energy': 'Hyperfine energies of the lower level (hyperfine structure).',
    'HFSE_Energy': 'Hyperfine energies of the upper level (hyperfine structure).',
    'RedDipoleOp': 'Kronecker product of the eigenvector matrices (hyperfine structure).',
    'Red_Dipole': 'Same matrix as RedDipoleOp (hyperfine structure).',
    'w3coeffs_ground': 'CG coefficient matrix of the lower level (hyperfine structure).',
    'w3coeffs_excited': 'CG coefficient matrix of the upper level (hyperfine structure).',
}


class SpectrumResult(dict):
    """Dictionary of results that also allows attribute access.

    `result['signal']` and `result.signal` are equivalent, so existing code that
    indexes the result keeps working. `describe()` explains the keys present.
    """

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"result has no entry '{name}'; available: {', '.join(sorted(self))}") from None

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self))

    def copy(self):
        return type(self)(self)

    def to_dict(self):
        """Plain dict copy, e.g. for pickling without this class."""
        return dict(self)

    def describe(self):
        """Return {key: description} for the keys in this result."""
        return {key: KEY_DESCRIPTIONS.get(key, '(undocumented)') for key in self}
