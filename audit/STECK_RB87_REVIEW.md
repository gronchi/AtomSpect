# Rb-87 physics coverage review

Reviewed 2026-10-14 against Daniel A. Steck, *Rubidium 87 D Line Data*, revision 2.3.4 (8 August 2025), https://steck.us/alkalidata/rubidium87numbers.pdf. Optical pumping, driven optical Bloch dynamics, saturation, laser cooling, and trapping are excluded. Spontaneous-emission strengths and observation geometry remain in scope.

The static magnetic/E1 core is implemented, but AtomSpect does not implement every non-pumping topic in this reference. Agreement of matrix assembly with analytic formulas does not establish agreement at the precision of measured atomic parameters.

Implementation follow-up: axial scalar/tensor dc Stark shifts are now available in the ordinary spectrum, exact energy fans, and the calibrated atomic API. Zero or omitted electric field preserves the previous output exactly. The golden-reference archive has also been updated for the requested 20 C atmospheric default; the original review's failing-reference status below is retained as historical context.

## Coverage map

| Reference | Implementation | Assessment |
|---|---|---|
| Eq. (1) | No vapor-pressure model | Missing; not needed for normalized sticks, needed for vapor-density predictions |
| Eq. (2) | `units.py`, `transitions._wavelengths` | Frequency, wavenumber, and vacuum wavelength conversions present |
| Eq. (3) | `units.Vac_to_air`, `air_to_vac` | Revised Edlen relation and numerical inverse present; default dry air, 101325 Pa, 20 C |
| Eq. (4) | `atomic_data.RadiativeReference` | Physical oscillator-strength/rate conversion present in the newer API |
| Eqs. (5)-(6), (9) | No recoil helpers | Missing; small recoil offsets are not added to spectra |
| Eq. (7) | `lineshapes._centres`, `_doppler_sigma` | First-order Doppler shift and thermal broadening present |
| Eqs. (11)-(17) | `levels.py`, `hfs.Delta_E_HFS`, `atomic.add_hyperfine` | Angular bases and dipole/quadrupole HFS present; octupole C omitted |
| Eqs. (18)-(24) | `hamiltonian.py`, `angular.py`, `atomic.py` | Electronic/nuclear magnetic operators, Lande factors, weak-field limits present; input accuracy caveats below |
| Eq. (25) | `hamiltonian.ZeemanhighHFS`, `fan.py` | HFS Paschen-Back energy approximation present; main HFS approximation spectra unavailable |
| Eqs. (26)-(29) | `Zeeman_func_HFS`, `atomic.add_hyperfine` | Numerical Hamiltonians reproduce Breit-Rabi curvature; no dedicated public clock-shift helper |
| Eqs. (30)-(32) | `stark.stark_hamiltonian`, `atomic.add_dc_stark` | Axial scalar/tensor shifts and Stark/HFS mixing implemented; cross-J Stark tensors and nonparallel fields omitted |
| Eqs. (34)-(42), Tables 8-20 | `angular.dipolestr_HFS`, `atomic.py` | Wigner recoupling and E1 strengths present; tests sample components and exhaust Table 8, not every signed table entry |
| Eqs. (58)-(59) | `polarization.py`, `AtomicComponents.directional_rates` | Relative pi/sigma patterns present; newer API supplies absolute rates per solid angle |

## Findings affecting spectra

1. **Legacy magnetic constants limit precision.** `constants.muBcm` is 25.8 ppm above `AtomicConstants().mu_B_cm`. `lande_gi` uses a rounded electron/proton mass ratio 0.344% above the newer constants; this affects only the small nuclear contribution. The legacy solver derives electronic gJ from fixed LS values instead of accepting measured level-specific gJ. New Model A accepts measured gJ; Model C supports explicit magnetic matrix elements. Model B retains strict LS coupling. The nuclear sign convention is internally consistent: the package uses a positive moment-derived factor with a minus sign, whereas Steck's tabulated factor is negative.

   As a numerical sensitivity example, using the tabulated nuclear factor directly, the legacy ground-state clock curvature is 575.16870 Hz/G^2; the newer magnetic scale and measured ground gJ give 575.14595 Hz/G^2. This comparison isolates constants, rather than fitting the result. The rounded table value is 575.15 Hz/G^2.

2. **Physical rates exist separately from the ordinary spectrum.** `RadiativeReference` can normalize E1 amplitudes from A, f, gf, lifetime plus branching fraction, or reduced strength. `calculate_components` uses the physical frequency-cubed rate. `AtomSpect_Main` still returns normalized relative angular strengths. Its optional `RadiativeData` attaches `AtomicComponents` without rebuilding `SpecOut` from those rates. The old `A_Ein`/`A_Ein2` functions remain documented invalid absolute-rate placeholders.

3. **Linewidths are selected phenomenologically.** Lorentzian and Voigt profiles take a supplied width. Neither synthesis path automatically derives a natural width from the sum of upper/lower decay rates. For an isolated upper state with stable lower state, use ordinary-frequency FWHM A/(2*pi), not A. The D2 natural wavelength width is approximately 1.232e-5 nm. The current profile machinery can represent such a supplied narrow-line width, but does not calculate or combine it with instrumental and collision widths automatically.

4. **Population assumptions remain the caller's responsibility.** The legacy relative signal does not include upper-state populations. `AtomicComponents.emission_weights` and `synthesize_spectrum` accept explicit incoherent populations and optionally weight photon counts by h*nu for power. They do not solve population kinetics, equilibrium, or coherent density-matrix emission. Excluding pumping does not determine these populations.

5. **A single LS radial norm is an approximation.** The legacy pipeline and Model B share an orbital normalization across fine-structure branches. The tabulated D1/D2 dipoles imply orbital strengths differing by about 0.25%. Model B therefore cannot reproduce both measured central values exactly with its one norm; separately calibrated Model A lines or explicit Model C amplitudes can represent independent branch data. This is a limitation of the selected model, not an angular-momentum algebra error.

6. **No packaged precision Rb-87 dataset.** Energies, HFS constants, isotope mass, magnetic moments, and radiative references must be supplied. The existing Rb example uses nuclear spin 5/2 and Rb-85 constants. Merely changing its title or mass does not make it an Rb-87 calculation. Different isotopes must be calculated separately and combined with appropriate abundance/population weights.

7. **Air settings do not propagate from the input deck.** The conversion helpers expose atmospheric parameters; the ordinary spectrum calls the default conversion. Air temperature (20 C default) and atom kinetic temperature (`Temp`, K) are separate physical quantities. Relative humidity must be converted externally to water-vapor partial pressure. Rounded legacy eV constants also limit unit-conversion precision.

8. **Higher-order physics remains absent.** Both HFS implementations accept A/B only and omit octupole C and electronic-J off-diagonal HFS tensors. Diamagnetic terms, cross-J Stark tensors, and nonparallel electric/magnetic fields are absent. Axial scalar/tensor dc electric-field operators were added after the initial review. Numerical HFS mixing already produces the ordinary quadratic clock shift; that should not be confused with omitted diamagnetism. Partial decay sums require a complete set of decay channels before being interpreted as inverse lifetimes. No experimental validation or uncertainty propagation was established here.

The dipole normalization difference is intentional: for these transitions the newer API's standard electronic reduced dipole is sqrt(2) times the lower-J-normalized dipole tabulated by Steck. Directly inserting that tabulated dipole as the newer API's reduced amplitude would underestimate the strength by two. Tests convert conventions before comparing.

## Validation

Added `tests/test_steck_rb87.py`: 11 passing cases covering Rb-87 zero-field A/B intervals and centers of gravity; ground-manifold Breit-Rabi energies at 0, 1e-4, .01, .1 and 1 T; all Table 8 branch factors for every lower magnetic sublevel; representative Table 9/10 strengths; D1/D2 lifetime, oscillator-strength and reduced-dipole consistency; per-state decay sums; and absolute pi/sigma radiation patterns.

The Breit-Rabi assembly test deliberately uses identical magnetic constants on both sides. It validates diagonalization, mixing, units, and sign conventions, not measured g-factor accuracy. Squared component checks do not certify all signed table entries; existing coherent-operator and phase-invariance tests provide additional algebraic checks.

Full suite: **409 passed, 1 failed**. The failure is `tests/test_golden.py::test_outputs_match_reference`: its frozen air-wavelength/spectrum outputs predate the requested default change from 15 C to 20 C. A separate process overriding only the forward/reverse air conversions to 15 C passes both golden tests. The archive was left unchanged during this review. Production physics code was not changed by this review.

Follow-up Stark verification adds 19 passing tests: scalar field-squared and differential transition shifts; electronic tensor trace; independent spin-Hamiltonian eigenvalues through HFS mixing; weak-field tensor expectations; basis permutations; exact zero-field spectrum/fan equality; calibrated API compatibility; and frequency-cubed rate updates. The API requires Efield in V/cm and absolute alpha/h in Hz/(V/cm)^2, scalar or one per electronic J. It does not infer polarizabilities from the species.

Priorities for precision spectroscopy: use consistent magnetic constants and explicit measured gJ/gI inputs; connect calibrated rates and explicit populations to the desired spectrum; derive natural widths separately from instrumental widths; then add broader Stark or higher multipole HFS terms when required by the experiment.
