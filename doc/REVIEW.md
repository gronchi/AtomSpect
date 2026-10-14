# Review and repair record

Covers the earlier audit of upstream commit `879ff7b` and a repository review on 2026-10-13 starting from `aab00bb`. The latest review examined the packaged solver, profiles, convolution, input/result handling, persistence, plotting, tests, and documentation. This record distinguishes existing repairs from the changes made in the latest review. The documentation layout follows `EZSSS/doc`: a theory manual, generated numerical evidence and figures, runnable scripts, build targets, and review notes.

## Repairs from the 2026-10-13 review

| Finding | Repair and evidence |
|---|---|
| Weak-field `lande_gf` added the nuclear contribution while the exact Hamiltonian subtracts it | Use the exact Hamiltonian's sign; independently compare `gF*mF` against projected matrix diagonals |
| `ZeemanSplits_G` returned eigenvectors instead of documented energies | Return eigenvalues for both fine and hyperfine spectra; test against the exact eigensystem |
| Sorted approximation bases overwrote metadata belonging to the exact eigensystem | Keep exact basis lists and zero-field energies when approximation spectra are requested |
| Overlapping padded bins produced repeated wavelengths and could duplicate sticks | Merge windows for gaps at most twice the padding; compare the curve against a direct sum of analytic profiles |
| Truly empty stick arrays failed at `max`, Python lists failed indexing, and integer wavelengths truncated fractional metadata | Convert sticks to floating arrays, broadcast floating mass/temperature arrays, and handle empty input before reduction |
| Coarse grids could contain zero samples; invalid widths and sampling factors failed cryptically | Retain at least two samples and reject nonpositive/nonfinite sampling inputs |
| `MultiSpec` ignored `TempeV` and silently dropped spectra when scale counts differed | Apply temperature precedence in kelvin and require one scale per spectrum |
| Slider temperature changes were overridden by an existing `TempeV` | Update both temperature representations when the deck already contains `TempeV` |
| Zero optical transmission divided by zero during normalization | Return zero strengths and a zero spectrum for a dark observation |
| Scalar HFS B constants failed, and malformed constant/energy arrays fell through broad exceptions | Validate one energy per J; expand scalar A/B constants consistently and reject wrong array lengths |
| A ground-only high-field fan discarded `EtermG` if `EtermE` was absent | Read term energies separately and require only the selected term's energy |
| NumPy spectrum saving used ambiguous truth testing; loading appended `.npy` twice | Accept array/dictionary spectra explicitly and load paths with or without the extension |
| Peak normalization failed for empty input | Return an empty array unchanged |
| Plotting NumPy bounds raised ambiguous truth errors; raw intensities were clipped to 1.1; missing style labels failed; GUI startup required an optional title | Accept array bounds and optional labels/title, autoscale raw curves, honor zero marker/measurement scales, and draw all spectra with a shared style |
| Nuclear spin zero divided by zero; NumPy scalar J was treated as iterable | Return zero nuclear g for zero spin/moment and accept NumPy scalar angular momenta |
| Invalid temperatures, masses, transmissions and split widths caused NaNs or late errors | Validate profile settings while parsing the input deck |

These cases are covered by [test_review_regressions.py](../tests/test_review_regressions.py). The sign check and convolution check also run during documentation regeneration. Public docstrings now state actual return structures, units, basis conventions, and limitations; placeholder `TYPE`/`DESCRIPTION` text and nonexistent parameters were removed. The air-to-vacuum helper is explicitly documented as approximate. `HZeeman` outputs a dimensionless operator, requiring multiplication by `muBcm*B`, rather than an energy matrix at 1 T. GUI startup is exercised with the noninteractive plotting backend; mouse events and dialogs remain untested.

## What the audit found, and the state on `development`

| Finding | State |
|---|---|
| Polarization labels were assigned to basis pairs, not eigenstate pairs, giving false $\pi/\sigma$ and $\|\Delta m\|>1$ labels | Fixed for non-HFS and HFS; labels now follow definite-$M$ eigenpairs |
| `PsiHFS=True` selected a broadcast branch with non-physical hyperfine energies; B constants were assigned to the wrong states | Fixed; both settings now give the same, correct Hamiltonian |
| Non-HFS high-field Hamiltonian was non-Hermitian (vector broadcast into a matrix) | Fixed; eigenvectors returned in the coupled basis |
| HFS high-field fan used the wrong branch, indices and units | Replaced by a defined hyperfine Paschen-Back approximation |
| Boolean approximation flags tested key presence | Fixed; flags parse Booleans and yes/no strings consistently |
| `s_ground`/`s_excited` documented as multiplicity but used as $S$ | Documentation corrected |
| Profile normalization differed between functions; widths mixed HWHM and FWHM | All profiles are unit-area and widths are FWHM |
| Empty or duplicated stick lists could crash or hang the convolution | Earlier duplicate-list repair extended in the latest review to truly empty arrays and overlapping windows |
| Default Lorentzian branch passed nanometres where metres were expected | Fixed |
| Low-field path used $g_S=2$ while the exact path used 2.002319 | Fixed |
| `ev_to_Hz` returned MHz | Renamed `ev_to_MHz` |
| `A_Ein`, `f_osc`, `S_line` are not valid absolute quantities | Returned only with `RadiativePlaceholders`, with a warning |
| Default profile handling and a missing temperature failed late | Temperature errors now fail during parsing; ion velocity is honored; `Skewness` is effective only for `Skewed`, as clarified in this review |
| Slider treated a 0 degree filter angle as "no filter" | Fixed |
| `fxnwindow` truncates Lorentzian and Voigt wings | A warning appears above 2% area loss; the default is unchanged |

## Still open

- The hyperfine-constant estimator (now `AtomSpect/hfs_estimate.py`, formerly the root script `HFS_Const_Calcs.py`) was moved into the package with its duplicated constants and unit helpers removed. Its numbers are pinned by `tests/test_hfs_estimate.py`. The quadrupole routines were repaired: the elementary charge was 1.602e-16 instead of 1.602e-19 and a missing 1/(4 pi eps0) and a J-to-Hz conversion were absorbed into odd factors, so `Bhfsconst` was about 11 times too large and `Bhfs_kopf` carried a spurious factor of 1.0546; both now evaluate the SI expression B = e Q V_zz, `qe` is gone, and the misapplied `gI_kopf` was removed. The A-constant model itself (hydrogenic with corrections) is unchanged and not validated beyond its table.
- Absolute radiative quantities need a reduced radial matrix element as input.
- Approximation spectra (`DoLowSig`, `DoHighSig`) are not implemented with hyperfine structure.
- Equal-energy eigenvectors receive a reconstructed basis, so only summed strength at coincident lines is meaningful.
- The non-HFS high-field approximation neglects fine structure and its polarization labels are not resolved by eigenpair.
- Convolution bins and `spec_window` use rest wavelengths. Large Doppler shifts can move peaks outside their evaluation windows. Gaussian tails can also be clipped without a warning; the Voigt warning estimates Lorentzian tails only. Increase the windows and check numerical convergence.
- `air_to_vac` evaluates the refractive index at the air wavelength rather than solving the inverse relation. Rounded legacy eV conversion constants also limit precision.
- `Zsig_norad` retains raw eigensystem bases; individual strengths at degeneracies are not guaranteed to be basis independent.
- `Calc_HFS_Vals` is reserved and ignored. Direct low-level profile calls require physically valid widths, temperatures, and velocity shapes; they are less thoroughly validated than the main input deck.
- The plotting GUI still starts from hard-coded settings, changes the input dictionary, and needs an interactive Tk/Matplotlib environment. `plotZfan` saves only the last figure when both terms are plotted.

## Changes that alter outputs

- Lorentzian and Voigt widths are half of what they were: `pixel_size` is now a FWHM. The example input decks that use `Skewness` were doubled to keep their fitted widths; user decks that relied on the old convention need the same change.
- Hyperfine spectra with the example constants for Rb changed when the Hamiltonian was repaired; the ground-state A constant in three decks was also off by $10^{3}$.
- Latest review: weak-field HFS slopes change with the corrected nuclear sign; `ZeemanSplits_G` changes shape from a matrix to an energy vector. Callers needing eigenvectors should use `EigGround[1]` in the non-HFS result or the lower-level solver directly for HFS.
- Latest review: overlapping convolution bins merge and grids can change; dark observations now return zeros. Sorted approximation requests retain exact basis metadata. Other returned keys and ordinary profile normalization are preserved.

The golden archive was updated selectively: eleven `main/.../ZeemanSplits_G` arrays and `levels/rbg/sortTrue[3]` (the weak-field energies affected by the nuclear sign). Every other archived array was retained, including basis-dependent quantities. The new regressions independently check the intended behavior; golden files alone do not establish correctness.

## Documentation reproducibility

Added separate `data` and `plots` targets, a Python manual builder for Windows, staged generation, an example that asserts finite/ordered output, and provenance with UTC time, platform, runtime versions, commit and source hash. CSV data now include both exact and approximation fan curves. The manual clarifies that `SpecOut` has a nm axis with densities per metre, that padding merges at twice `fxnwindow`, and that finite bins do not adapt to velocity shifts. It also records the corrected nuclear sign and dark-spectrum behavior.

## Visual style alignment with EZSSS

The first documentation pass aligned the folder structure. The subsequent style revision uses the actual `EZSSS_theory_manual.tex` preamble and layout conventions: separate cover, compact Roman-numbered contents, running section headers, microtypography, padded blue implementation notes and orange caveats, full-width booktabs tables, a TikZ calculation-flow diagram with a routine/section map, an options appendix, and a numbered bibliography with DOI links. Plot settings match EZSSS's serif fonts, Computer Modern mathematics, inward ticks, thin axes, compact dimensions, and blue/orange/aqua/grey colors. The documentation retains AtomSpect's own model and conventions.

Color roles were subsequently aligned as well: pi is blue and both sigma signs are orange, with line style distinguishing the signs. The Doppler-only reference is grey dashed, broadened profiles use EZSSS's blue sequence, and fan approximations use grey dashed lines. Matching a palette alone had left these meanings inconsistent.

The cover now contains the title and version details; the descriptive summary and contents appear together on physical page two. The PDF was rebuilt and those pages were checked visually and by text extraction.

## README language and examples

Rewrote the main README in consistent technical English, with current installation instructions, a self-contained Ar spectrum example, plotting and fan examples, input/output tables, density-unit guidance, and a concise statement of limitations. Removed stale claims about HFS polarization labeling, non-dipole emission, and implementation status. Updated the examples guide and retained the publication, data acknowledgments, contact, and license information. All three Python blocks in the main README executed successfully with the noninteractive plotting backend; their result arrays and shapes were checked. Local documentation links resolve.

- The He example decks used `amu: 2`; helium is mass 4. The Doppler width scales as $1/\sqrt{M}$, so the decks that use a Doppler profile (`GaussianInstrum` in `SpectPredict.py` and the two slider decks of `He_Example.py`) were too broad by $\sqrt2$. They now use 4.0026. The input check warns when a Doppler profile is selected without `amu`, because the default of 1 amu overstates the width by $\sqrt M$.

## Validation scope

Analytic checks run by `scripts/make_figures.py` (see the manual) and the repository test suite (polarization, high field, hyperfine, flags, profiles, golden outputs). Continuous integration runs the tests on Python 3.11 and 3.13. None of this validates the model against experiment, and none of it covers the regimes in the manual's list of known limits.

The 2026-10-13 local verification passed 311 tests, including 29 new regression cases, and all 11 documentation checks. The minimal example returned 135 eigenstate pairs and 330 continuous samples. Data-only regeneration, plotting from CSV, complete generation, and two-pass PDF compilation were exercised. After the style revision, the rebuilt manual has 14 pages including the cover and contents. Its rendered pages were visually reviewed against EZSSS, including equations, tables, the flowchart and all four numerical figures. It compiles with no LaTeX warnings, overfull boxes or underfull boxes. The style revision regenerated all figures and reran the 11 documentation checks. Consult `figures/data/validation_checks.csv` and `provenance.txt` for measured residuals and runtime versions.

Tests used the Python SymPy arithmetic backend because the local Anaconda `gmpy2` DLL failed during import. The remaining test warnings concern intentional wing truncation and opt-in radiative placeholders. Linux CI, other interpreters, interactive GUI events, full experimental example fitting, research notebooks, and the hyperfine estimator were not validated in this review.
