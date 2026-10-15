# AtomSpect documentation

The [theory manual](AtomSpect_theory_manual.pdf) states the model as implemented: LS-term Zeeman structure, hyperfine structure, E1 strengths, polarization, line profiles and convolution. It also lists the checks that are run and the regimes in which the results should not be trusted.

The manual now covers axial scalar/tensor dc Stark shifts, the revised Edlén vacuum/air conversion (dry air at 20 °C by default), and the separate calibrated atomic API. See the [DC field input example](../README.md#dc-electric-fields) and the [Rb-87 benchmarks](../tests/test_steck_rb87.py). With zero or omitted `Efield`, Stark code leaves the existing result unchanged. Nonzero electric fields require polarizability data and the exact solver; the field must follow the quantization axis.

The Rb-87 benchmarks check hyperfine intervals, Breit–Rabi energies, branch factors, representative dipole components, lifetime/oscillator conversions, and emission patterns. Stark checks compare against an independent spin Hamiltonian, weak-field tensor expectations, and exact zero-electric-field output equality. These tests supplement the 11 figure-generation checks; they do not certify every signed dipole-table entry or establish experimental accuracy.

The PDF follows EZSSS's visual style: a separate cover, a descriptive summary and Roman-numbered contents on page two, section headers, padded implementation and caveat boxes, a calculation-flow diagram and routine map, full-width tables, an input-options appendix, and numbered references. Numerical figures use its serif fonts, inward ticks, and line weights. Color meanings also follow EZSSS: blue for pi, orange for both sigma branches (distinguished by solid/dashed stems), grey dashed reference curves, and a blue sequence for broadened profiles.

## Start with a working example

From the repository root:

```sh
python doc/scripts/example_minimal.py          # prints the strongest components and the convolved peak
python doc/scripts/example_minimal.py --plot   # also shows the spectrum
```

It computes a He I 447 nm Zeeman pattern at 0.5 T, with polarization labels, and convolves it with a skewed Lorentzian. The result is a `SpectrumResult`, a dictionary whose entries are also attributes (`result.signal`, `result.wave_air`); `result.describe()` explains every key.

## What is here

| Path | Contents |
|------|----------|
| `AtomSpect_theory_manual.tex` / `.pdf` | Manual source and rebuilt PDF |
| `figures/*.pdf` | Four generated illustrations |
| `figures/data/*.csv` | Numerical data behind the figures and the analytic-check errors |
| `figures/data/validation_checks.tex` | The check table included in the manual |
| `figures/data/provenance.txt` | Generation time and package versions |
| `scripts/make_figures.py` | Runs the analytic checks, then writes the data and figures |
| `scripts/plot_figures.py` | Redraws all four figures from the CSV files without solving again |
| `scripts/build_manual.py` | Builds the PDF with Python on Windows or Unix |
| `scripts/example_minimal.py` | Runnable minimal example |

## Build and regenerate

Run these from the repository root:

```sh
make -C doc            # compile the manual from the checked-in source and figures
make -C doc check      # run the minimal example
make -C doc data       # regenerate numerical data and analytic checks
make -C doc plots      # redraw figures from existing CSV data
make -C doc figures    # rerun the analytic checks and redraw figures and data
make -C doc rebuild    # figures, then the manual
make -C doc clean      # remove LaTeX auxiliary files
```

Without `make` (for example on Windows):

```sh
python doc/scripts/example_minimal.py
python doc/scripts/make_figures.py --data-only
python doc/scripts/plot_figures.py
python doc/scripts/build_manual.py
# Or regenerate data, figures and the PDF in one command:
python doc/scripts/build_manual.py --rebuild
```

Requirements: `pdflatex` with the packages in the TeX preamble; Python with NumPy, SciPy, SymPy and matplotlib (the package's own dependencies). The Makefile defaults to `python` and `pdflatex`; override with `make -C doc figures PYTHON=/path/to/python`.

The main README shows the same calculation flow as a Mermaid flowchart, which GitHub renders directly. Update it by hand when the manual's TikZ diagram changes.

## Interpreting the checks and figures

`validation_checks.csv` lists 11 checks with measured errors and tolerances. The generator stages data and requested plots in a temporary directory; checks and calculations must succeed before it copies files into `figures/`. Checks cover weak-field $^3$P$_2$ splitting, the $^{85}$Rb 5S hyperfine interval, strength normalization, forbidden components, absence of $\pi$ light along the field, profile areas and widths, the weak-field HFS nuclear sign, overlapping-bin summation, and zero-transmission spectra. They validate those cases only; they do not establish agreement with experiment. Provenance records UTC time, versions, source commit, source-content hash, and figure settings. Both fan CSV files include the approximation curves shown in their PDFs.

The figures illustrate the implemented model. The energies and hyperfine constants come from the repository examples and were not independently validated here.

`signal` sums to one for a nonzero observation and is zero for a dark observation. `SpecOut` stores wavelengths in nm and intensities per metre; multiply intensities by `1e-9` for a density per nm. Finite windows and coarse sampling can lose area. Bins follow Doppler-shifted centres and merge when padded windows overlap; counterstreaming profiles are counted once per evaluation point. All positive strengths are retained unless `relative_cutoff` is supplied. Zeros immediately outside each bin prevent interpolation from adding area across empty gaps. `spec_window` and displayed sticks still use rest wavelengths, so include every contributing rest centre. The symmetric `Lorrentzian` ignores `Skewness`; use `Skewed` for unequal half widths. Increase `fxnwindow` and sampling to check convergence. The Voigt warning estimates Lorentzian-tail loss only.

`test_convolution_physics.py` independently checks finite-window Lorentzian area, weak-line weights, shifted/counterstreaming peaks, mixtures with different temperatures and masses, and the bounded size of the temporary profile matrix. Binning reuses sorted index groups, `MultiSpec` concatenates each output once, and profile summation is chunked to avoid allocating an unbounded wavelength-by-line array.

`ZeemanSplits_G` is a ground-energy vector in cm^-1; it previously contained eigenvectors. See the manual for compatibility conventions and remaining limitations. On this Windows installation, a broken Anaconda `gmpy2` DLL required `MPMATH_NOGMPY=1` and `SYMPY_GROUND_TYPES=python` for tests; documentation scripts select this backend by default. This is a local runtime workaround.

## Plotting behavior

`plotZfan(..., flags="GE", savefig=True, figname="fan")` saves both `fan_Ground.png` and `fan_Excited.png`, each with axis labels and a legend. Ground-only requests use the ground filename. `markercount=0` hides markers; `markercount=2` also works with weak-field curves.

`MakeSlider` honors `Bmax` and `Tmax`, starts from the input temperature within its allowed range, and retains polarization labels when updating. An empty component may gain transitions as the field changes. See the [main README](../README.md) for initial settings and backend requirements.

Convolution evaluates plot-window edges inside a bin directly, even when the window is narrower than the sample spacing. Edges outside all evaluated bins remain zero; finite-bin and sampling limitations still apply.
