# AtomSpect refactor plan

Branch: `development` (unstable; based on `test/all-fixes`). `main` stays the stable line.

## 1. Problem

`AtomSpect.py` is a 3,150-line module holding constants, atomic physics, Hamiltonians,
dipole/polarization, hyperfine structure, Zeeman fans, lineshapes, convolution, plotting,
a Tk/matplotlib slider GUI, file I/O and the `AtomSpect_Main` driver (about 360 lines,
with many string-keyed options parsed inline). Consequences seen in the audit:

- Fixes had to be made in two parallel code paths (non-HFS vs `_HFS` duplicates of
  `stateprocess`, `Zeeman_func`, `Zeeman_signal`, `dipolestr`, eigenvector helpers).
- Importing the module pulls in matplotlib, tkinter and sympy even for batch use.
- Behaviour is controlled by dictionary key presence and stringly-typed flags.
- Unit conventions (nm vs m, cm^-1 vs MHz, HWHM vs FWHM) are implicit and repeated.
- Tests monkeypatch module globals (`AtomSpect.Zeeman_func`, `AtomSpect.muBcm`).

## Status

- Phase 0 done: `pyproject.toml` (version `0.1.0.dev0`), GitHub Actions workflow, golden-output
  tests (`tests/golden_cases.py`, `tests/test_golden.py`) and a public-API name test.
- Phase 1 done: `AtomSpect.py` is now the package `AtomSpect/`. Moved out unchanged:
  `constants`, `units`, `lineshapes`, `convolution`, `io`, `plotting/{core,slider,fan}`.
- Phase 2 done: physics moved unchanged into `angular` (Lande factors, CG, dipole strengths),
  `hfs`, `radiative`, `polarization`, `hamiltonian`, `levels`, `transitions` and `fan`. Imports
  are acyclic: hamiltonian -> angular; levels -> hamiltonian, hfs; transitions -> hamiltonian,
  polarization; fan -> levels. `AtomSpect/__init__.py` (~420 lines) now holds only
  `AtomSpect_Main`, its flag helper and the re-exports. The old HFS/non-HFS duplicates are still
  separate functions (phase 4).
- Phase 3 done: `config.py` holds `InputDeck` (typed fields, defaults, required-key check,
  unknown-key warning with a did-you-mean hint, `ConvolutionSettings`) and the flag parser.
  `spectrum.py` holds `AtomSpect_Main`, now a short pipeline over `_solve_hfs` / `_solve_fine`
  (+ `_approximation_signals`) and `_convolve`. Output, including `data['Input']`, is unchanged
  (golden tests). Defects found while doing it and deliberately preserved for now:
  without `Convfxn` the keys `Skewness`, `Temp` and `ion_vel` are silently ignored;
  `Convfxn='Gaussian'/'Voigt'` without `Temp`/`TempeV` raises KeyError late, after the solve;
  the slider writes a `PolAngle` key that the driver does not read.
- Phase 4 done (scoped): the shared parts of the HFS/non-HFS pairs are merged; the two state
  layouts stay separate because `[J,mJ,L,mL,S,mS]` and `[F,mF,J,mJ,I,mI,L,mL,S,mS]` need different
  quantum-number bookkeeping. Merged: `_zeeman_operator` (replaces four copies of an O(N^2) Python
  loop with one matrix product), `_definite_m_eigenvectors` (one helper for both cases, now with the
  offset-subtracted HFS form), `_signal_core` (one E1 pipeline; `Zeeman_signal`/`_zeeman_signal`/
  `Zeeman_signal_HFS` are thin wrappers), `_cg_matrix` and a single sorted/unsorted tail in
  `stateprocess` and `stateprocess_HFS`. `Zeeman_func(Lowfield=True, Highfield=True)` now raises
  instead of failing later with an unbound variable; `PsiHFS` is accepted but has no effect (both of
  its settings built the same Hamiltonian). The golden test now ignores eigenvector-gauge outputs
  (`afcf`, eigenvector matrices), which differ across LAPACK builds.
- Phase 5 done: `AtomSpect_Main` returns a `SpectrumResult` (a `dict` subclass in `result.py`) with
  attribute access, `copy()`, `to_dict()` and `describe()`, backed by `KEY_DESCRIPTIONS`; a test
  fails if a returned key is undocumented. String-key access is unchanged and not deprecated yet.
  `Savenpy_Dict` stores a plain dict so saved files do not depend on the class. The examples were
  not rewritten; they keep working through the dict interface.
  `plotting/slider.py` and `plotting/fan.py` import the driver late (inside functions)
  until phase 3 removes that dependency.
- Naming: the package keeps the name `AtomSpect` (not `atomspect`). On Windows/macOS a
  lowercase package would collide with the existing module name, and keeping it means
  `from AtomSpect import ...` in all examples and user scripts is unchanged, so no shim file
  is needed.

## 2. Target layout

```
AtomSpect/
  __init__.py          public API re-exports (keeps `from AtomSpect import ...` working)
  constants.py         physical constants, g factors
  units.py             energy/wavelength conversions, vacuum<->air
  angular.py           make_Jmj, CG/3j/6j wrappers, dipole strengths (shared, no HFS/non-HFS split)
  levels.py            Level / HFSLevel containers, stateprocess builders
  hamiltonian.py       Zeeman (low/high/intermediate) and HFS Hamiltonian assembly, eigh wrappers
  hfs.py               HFS interval/constants, Paschen-Back, FastZfan
  polarization.py      polarization(), eigenpair labelling, filter model
  transitions.py       Zeeman_signal / HFS signal -> sticks (wavelengths, strengths, labels)
  radiative.py         A_Ein, f_osc, S_line (flagged experimental until audit items 12-13 are decided)
  lineshapes.py        Gaussian, GaussianInstrum, Voigt, Lorentzian (unit area, FWHM)
  convolution.py       make_stickbins, Convol_Sticks, Convol_Spect, MultiSpec
  config.py            InputDeck dataclass: parsing, defaults, validation, flag handling
  spectrum.py          AtomSpect_Main orchestration; returns a result object (dict-compatible)
  fan.py               ZeemanFan data generation
  io.py                read_Spectra, read_dict, read_HFSInput, Savenpy_Dict, Loadnpy_Dict
  plotting/
    core.py            plotarray, PlotFunction, legend helpers
    fan.py             plotZfan
    slider.py          MakeSlider and helpers (only module importing tkinter)
```

Dependency direction is strictly downward: constants -> units -> angular -> levels ->
hamiltonian/hfs -> polarization -> transitions -> convolution -> spectrum -> plotting.
`plotting` and `io` depend on the core, never the reverse.

## 3. Principles

1. **Behaviour-preserving moves first.** Each step moves code without editing it, the full
   test suite (189 tests) and example regeneration must stay green before the next step.
2. **Keep the public API.** The `AtomSpect.py` shim re-exports every current public name so
   examples, user scripts and the published paper's instructions keep working.
3. **One implementation per physical step.** Collapse the non-HFS/HFS pairs into a single
   routine parameterised by the level container (I = 0 is the special case), done only
   after the move-only phase, with the existing tests as the guard.
4. **Explicit inputs.** Replace key-presence checks with an `InputDeck` dataclass:
   validated types, documented defaults, unknown-key warnings, one place for flag parsing
   (the Boolean-flag fix lives here), and `s_ground/s_excited` validation (S, not 2S+1).
5. **Make units explicit** in names (`wave_nm`, `wave_m`, `E_cm`, `A_MHz`) and convert at
   module boundaries only.
6. **Lazy GUI/plot imports** so `import AtomSpect` works headless without tkinter.

## 4. Phases

| Phase | Work | Exit criteria |
|---|---|---|
| 0 | Create package skeleton; add `pyproject.toml` (version `0.x.devN`), CI running pytest headless, pin sympy/numpy/scipy ranges | `pip install -e .` works; tests run in CI |
| 1 | Pure moves: constants, units, lineshapes, convolution, io, plotting, slider out of the monolith; shim re-exports | all tests + examples identical (PDF diffs only from timestamps) |
| 2 | Move physics: angular, levels, hamiltonian, hfs, polarization, transitions, fan | same; tests stop patching module globals and use public seams |
| 3 | `InputDeck` dataclass; split `AtomSpect_Main` into build levels -> solve -> signals -> convolve -> assemble result | `AtomSpect_Main(dict)` output unchanged; typos in keys now warn |
| 4 | Merge HFS/non-HFS duplicates (stateprocess, Zeeman_func, signal, dipolestr, eigenvector helpers) | regression tests pass; net line count drops |
| 5 | Result object with attribute access plus dict compatibility; deprecate string keys over time | examples migrated |
| 6 | Resolve open audit items: radiative outputs (12-13), `fxnwindow` wing truncation (10), HFS-constant estimator, `make_stickbins`/convolution performance | each has its own tests |
| 7 | Docs: API reference, README keyword table generated from `InputDeck`, move audit/PR notes out of the repo root into `docs/` | root contains only package, tests, examples, docs |

## 5. Test strategy

- Keep `tests/` as the safety net; add golden-output tests that run each example input deck
  through `AtomSpect_Main` and compare stick positions/strengths and convolved spectra
  (tolerance ~1e-10) against values saved before the refactor.
- Add import tests: `import AtomSpect` must not import tkinter or matplotlib.
- Add a test that every name exported by the old module still resolves via the shim.

## 6. Housekeeping

- Untracked clutter (audit reports, PR descriptions, scratch tests, evidence scripts) now lives
  in the git-ignored `audit/` folder. Still to relocate: `Flowchart.pptx`, `ExamplesList.xlsx`,
  `__pycache__`.
- Keep CRLF/LF consistent (add `.gitattributes`) so moves produce reviewable diffs.
- `Function Testing/` and `HFS Calcs/` scripts become examples or tests.
- Make `Examples/` scripts independent of the working directory (He example already is).

## 7. Risks

- Circular imports between plotting, fan and main: resolved by the downward layering above.
- Hidden global state (module constants patched by tests/users): keep constants in one
  module and read them at call time.
- Numerical drift while merging HFS and non-HFS paths: do it only in phase 4, guarded by the
  golden tests and the polarization/HFS suites.
- Upstream (`Lnofs/AtomSpect`) divergence: the layout change makes upstream merges hard,
  so upstream the fix branches first, and rebase `development` on upstream `main` before phase 1.

## 8. Releases

- `main`: stable, receives fixes only.
- `development`: unstable, receives the refactor; tagged `v0.x.devN` pre-releases.
- Promote to `main` after phase 4 (behaviour-equivalent, deduplicated) once examples and
  golden tests pass.
