# Examples

Start with the [minimal documentation example](../doc/scripts/example_minimal.py), which runs from the repository root and requires no measured data files:

```sh
python doc/scripts/example_minimal.py
```

Install AtomSpect first, then run the scripts from any directory. The Ar, C III, He, Rb, and spectrometer examples change to their own folder so data files and output figures stay next to them. The C III example includes an optional NetCDF workflow using `xarray` and `h5netcdf`; CSV data are also provided.

| Script | Workflow |
|---|---|
| [Ar_Example.py](Ar_ex1/Ar_Example.py) | Overlay multiple Ar lines and compare calculated spectra with measurements |
| [CIII_ex1.py](CIII_ex1/CIII_ex1.py) | Combine lines with `MultiSpec`, apply velocity shifts and instrumental broadening, and compare temperature cases |
| [He_Example.py](He_ex1/He_Example.py) | Compare He polarization components, analyzer settings, and magnetic-field cases |
| [Rb_ex.py](Rb_ex1/Rb_ex.py) | Calculate hyperfine spectra and energy fans |
| [SpectPredict.py](SpectrometerPredict/SpectPredict.py) | Explore spectral features with field, temperature, width, and geometry sliders |
| [LineShapeExamples.py](LineShapes/LineShapeExamples.py) | Compare five line profiles, bulk Doppler shifts, and equal counter-streaming populations |
| [PolarizationExample.py](Polarization/PolarizationExample.py) | Plot the current E1 polarization factors versus viewing angle, analyzer angle, and transmission |
| [HFSImport.py](HFSConstants/BulkHFSImport/HFSImport.py) | Estimate hyperfine A constants from CSV inputs and compare them with measured values |
| [ManualDictHFS.py](HFSConstants/ManualHFS/ManualDictHFS.py) | Explore hyperfine-constant estimates using explicit atomic dictionaries and electron coupling |

For example, from the repository root:

```sh
cd Examples/Rb_ex1
python Rb_ex.py
```

The line-shape and polarization examples replace the exploratory scripts formerly in `Function Testing/`. They use the current package functions, require no measured data, and display figures by default. Add `--no-show` to run without windows and `--output-dir` to save PDF and PNG figures to a chosen directory:

```sh
python Examples/LineShapes/LineShapeExamples.py --no-show --output-dir audit/lineshapes
python Examples/Polarization/PolarizationExample.py --no-show --output-dir audit/polarization
```

Line profiles are normalized to unit peak for comparison; the underlying functions return densities per metre. Polarization plots show intensity factors for individual components before multiplication by atomic E1 strengths. Existing line-shape figures were moved with the original example; the commands above generate fresh figures.

The former `HFS Calcs/` workflows and their CSV inputs, spreadsheet, and figures now live in `HFSConstants/`. These use the semi-empirical hyperfine-constant estimator, which is a research aid rather than a source of validated atomic constants. Run them from the repository root with:

```sh
python Examples/HFSConstants/ManualHFS/ManualDictHFS.py
python Examples/HFSConstants/BulkHFSImport/HFSImport.py --no-show --output-dir audit/hfs
```

The bulk example finds its input CSV beside the script and writes `HFSOutput2.csv` to `--output-dir` (or beside the script if omitted). Use `--input` for a different table, `--scan-corrections` to compare correction combinations, and `--verbose` to print cases with large errors. Omit `--no-show` to display the comparison plots. The manual example prints results and writes no files.

These scripts illustrate calculations and plotting; they are not a general validation against experiment. `other_sticks` denotes pairs without an E1 component and does not identify non-dipole emission. Some examples write plot files when run.

See the [main README](../README.md) for installation and units, the [theory manual](../doc/AtomSpect_theory_manual.pdf) for the model, and the [review record](../doc/REVIEW.md) for known limitations.
