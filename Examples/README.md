# Examples

Start with the [minimal documentation example](../doc/scripts/example_minimal.py), which runs from the repository root and requires no measured data files:

```sh
python doc/scripts/example_minimal.py
```

The examples below demonstrate larger workflows. Install AtomSpect first, then run each script from any directory: the scripts change to their own folder, so their data files and output figures stay next to them. The C III example includes an optional NetCDF workflow using `xarray` and `h5netcdf`; CSV data are also provided.

| Script | Workflow |
|---|---|
| [Ar_Example.py](Ar_ex1/Ar_Example.py) | Overlay multiple Ar lines and compare calculated spectra with measurements |
| [CIII_ex1.py](CIII_ex1/CIII_ex1.py) | Combine lines with `MultiSpec`, apply velocity shifts and instrumental broadening, and compare temperature cases |
| [He_Example.py](He_ex1/He_Example.py) | Compare He polarization components, analyzer settings, and magnetic-field cases |
| [Rb_ex.py](Rb_ex1/Rb_ex.py) | Calculate hyperfine spectra and energy fans |
| [SpectPredict.py](SpectrometerPredict/SpectPredict.py) | Explore spectral features with field, temperature, width, and geometry sliders |

For example, from the repository root:

```sh
cd Examples/Rb_ex1
python Rb_ex.py
```

These scripts illustrate calculations and plotting; they are not a general validation against experiment. `other_sticks` denotes pairs without an E1 component and does not identify non-dipole emission. Some examples write plot files when run.

See the [main README](../README.md) for installation and units, the [theory manual](../doc/AtomSpect_theory_manual.pdf) for the model, and the [review record](../doc/REVIEW.md) for known limitations.
