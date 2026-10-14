"""Regression tests for lineshape normalization, widths, binning and empty spectra."""
import os
import sys

os.environ.setdefault("MPMATH_NOGMPY", "1")
os.environ.setdefault("SYMPY_GROUND_TYPES", "python")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib
matplotlib.use("Agg")
import numpy as np
import AtomSpect as A

X0 = 600e-9
X = np.linspace(X0 - 2e-9, X0 + 2e-9, 400001)
FWHM = 5e-12


def area(y):
    return np.trapezoid(y, X) if hasattr(np, "trapezoid") else np.trapz(y, X)


def test_profiles_have_equal_unit_area():
    ys = [A.Gaussian(X, X0, 300, 40),
          A.GaussianInstrum(X, X0, 300, 40, pixel_size=FWHM),
          A.Voigt(X, X0, 300, 40, pixel_size=FWHM),
          A.Lorrentzian(X, X0, pixel_size=FWHM)]
    for y in ys[:3]:
        assert abs(area(y) - 1) < 1e-3
    # Lorentzian wings are long; the window only holds most of the area.
    assert 0.98 < area(ys[3]) <= 1.0


def test_lorentzian_and_voigt_fwhm_match_pixel_size():
    y = A.Lorrentzian(X, X0, pixel_size=FWHM)
    half = X[y >= y.max()/2]
    assert abs((half[-1] - half[0]) - FWHM) < 2e-14
    v = A.Voigt(X, X0, 1e-6, 40, pixel_size=FWHM)
    half = X[v >= v.max()/2]
    assert abs((half[-1] - half[0]) - FWHM) < 2e-13


def test_lorentzian_centre_is_single_valued():
    x = np.array([X0 - 1e-12, X0, X0 + 1e-12])
    y = A.Lorrentzian(x, X0, pixel_size=FWHM)
    assert y[1] >= y[0] and y[1] >= y[2] and abs(y[0] - y[2]) < 1e-6*y[1]


def test_fallback_convolution_matches_named_lorentzian():
    a = A.Convol_Sticks(600.0, [599.9, 600.1], 300, 40, 0.002, function="Lorrentzian")
    b = A.Convol_Sticks(600.0, [599.9, 600.1], 300, 40, 0.002, function="unknown")
    assert np.allclose(a[1], b[1])


def test_make_stickbins_duplicate_last_wavelength_terminates():
    w = np.array([0., .01, .01, 1.])
    out = A.make_stickbins(w, np.ones(4), windsize=.1)
    assert len(out["bin_windows"]) == 2
    assert [len(b) for b in out["binned_waves"]] == [3, 1]


def test_make_stickbins_empty():
    assert A.make_stickbins(np.array([]), np.array([]))["bin_windows"] == []


def test_empty_convolution_does_not_raise():
    waves = np.array([600.0, 600.1])
    spec, sticks, wins = A.Convol_Spect(waves, np.zeros(2), [500, 700], [590, 610], 40, 0.002,
                                        functiontype="Lorrentzian")
    assert np.all(spec[1] == 0)
    spec, _, _ = A.Convol_Spect(waves, np.ones(2), [100, 200], [590, 610], 40, 0.002,
                                functiontype="Lorrentzian")
    assert np.all(spec[1] == 0)


def test_lande_gj_uses_exact_electron_g():
    assert A.lande_gj(1.5, 1, 0.5) == A.lande_gj2(1.5, 1, 0.5)
    assert abs(A.lande_gj(0.5, 0, 0.5) - A.g_S) < 1e-12
