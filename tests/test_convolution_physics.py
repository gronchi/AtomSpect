"""Independent flux, Doppler, and sampling checks for spectrum assembly."""
import os

os.environ.setdefault('MPMATH_NOGMPY', '1')
os.environ.setdefault('SYMPY_GROUND_TYPES', 'python')

import numpy as np
import pytest
from scipy import constants as sc

import AtomSpect as A
import AtomSpect.convolution as convolution


def integral(spectrum):
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return integrate(spectrum[1], spectrum[0]*1e-9)


def test_lorentzian_area_has_no_artificial_bridges_between_bins():
    spectrum, _, bins = A.Convol_Spect([600., 620.], [1., 1.], [599, 621],
                                      [590, 630], 40., .002, wind_size=.1)
    # Each retained Lorentzian has exact CDF mass 2/pi*atan(padding/HWHM).
    expected = 2 * 2/np.pi*np.arctan(.1/.001)
    assert integral(spectrum) == pytest.approx(expected, rel=1e-7)
    gap = (spectrum[0] > bins[0][1]) & (spectrum[0] < bins[1][0])
    assert np.all(spectrum[1][gap] == 0)
    assert np.all(np.diff(spectrum[0]) > 0)


def test_weak_lines_are_retained_without_changing_their_relative_area():
    spectrum, sticks, _ = A.Convol_Spect([600., 600.], [1., .001], [599, 601],
                                        [599, 601], 40., .002)
    assert len(sticks[0]) == 2  # Previously the second line was silently discarded.
    expected = 1.001 * 2/np.pi*np.arctan(.1/.001)
    assert integral(spectrum) == pytest.approx(expected, rel=1e-7)
    cut, selected, _ = A.Convol_Spect([600., 600.], [1., .001], [599, 601],
                                      [599, 601], 40., .002, relative_cutoff=.01)
    assert len(selected[0]) == 1
    assert integral(spectrum)/integral(cut) == pytest.approx(1.001, rel=1e-12)


@pytest.mark.parametrize('bidir, velocity', [(False, 150000.), (False, -150000.),
                                          (True, 150000.), (True, [150000., 70000.]),
                                          (True, 0.)])
def test_doppler_centres_survive_outside_rest_padding(bidir, velocity):
    rest, width = 600., .004
    spectrum, _, bins = A.Convol_Spect([rest], [1.], [599, 601], [599, 601],
                                      40., width, Temperature_in=0.,
                                      functiontype='GaussianInstrum', wind_size=.02,
                                      ionvel=velocity, bidir=bidir, specres=20)
    if bidir:
        v = np.atleast_1d(velocity)
        velocities = [v[0], -(v[-1])]
    else:
        velocities = [velocity]
    active = np.zeros(len(spectrum[0]), dtype=bool)
    for left, right in bins:
        active |= (spectrum[0] >= left) & (spectrum[0] <= right)
    sigma = width/np.sqrt(8*np.log(2))
    expected = sum(np.exp(-.5*((spectrum[0][active]-rest*(1+v/sc.c))/sigma)**2)
                   / (np.sqrt(2*np.pi)*sigma)*1e9/len(velocities) for v in velocities)
    np.testing.assert_allclose(spectrum[1][active], expected, rtol=1e-7, atol=1e-7)
    assert np.all(spectrum[1][~active] == 0)
    assert integral(spectrum) == pytest.approx(1., abs=1e-6)


def test_gaussian_mixture_uses_each_lines_temperature_mass_and_strength():
    centres = np.array([600., 600.012])
    weights = np.array([.3, 1.7])
    temperatures, masses = np.array([300., 4000.]), np.array([4., 40.])
    spectrum, _, bins = A.Convol_Spect(centres, weights, [599, 601], [599.98, 600.04],
                                      masses, .004, Temperature_in=temperatures,
                                      functiontype='GaussianInstrum')
    thermal_sigma = centres*np.sqrt(sc.k*temperatures/(masses*A.amu_kg*sc.c**2))
    sigma = np.sqrt(thermal_sigma**2 + .004**2/(8*np.log(2)))
    expected = sum(w*np.exp(-.5*((spectrum[0]-centre)/s)**2)/(np.sqrt(2*np.pi)*s)*1e9
                   for w, centre, s in zip(weights, centres, sigma))
    np.testing.assert_allclose(spectrum[1], expected, rtol=1e-7)


def test_chunked_profile_sum_preserves_flux_and_bounds_matrix_size(monkeypatch):
    original = convolution._profile
    shapes = []
    def measured(function, grid, centres, *args):
        shapes.append((grid.size, centres.size))
        return original(function, grid, centres, *args)
    args = ([600., 600.001, 600.002], [1., .4, .2], [599, 601], [599.99, 600.01], 40., .002)
    expected = A.Convol_Spect(*args)[0]
    monkeypatch.setattr(convolution, '_MAX_PROFILE_ELEMENTS', 2)
    monkeypatch.setattr(convolution, '_profile', measured)
    result = A.Convol_Spect(*args)[0]
    np.testing.assert_allclose(result, expected, rtol=1e-14)
    assert max(rows*columns for rows, columns in shapes) <= 2


@pytest.mark.parametrize('extra', [dict(atomic_weight_amu=0), dict(Temperature_in=-1),
                                  dict(Temperature_in=0), dict(ionvel=np.inf),
                                  dict(ionvel=sc.c), dict(ionvel=[1., 2.]),
                                  dict(relative_cutoff=-1)])
def test_invalid_physical_inputs_raise(extra):
    kwargs = dict(atomic_weight_amu=40., resolution_nm=.002, functiontype='Gaussian')
    kwargs.update(extra)
    with pytest.raises(ValueError):
        A.Convol_Spect([600.], [1.], [599, 601], [599, 601], **kwargs)


def test_negative_emission_strength_is_rejected():
    with pytest.raises(ValueError, match='nonnegative'):
        A.Convol_Spect([600.], [-1.], [599, 601], [599, 601], 40., .002)


def test_binning_keeps_duplicate_and_unsorted_metadata_aligned():
    waves = np.array([620., 600.002, 600., 620., 600.001])
    labels = np.arange(len(waves))
    result = A.make_stickbins(waves, labels, windsize=.01)
    for group, indices in zip(result['binned_waves'], result['binned_data'][0]):
        np.testing.assert_array_equal(group, waves[indices])
        assert np.all(np.diff(indices) > 0)
    assert sum(map(len, result['binned_waves'])) == len(waves)
