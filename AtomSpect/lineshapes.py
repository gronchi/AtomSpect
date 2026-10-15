"""Unit-area line profiles; wavelengths in metres, widths are FWHM.

Profiles return densities per metre. Temperatures are in kelvin and masses
are in atomic mass units. Custom profiles used by Convol_Sticks should follow
the existing functions' wavelength, temperature, width and velocity arguments.
"""
import numpy as np
import scipy as scp
import scipy.special  # noqa: F401  (older SciPy does not load submodules on attribute access)

from .constants import amu_kg, c_light, kboltz


def _doppler_sigma(x0, Temp_in, amu):
    """Doppler standard deviation in metres for rest wavelength x0 (m), Temp_in (K) and mass number amu."""
    return x0*np.sqrt(kboltz*Temp_in/(amu*amu_kg*c_light*c_light))


def _centres(x0, v_ion, bidir):
    """Doppler-shifted line centres and their weights.

    Without bidir, one centre shifted by v_ion (m/s). With bidir, two equal halves:
    a scalar v gives +v and -v; a pair [v1, v2] gives +v1 and -v2.
    """
    if not bidir:
        return [(x0*(1 + v_ion/c_light), 1.0)]
    v = np.atleast_1d(np.asarray(v_ion, dtype=float))
    v_plus, v_minus = (v[0], v[1]) if v.size > 1 else (v[0], v[0])
    return [(x0*(1 + v_plus/c_light), 0.5), (x0*(1 - v_minus/c_light), 0.5)]


def Gaussian(x, x0, Temp_in, amu, v_ion=0, bidir=False):
    """Return a unit-area Doppler Gaussian, evaluated at x in metres.

    x0 is the rest wavelength in metres, Temp_in is a positive temperature
    in K and amu is a positive mass number. v_ion is in m/s; with bidir=True,
    a scalar gives +/-v, or [v1,v2] gives +v1 and -v2, with equal weights.
    The returned density is per metre. Motion uses the first-order shift.
    All arguments broadcast, so several lines can be evaluated at once."""
    sigma = _doppler_sigma(x0, Temp_in, amu)
    return sum(weight*np.exp(-0.5*((x - mean)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
               for mean, weight in _centres(x0, v_ion, bidir))


def GaussianInstrum(x, x0, Temp_in, amu, pixel_size=0, v_ion=0, bidir=False):
    """Return a unit-area Gaussian with Doppler and instrument widths.

    x, x0 and pixel_size are in metres; pixel_size is the instrument FWHM.
    Temp_in (K) and amu set the Doppler variance, added in quadrature to
    the instrument variance. v_ion (m/s) and bidir follow Gaussian.
    The returned density is per metre."""
    W = np.sqrt(pixel_size**2 + 8*np.log(2)*_doppler_sigma(x0, Temp_in, amu)**2)  # FWHM
    return sum(weight*2*np.sqrt(np.log(2)/np.pi)/W*np.exp(-4*np.log(2)*((x - mean)/W)**2)
               for mean, weight in _centres(x0, v_ion, bidir))


def Voigt(x, x0, Temp_in, amu, pixel_size=1, v_ion=0, bidir=False):
    """Return a unit-area Doppler/Lorentzian convolution in inverse metres.

    x, x0 and pixel_size are in metres. Temp_in (K) and amu set the
    Doppler standard deviation; pixel_size is the Lorentzian FWHM,
    passed to SciPy as HWHM=pixel_size/2. v_ion (m/s) and bidir follow
    Gaussian. A finite evaluation window omits the remaining wings."""
    sigma = _doppler_sigma(x0, Temp_in, amu)
    return sum(weight*scp.special.voigt_profile(x - mean, sigma, pixel_size/2)
               for mean, weight in _centres(x0, v_ion, bidir))


def Lorrentzian(xrange, peak, pixel_size=0.002222, Skewness=[1, 1], v_ion=0, bidir=False):
    """Return a unit-area split Lorentzian in inverse metres.

    xrange, peak and pixel_size are in metres. The historical spelling is
    retained. pixel_size is the symmetric FWHM; Skewness=[left,right]
    multiplies the two half widths pixel_size/2. Both factors must be positive.
    v_ion is in m/s; bidir averages scalar +/-v or the pair [+v1,-v2].
    The profile has a single continuous peak and integrates to one on the
    infinite wavelength axis."""
    left, right = Skewness[0]*pixel_size/2, Skewness[1]*pixel_size/2  # half widths

    def split_lorentzian(mean):
        dx = xrange - mean
        w = np.where(dx < 0, left, right)
        return w**2/(dx**2 + w**2)*2/(np.pi*(left + right))

    return sum(weight*split_lorentzian(mean) for mean, weight in _centres(peak, v_ion, bidir))


def Normalize(list_in, scaling=1.0):
    """Scale an array to peak scaling; return zeros and empty arrays unchanged."""
    data = np.asarray(list_in)

    if data.size == 0:
        return data.copy()
    max_val = np.max(data)

    if max_val == 0:
        return data  # Return as-is if there's no signal to avoid div by zero

    return (data / max_val) * scaling
