"""Unit-area line profiles; wavelengths in metres, widths are FWHM.

Profiles return densities per metre. Temperatures are in kelvin and masses
are in atomic mass units. Custom profiles used by Convol_Sticks should follow
the existing functions' wavelength, temperature, width and velocity arguments.
"""
import numpy as np
import scipy as scp
import scipy.special  # noqa: F401  (older SciPy does not load submodules on attribute access)

from .constants import amu_kg, c_light, kboltz


def Gaussian(x, x0, Temp_in, amu, v_ion=0, bidir=False):
    # Dickhauer, but added the ion velocity shift to the mean calculation
    # Gradic2022 including velocity addition+-
    """Return a unit-area Doppler Gaussian, evaluated at x in metres.

    x0 is the rest wavelength in metres, Temp_in is a positive temperature
    in K and amu is a positive mass number. v_ion is in m/s; with bidir=True,
    a scalar gives +/-v, or [v1,v2] gives +v1 and -v2, with equal weights.
    The returned density is per metre. Motion uses the first-order shift."""
    tdop = x0*np.sqrt(kboltz*Temp_in/(amu*amu_kg*c_light*c_light))
    if bidir == True:  # First try having v_ion be an array, so that the counter-streaming velocities can be different.
        try:
            mean1 = x0*(1 + v_ion[0]/c_light)
            mean2 = x0*(1 - v_ion[1]/c_light)

            xx1 = (x-mean1)/tdop
            xx2 = (x-mean2)/tdop
            G1 = (1/(np.sqrt(2*np.pi)*tdop))*np.exp(-0.5*xx1**2)
            G2 = (1/(np.sqrt(2*np.pi)*tdop))*np.exp(-0.5*xx2**2)
            G = (G1+G2)/2

        except:  # If not an array, just do plus and minus the same velocity
            mean1 = x0*(1 + v_ion/c_light)
            mean2 = x0*(1 - v_ion/c_light)

            xx1 = (x-mean1)/tdop
            xx2 = (x-mean2)/tdop
            G1 = (1/(np.sqrt(2*np.pi)*tdop))*np.exp(-0.5*xx1**2)
            G2 = (1/(np.sqrt(2*np.pi)*tdop))*np.exp(-0.5*xx2**2)
            G = (G1+G2)/2
    else:  # If bidir is False, do normal doppler shifted.
        mean = x0*(1 + v_ion/c_light)
        xx = (x-mean)/tdop
        G = (1/(np.sqrt(2*np.pi)*tdop))*np.exp(-0.5*xx**2)
    return G


def GaussianInstrum(x, x0, Temp_in, amu, pixel_size=0, v_ion=0, bidir=False):
    # Hey1996 for including instrum as W, but added the ion velocity shift to the mean calculation
    # Gradic2022 including velocity addition+-

    """Return a unit-area Gaussian with Doppler and instrument widths.

    x, x0 and pixel_size are in metres; pixel_size is the instrument FWHM.
    Temp_in (K) and amu set the Doppler variance, added in quadrature to
    the instrument variance. v_ion (m/s) and bidir follow Gaussian.
    The returned density is per metre."""
    tdop = (x0*np.sqrt(kboltz*Temp_in/(amu*amu_kg*c_light*c_light)))**2
    W = np.sqrt((pixel_size)**2 + 8*np.log(2)*tdop)
    if bidir == True:
        try:  # First try having v_ion be an array, so that the counter-streaming velocities can be different.
            mean1 = x0*(1 + v_ion[0]/c_light)
            mean2 = x0*(1 - v_ion[1]/c_light)
            xx1 = (x-mean1)/W
            xx2 = (x-mean2)/W

            G1 = (2*np.sqrt(np.log(2))/(np.sqrt(np.pi)*W))*np.exp(-4*np.log(2)*xx1**2)
            G2 = (2*np.sqrt(np.log(2))/(np.sqrt(np.pi)*W))*np.exp(-4*np.log(2)*xx2**2)
            G = (G1 + G2)/2
        except:  # If not an array, just do plus and minus the same velocity
            mean1 = x0*(1 + v_ion/c_light)
            mean2 = x0*(1 - v_ion/c_light)
            xx1 = (x-mean1)/W
            xx2 = (x-mean2)/W

            G1 = (2*np.sqrt(np.log(2))/(np.sqrt(np.pi)*W))*np.exp(-4*np.log(2)*xx1**2)
            G2 = (2*np.sqrt(np.log(2))/(np.sqrt(np.pi)*W))*np.exp(-4*np.log(2)*xx2**2)
            G = (G1 + G2)/2
    else:  # If bidir is False, do normal doppler shifted.
        mean = x0*(1 + v_ion/c_light)
        xx = (x-mean)/W
        G = (2*np.sqrt(np.log(2))/(np.sqrt(np.pi)*W))*np.exp(-4*np.log(2)*xx**2)
    return G


def Voigt(x, x0, Temp_in, amu, pixel_size=1, v_ion=0, bidir=False):
    """Return a unit-area Doppler/Lorentzian convolution in inverse metres.

    x, x0 and pixel_size are in metres. Temp_in (K) and amu set the
    Doppler standard deviation; pixel_size is the Lorentzian FWHM,
    passed to SciPy as HWHM=pixel_size/2. v_ion (m/s) and bidir follow
    Gaussian. A finite evaluation window omits the remaining wings."""
    tdop = (x0*np.sqrt(kboltz*Temp_in/(amu*amu_kg*c_light*c_light)))

    # Calculate a Voigt profile using the dopler broadening as the sigma and pixel size as the gamma for the lorrentzian
    if bidir == True:
        try:  # First try having v_ion be an array, so that the counter-streaming velocities can be different.
            mean1 = x0*(1 + v_ion[0]/c_light)
            mean2 = x0*(1 - v_ion[1]/c_light)
        except:  # If not an array, just do plus and minus the same velocity
            mean1 = x0*(1 + v_ion/c_light)
            mean2 = x0*(1 - v_ion/c_light)
        V1 = scp.special.voigt_profile(x-mean1, tdop, pixel_size/2)
        V2 = scp.special.voigt_profile(x-mean2, tdop, pixel_size/2)
        V = (V1 + V2)/2
    else:  # If bidir is False, do normal doppler shifted.
        mean = x0*(1 + v_ion/c_light)
        V = scp.special.voigt_profile(x-mean, tdop, pixel_size/2)
    # pixel_size is the Lorentzian FWHM; scipy takes the HWHM. Area-normalized like Gaussian.
    return V


def Lorrentzian(xrange, peak, pixel_size=0.002222, Skewness=[1, 1], v_ion=0, bidir=False):
    """Return a unit-area split Lorentzian in inverse metres.

    xrange, peak and pixel_size are in metres. The historical spelling is
    retained. pixel_size is the symmetric FWHM; Skewness=[left,right]
    multiplies the two half widths pixel_size/2. Both factors must be positive.
    v_ion is in m/s; bidir averages scalar +/-v or the pair [+v1,-v2].
    The profile has a single continuous peak and integrates to one on the
    infinite wavelength axis."""

    # pixel_size is the FWHM; the half-widths on each side are therefore half of it.
    l_w = Skewness[0]*pixel_size/2
    r_w = Skewness[1]*pixel_size/2

    def _lorentz(mean):
        dx = xrange - mean
        # Unit-area (split) Lorentzian: the area is pi*(l_w + r_w)/2 at unit peak.
        w = np.where(dx < 0, l_w, r_w)
        return w**2/(dx**2 + w**2)*2/(np.pi*(l_w + r_w))

    if bidir == True:
        try:  # First try having v_ion be an array, so that the counter-streaming velocities can be different.
            mean1 = peak*(1 + v_ion[0]/c_light)
            mean2 = peak*(1 - v_ion[1]/c_light)
        except:  # If not an array, just do plus and minus the same velocity
            mean1 = peak*(1 + v_ion/c_light)
            mean2 = peak*(1 - v_ion/c_light)
        SL = (_lorentz(mean1) + _lorentz(mean2))/2
    else:  # If bidir is False, do normal doppler shifted.
        mean = peak*(1 + v_ion/c_light)  # Doppler shift to peak
        SL = _lorentz(mean)
    return SL

def Normalize(list_in, scaling=1.0):
    """Scale an array to peak scaling; return zeros and empty arrays unchanged."""
    data = np.asarray(list_in)

    if data.size == 0:
        return data.copy()
    max_val = np.max(data)

    if max_val == 0:
        return data  # Return as-is if there's no signal to avoid div by zero

    return (data / max_val) * scaling
