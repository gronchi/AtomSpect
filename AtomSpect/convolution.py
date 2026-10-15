"""Stick binning and convolution into spectra."""
import numpy as np

from .config import EV_TO_K
from .constants import c_light
from .lineshapes import Gaussian, GaussianInstrum, Lorrentzian, Voigt, _centres


def MultiSpec(appendlist, scalelist):
    """Combine result sticks and their per-component temperature and mass.

    appendlist contains results with wave_air, signal and Input; scalelist
    must supply one relative scale per result. Return [waves_nm, signals,
    temperatures_K, masses_amu]. TempeV takes precedence over Temp; defaults
    are 300 K and mass 1. No additional strength normalization is applied."""
    if len(appendlist) != len(scalelist):
        raise ValueError('MultiSpec requires one scale per spectrum')
    columns = [[], [], [], []]

    for scaler, obj in zip(scalelist, appendlist):
        waves = np.asarray(obj['wave_air'], dtype=float)
        signals = np.asarray(obj['signal'], dtype=float)
        if waves.ndim != 1 or signals.shape != waves.shape:
            raise ValueError('Each spectrum requires matching one-dimensional wavelengths and signals')

        # Capture metadata for each component to allow different T/amu during convolution
        # Defaults to 300K and 1 amu if not found
        deck = obj['Input']
        t_val = deck['TempeV']*EV_TO_K if 'TempeV' in deck else deck.get('Temp', 300)
        a_val = deck.get('amu', 1)

        values = (waves, scaler*signals, np.broadcast_to(t_val, waves.shape),
                  np.broadcast_to(a_val, waves.shape))
        for column, value in zip(columns, values):
            column.append(value)

    return [np.concatenate(column).astype(float, copy=False) if column else np.empty(0)
            for column in columns]

def Convol_Sticks(x0, xwind, Temp_in, amu_in, stepsize_nm, function="", Skew_consts=[1, 1], vel_ion=0, specres=10, bidirect=False):
    """Evaluate one unit-area profile on a wavelength grid in nm.

    x0 and xwind are the rest centre and two evaluation bounds in nm.
    Temp_in is in K, amu_in is the mass number, and stepsize_nm is the
    instrument FWHM and nominal pixel width. specres sets samples per pixel;
    the endpoint-inclusive grid has at least two samples. Unknown function
    names use a symmetric Lorrentzian. Skew_consts applies only to Skewed.
    vel_ion is in m/s; bidirect averages the two counterstreaming components.

    Return [wavelength_nm, profile_per_m]. Integrate against wavelength in
    metres for unit area; finite windows truncate the analytic tails."""
    _validate_sampling(stepsize_nm, specres)
    _window(xwind)
    if not np.isscalar(x0) or not np.isfinite(x0) or x0 <= 0:
        raise ValueError('x0 must be a finite positive wavelength')
    _validate_profile_inputs(function, Temp_in, amu_in, Skew_consts, vel_ion, bidirect)
    xmin, xmax = np.min(xwind), np.max(xwind)
    waveout = np.linspace(xmin, xmax, max(2, int(round((xmax - xmin)/(stepsize_nm/specres)))))
    return [waveout, _profile(function, waveout, x0, Temp_in, amu_in, stepsize_nm,
                              Skew_consts, vel_ion, bidirect)]


# Profile name -> function of (wavelength_m, centre_m, temperature_K, mass_amu, width_m, skewness,
# velocity, bidir). Unknown names fall back to the symmetric Lorentzian.
_PROFILES = {
    'Gaussian': lambda x, x0, T, M, w, skew, v, bi: Gaussian(x, x0, T, M, v_ion=v, bidir=bi),
    'GaussianInstrum': lambda x, x0, T, M, w, skew, v, bi: GaussianInstrum(x, x0, T, M, pixel_size=w, v_ion=v, bidir=bi),
    'Voigt': lambda x, x0, T, M, w, skew, v, bi: Voigt(x, x0, T, M, pixel_size=w, v_ion=v, bidir=bi),
    'Skewed': lambda x, x0, T, M, w, skew, v, bi: Lorrentzian(x, x0, pixel_size=w, Skewness=skew, v_ion=v, bidir=bi),
    'Lorrentzian': lambda x, x0, T, M, w, skew, v, bi: Lorrentzian(x, x0, pixel_size=w, Skewness=[1, 1], v_ion=v, bidir=bi),
}


def _profile(function, wave_nm, centre_nm, temperature, mass, width_nm, skewness, v_ion, bidir):
    """Evaluate a named unit-area profile (per metre) with nm inputs; arguments broadcast."""
    profile = _PROFILES.get(function, _PROFILES['Lorrentzian'])
    return np.asarray(profile(1e-9*np.asarray(wave_nm), 1e-9*np.asarray(centre_nm), temperature, mass,
                              1e-9*width_nm, skewness, v_ion, bidir))


def make_stickbins(sticks_in, *data_arrays, windsize=.1):
    """Group sticks into disjoint evaluation windows, preserving aligned data.

    sticks_in is a finite one-dimensional wavelength array in nm; each
    additional data array must have the same shape. windsize is a positive
    padding in nm. Neighbouring sticks with gaps at most 2*windsize share a
    bin so evaluation windows cannot overlap. Return binned_waves,
    binned_data (one list of bins per data array), and bin_windows."""
    wavered = np.asarray(sticks_in, dtype=float)
    if not np.isfinite(windsize) or windsize <= 0:
        raise ValueError('windsize must be finite and positive')
    if wavered.ndim != 1 or not np.all(np.isfinite(wavered)):
        raise ValueError('sticks_in must be a finite one-dimensional array')
    if any(np.shape(arr) != wavered.shape for arr in data_arrays):
        raise ValueError('data arrays must match sticks_in')
    if wavered.size == 0:
        return {'binned_waves': [], 'binned_data': [[] for _ in data_arrays], 'bin_windows': []}
    order = np.argsort(wavered, kind='stable')
    srt = wavered[order]
    # Merge overlapping evaluation windows so each stick is used once and
    # the final wavelength grid remains ordered.
    breaks = np.where(np.diff(srt) > 2*windsize)[0]
    firsts = np.concatenate([[0], breaks + 1])
    lasts = np.concatenate([breaks, [len(srt) - 1]])
    wavewind = [[srt[a] - windsize, srt[b] + windsize] for a, b in zip(firsts, lasts)]

    wavebins = []
    binned_data = [[] for _ in data_arrays]
    arrays = [np.asarray(arr) for arr in data_arrays]
    for first, last in zip(firsts, lasts):
        # Preserve the original within-bin order without scanning all sticks
        # again for each separate window.
        indices = np.sort(order[first:last+1])
        wavebins.append(wavered[indices])
        for bins, arr in zip(binned_data, arrays):
            bins.append(arr[indices])

    return {'binned_waves': wavebins, 'binned_data': binned_data, 'bin_windows': wavewind}

def _validate_sampling(resolution, specres):
    """Reject invalid widths or sampling factors before allocating a grid."""
    if not np.isscalar(resolution) or not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('resolution_nm must be finite and positive')
    if not np.isscalar(specres) or not np.isfinite(specres) or specres <= 0:
        raise ValueError('specres must be finite and positive')


def _window(bounds):
    array = np.asarray(bounds, dtype=float)
    if array.shape != (2,) or not np.all(np.isfinite(array)) or array[0] == array[1]:
        raise ValueError('windows must contain two distinct finite bounds')
    return np.sort(array)


def _validate_profile_inputs(function, temperature, mass, skewness, velocity, bidir):
    if not np.all(np.isfinite(mass)) or np.any(np.asarray(mass) <= 0):
        raise ValueError('atomic masses must be finite and positive')
    if function in ('Gaussian', 'GaussianInstrum', 'Voigt'):
        temp = np.asarray(temperature)
        if not np.all(np.isfinite(temp)) or np.any(temp < 0) or (function == 'Gaussian' and np.any(temp == 0)):
            raise ValueError('Temperature must be finite and nonnegative (positive for Gaussian)')
    if function == 'Skewed':
        skew = np.asarray(skewness)
        if skew.shape != (2,) or not np.all(np.isfinite(skew)) or np.any(skew <= 0):
            raise ValueError('Skewness requires two finite positive factors')
    v = np.asarray(velocity)
    if (v.ndim > 1 or v.size not in ((1, 2) if bidir else (1,))
            or not np.all(np.isfinite(v)) or np.any(np.abs(v) >= c_light)):
        raise ValueError('velocity must be finite and subluminal: a scalar, or a pair with bidir=True')
    if not bidir and v.ndim != 0:
        raise ValueError('velocity must be a scalar unless bidir=True')


# Bound the temporary wavelength-by-line profile matrix (about 8 MiB in
# float64). Profiles can allocate a few additional arrays of this size.
_MAX_PROFILE_ELEMENTS = 1_000_000


def _sum_profiles(function, grid, centres, strengths, temperatures, masses,
                  width, skewness, velocity, bidir):
    values = np.zeros(len(grid))
    columns = min(len(centres), _MAX_PROFILE_ELEMENTS)
    rows = max(1, _MAX_PROFILE_ELEMENTS // columns)
    for start in range(0, len(grid), rows):
        stop = min(start+rows, len(grid))
        for first in range(0, len(centres), columns):
            last = first+columns
            profiles = _profile(function, grid[start:stop, None], centres[None, first:last],
                                temperatures[None, first:last], masses[None, first:last],
                                width, skewness, velocity, bidir)
            values[start:stop] += profiles @ strengths[first:last]
    return values

def Convol_Spect(waves_in, signal_in, spec_wind, plot_wind, atomic_weight_amu, resolution_nm,
                 Temperature_in=300, functiontype="", wind_size=.1,
                 Skewness_consts=[1, 1], ionvel=0, specres=10, bidir=False, *, relative_cutoff=0.):

    """Broaden and sum a relative stick spectrum in finite wavelength bins.

    waves_in and signal_in are matching one-dimensional arrays. spec_wind
    selects rest wavelengths using strict bounds; plot_wind clips the curve
    and displayed sticks. Bounds and resolution_nm are in nm. Mass and
    Temperature_in (K) can be scalars or arrays matching the sticks.
    resolution_nm is the instrument FWHM and sampling pixel width; specres
    is the positive samples-per-pixel factor. wind_size pads each merged bin.
    Skewness_consts applies only to Skewed. ionvel is in m/s and bidir averages
    counterstreaming profiles. Evaluation bins follow shifted centres.
    Unknown profiles fall back to Lorrentzian.

    Return (spectrum, reduced_sticks, bin_windows). spectrum contains
    [wavelength_nm, intensity_per_m]; integrate against metres. Components
    must have nonnegative strengths. All positive strengths are retained by
    default; optional relative_cutoff drops strengths <= cutoff*max(signal).
    Empty or fully rejected sticks
    return zero at the two plot bounds. Finite bins truncate wings and can
    are not renormalized. Zero samples immediately outside each bin prevent
    interpolation from adding artificial area across gaps. spec_wind still
    selects rest wavelengths, so include all contributing rest centres."""
    _validate_sampling(resolution_nm, specres)
    if not np.isfinite(wind_size) or wind_size <= 0:
        raise ValueError('wind_size must be finite and positive')
    waves_in = np.asarray(waves_in, dtype=float)
    signal_in = np.asarray(signal_in, dtype=float)
    if waves_in.ndim != 1 or signal_in.shape != waves_in.shape:
        raise ValueError('waves_in and signal_in must be matching one-dimensional arrays')
    if not np.all(np.isfinite(waves_in)) or not np.all(np.isfinite(signal_in)):
        raise ValueError('wavelengths and signals must be finite')
    if np.any(waves_in <= 0) or np.any(signal_in < 0):
        raise ValueError('wavelengths must be positive and signals nonnegative')
    if not np.isscalar(relative_cutoff) or not np.isfinite(relative_cutoff) or not 0 <= relative_cutoff <= 1:
        raise ValueError('relative_cutoff must be between zero and one')
    phys_left, phys_right = _window(spec_wind)
    left, right = _window(plot_wind)
    amus_arr = np.broadcast_to(np.asarray(atomic_weight_amu, dtype=float), waves_in.shape)
    temps_arr = np.broadcast_to(np.asarray(Temperature_in, dtype=float), waves_in.shape)
    _validate_profile_inputs(functiontype, temps_arr, amus_arr, Skewness_consts, ionvel, bidir)

    cutoff = relative_cutoff * np.max(signal_in) if signal_in.size else 0
    phys_reducer = (signal_in > cutoff) & \
                   (waves_in > phys_left) & (waves_in < phys_right)

    t_wav_phys, t_sig_phys = waves_in[phys_reducer], signal_in[phys_reducer]
    t_temp_phys, t_amu_phys = temps_arr[phys_reducer], amus_arr[phys_reducer]

    # Displayed sticks retain their rest wavelengths.
    plot_mask = (t_wav_phys >= left) & (t_wav_phys <= right)
    redsticks = [t_wav_phys[plot_mask], t_sig_phys[plot_mask]]

    # Nothing survived the filtering: return an empty (zero) spectrum instead of failing.
    if len(t_wav_phys) == 0:
        empty = [np.array([left, right]), np.zeros(2)]
        return empty, redsticks, []

    # Bin the observed centres. A counterstreaming line can occur in two
    # bins, but is evaluated only once at any wavelength, including when
    # its two stream windows overlap.
    shifted = np.concatenate([np.asarray(centre) for centre, _ in _centres(t_wav_phys, ionvel, bidir)])
    line_ids = np.tile(np.arange(len(t_wav_phys)), 2 if bidir else 1)
    tbins = make_stickbins(shifted, line_ids, windsize=wind_size)

    tempwavs, tempsigs = [np.array([left])], [np.zeros(1)]
    for ids, winds in zip(tbins['binned_data'][0], tbins['bin_windows']):
        if winds[1] < left or winds[0] > right:
            continue
        indices = np.unique(ids)
        twav = np.linspace(winds[0], winds[1], max(2, int(round(specres*(winds[1] - winds[0])/resolution_nm))))
        # Evaluate plot edges exactly, even if the window falls between grid samples.
        edges = [edge for edge in (left, right) if winds[0] <= edge <= winds[1]]
        twav = np.unique(np.concatenate((twav, edges)))
        twav = twav[(twav >= left) & (twav <= right)]
        tsig = _sum_profiles(functiontype, twav, t_wav_phys[indices], t_sig_phys[indices],
                             t_temp_phys[indices], t_amu_phys[indices], resolution_nm,
                             Skewness_consts, ionvel, bidir)
        tempwavs.extend((np.array([np.nextafter(winds[0], -np.inf)]), twav,
                         np.array([np.nextafter(winds[1], np.inf)])))
        tempsigs.extend((np.zeros(1), tsig, np.zeros(1)))

    tempwavs.append(np.array([right]))
    tempsigs.append(np.zeros(1))
    out_w, out_s = np.concatenate(tempwavs), np.concatenate(tempsigs)
    mask = (out_w >= left) & (out_w <= right)
    # Repeated plot endpoints prefer the evaluated profile over the initial
    # zero placeholder, without adding a duplicate wavelength.
    order = np.argsort(out_w[mask], kind='stable')
    grid, values = out_w[mask][order], out_s[mask][order]
    unique, starts = np.unique(grid, return_index=True)
    final_spectrum = [unique, np.maximum.reduceat(values, starts)]

    return final_spectrum, redsticks, tbins['bin_windows']
