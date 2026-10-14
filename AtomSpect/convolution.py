"""Stick binning and convolution into spectra."""
import numpy as np

from .config import EV_TO_K
from .lineshapes import Gaussian, GaussianInstrum, Lorrentzian, Voigt


def MultiSpec(appendlist, scalelist):
    """Combine result sticks and their per-component temperature and mass.

    appendlist contains results with wave_air, signal and Input; scalelist
    must supply one relative scale per result. Return [waves_nm, signals,
    temperatures_K, masses_amu]. TempeV takes precedence over Temp; defaults
    are 300 K and mass 1. No additional strength normalization is applied."""
    if len(appendlist) != len(scalelist):
        raise ValueError('MultiSpec requires one scale per spectrum')
    waves = np.array([])
    signals = np.array([])
    temps = np.array([])
    amus = np.array([])
    
    for scaler, obj in zip(scalelist, appendlist):
        waves = np.append(waves, obj['wave_air'])
        signals = np.append(signals, scaler * np.array(obj['signal']))
        
        # Capture metadata for each component to allow different T/amu during convolution
        # Defaults to 300K and 1 amu if not found 
        deck = obj['Input']
        t_val = deck['TempeV']*EV_TO_K if 'TempeV' in deck else deck.get('Temp', 300)
        a_val = obj['Input'].get('amu', 1)
        
        temps = np.append(temps, np.full_like(obj['wave_air'], t_val, dtype=float))
        amus = np.append(amus, np.full_like(obj['wave_air'], a_val, dtype=float))
        
    return [waves, signals, temps, amus]

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
    # hstep = stepsize_nm # step size in nm!! (Want 10x the steps per pixe for better shape?)
    _validate_sampling(stepsize_nm, specres)
    hstep = (1/specres)*stepsize_nm

    xmin = np.min(xwind)
    xmax = np.max(xwind)

    numwavesteps = max(2, int(round((xmax-xmin)/hstep)))
    waveout = np.linspace(xmin, xmax, numwavesteps)

    match function:
        case 'Skewed':
            # This one just does convolution over the entire range, should be fine because the range is fed by the binned stick windows.
            tlorry = Lorrentzian(1e-9*waveout, 1e-9*x0, pixel_size=1e-9*stepsize_nm, Skewness=Skew_consts, v_ion=vel_ion, bidir=bidirect)
            specout = tlorry
        case 'Lorrentzian':
            # This uses the skewed lorrentzian but sets to skewness such that the result is a normal lorrentzian function.
            tlorry = Lorrentzian(1e-9*waveout, 1e-9*x0, pixel_size=1e-9*stepsize_nm, Skewness=[1, 1], v_ion=vel_ion, bidir=bidirect)
            specout = tlorry
        case 'Gaussian':
            tlorry = Gaussian(1e-9*waveout, 1e-9*x0, Temp_in, amu_in, v_ion=vel_ion, bidir=bidirect)
            # tlorry[tlorry>1] =1 #Sometimes an odd range for the lorrentzian causes the central peak to be twice as high for a single point, this should correct for that.
            specout = tlorry
        case 'GaussianInstrum':
            tlorry = GaussianInstrum(1e-9*waveout, 1e-9*x0, Temp_in, amu_in, pixel_size=1e-9*stepsize_nm, v_ion=vel_ion, bidir=bidirect)
            # tlorry[tlorry>1] =1 #Sometimes an odd range for the lorrentzian causes the central peak to be twice as high for a single point, this should correct for that.
            specout = tlorry
        case 'Voigt':

            tlorry = Voigt(1e-9*waveout, 1e-9*x0, Temp_in, amu_in, pixel_size=1e-9*stepsize_nm, v_ion=vel_ion, bidir=bidirect)
            # tlorry[tlorry>1] =1 #Sometimes an odd range for the lorrentzian causes the central peak to be twice as high for a single point, this should correct for that.
            specout = tlorry

        case _:
            # If no function is otherwise given, default to Lorrentzian
            # This uses the skewed lorrentzian but sets to skewness such that the result is a normal lorrentzian function.
            tlorry = Lorrentzian(1e-9*waveout, 1e-9*x0, pixel_size=1e-9*stepsize_nm, Skewness=[1, 1], v_ion=vel_ion, bidir=bidirect)
            specout = tlorry

    return [waveout, np.array(specout)]

def make_stickbins(sticks_in, *data_arrays, windsize=.1):
    #This function does the dynamical windowing that allows for the fast optimization. It makes the bins of "nearby" signal sticks
    #And convolution only takes place in these windows. Sticks closer than windsize to their neighbour share a bin.
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
    srt = np.sort(wavered)
    # Merge overlapping evaluation windows so each stick is used once and
    # the final wavelength grid remains ordered.
    breaks = np.where(np.diff(srt) > 2*windsize)[0]
    firsts = np.concatenate([[0], breaks + 1])
    lasts = np.concatenate([breaks, [len(srt) - 1]])
    wavewind = [[srt[a] - windsize, srt[b] + windsize] for a, b in zip(firsts, lasts)]

    wavebins = []
    binned_data = [[] for _ in data_arrays]
    for windows in wavewind:
        mask = (wavered > windows[0]) & (wavered < windows[1])
        wavebins.append(wavered[mask])
        for i, arr in enumerate(data_arrays):
            binned_data[i].append(np.array(arr)[mask])
            
    return {'binned_waves': wavebins, 'binned_data': binned_data, 'bin_windows': wavewind}

def _validate_sampling(resolution, specres):
    """Reject invalid widths or sampling factors before allocating a grid."""
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('resolution_nm must be finite and positive')
    if not np.isfinite(specres) or specres <= 0:
        raise ValueError('specres must be finite and positive')

def Convol_Spect(waves_in, signal_in, spec_wind, plot_wind, atomic_weight_amu, resolution_nm, 
                 Temperature_in=300, functiontype="", wind_size=.1, 
                 Skewness_consts=[1, 1], ionvel=0, specres=10, bidir=False):

    # 1. Resolve inputs into arrays
    """Broaden and sum a relative stick spectrum in finite wavelength bins.

    waves_in and signal_in are matching one-dimensional arrays. spec_wind
    selects rest wavelengths using strict bounds; plot_wind clips the curve
    and displayed sticks. Bounds and resolution_nm are in nm. Mass and
    Temperature_in (K) can be scalars or arrays matching the sticks.
    resolution_nm is the instrument FWHM and sampling pixel width; specres
    is the positive samples-per-pixel factor. wind_size pads each merged bin.
    Skewness_consts applies only to Skewed. ionvel is in m/s and bidir averages
    counterstreaming profiles. Unknown profiles fall back to Lorrentzian.

    Return (spectrum, reduced_sticks, bin_windows). spectrum contains
    [wavelength_nm, intensity_per_m]; integrate against metres. Components
    at or below 0.05*max(signal)/N are removed. Empty or fully rejected sticks
    return zero at the two plot bounds. Finite bins truncate wings and can
    miss peaks Doppler-shifted outside the rest-wavelength padding."""
    _validate_sampling(resolution_nm, specres)
    if not np.isfinite(wind_size) or wind_size <= 0:
        raise ValueError('wind_size must be finite and positive')
    waves_in = np.asarray(waves_in, dtype=float)
    signal_in = np.asarray(signal_in, dtype=float)
    if waves_in.ndim != 1 or signal_in.shape != waves_in.shape:
        raise ValueError('waves_in and signal_in must be matching one-dimensional arrays')
    if not np.all(np.isfinite(waves_in)) or not np.all(np.isfinite(signal_in)):
        raise ValueError('wavelengths and signals must be finite')
    for window in (spec_wind, plot_wind):
        if np.size(window) != 2 or not np.all(np.isfinite(window)) or np.min(window) >= np.max(window):
            raise ValueError('windows must contain two distinct finite bounds')
    amus_arr = np.broadcast_to(np.asarray(atomic_weight_amu, dtype=float), waves_in.shape)
    temps_arr = np.broadcast_to(np.asarray(Temperature_in, dtype=float), waves_in.shape)

    # 2. Physics Reduction (spec_window) - used for the actual math
    cutoff = 0.05 * np.max(signal_in)/len(signal_in) if signal_in.size else 0
    phys_reducer = (signal_in > cutoff) & \
                   (waves_in > np.min(spec_wind)) & (waves_in < np.max(spec_wind))
    
    t_wav_phys, t_sig_phys = waves_in[phys_reducer], signal_in[phys_reducer]
    t_temp_phys, t_amu_phys = temps_arr[phys_reducer], amus_arr[phys_reducer]

    # 3. Stick Reduction (plot_window) - THIS IS THE FIX
    # This ensures redsticks contains ONLY what is visible on the plot
    plot_mask = (t_wav_phys >= np.min(plot_wind)) & (t_wav_phys <= np.max(plot_wind))
    redsticks = [t_wav_phys[plot_mask], t_sig_phys[plot_mask]]

    # Nothing survived the filtering: return an empty (zero) spectrum instead of failing.
    if len(t_wav_phys) == 0:
        empty = [np.array([np.min(plot_wind), np.max(plot_wind)]), np.zeros(2)]
        return empty, redsticks, []

    # 4. Binning and Convolution
    tbins = make_stickbins(t_wav_phys, t_sig_phys, t_temp_phys, t_amu_phys, windsize=wind_size)

    tempwavs, tempsigs = [], []
    for sticks, data_list, winds in zip(tbins['binned_waves'], zip(*tbins['binned_data']), tbins['bin_windows']):
        # Optimization: Skip bins entirely outside the plot window
        if winds[1] < np.min(plot_wind) or winds[0] > np.max(plot_wind):
            continue
            
        sigs, temps, amus = data_list
        twav = np.linspace(winds[0], winds[1], max(2, int(round(specres*(winds[1] - winds[0])/resolution_nm))))
        tsig = np.zeros_like(twav)
        
        for w, s, t, a in zip(sticks, sigs, temps, amus):
            tspec = Convol_Sticks(w, [winds[0], winds[1]], t, a, resolution_nm, 
                                 function=functiontype, Skew_consts=Skewness_consts, 
                                 vel_ion=ionvel, specres=specres, bidirect=bidir)
            tsig += s * tspec[1]

        tempwavs.extend(twav)
        tempsigs.extend(tsig)

    # 5. Final Spectrum Clipping
    out_w, out_s = np.array(tempwavs), np.array(tempsigs)
    mask = (out_w >= np.min(plot_wind)) & (out_w <= np.max(plot_wind))
    
    final_spectrum = [np.concatenate([[np.min(plot_wind)], out_w[mask], [np.max(plot_wind)]]),
                      np.concatenate([[0], out_s[mask], [0]])]

    return final_spectrum, redsticks, tbins['bin_windows']
