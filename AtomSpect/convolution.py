"""Stick binning and convolution into spectra."""
import numpy as np

from .lineshapes import Gaussian, GaussianInstrum, Lorrentzian, Voigt


def MultiSpec(appendlist, scalelist):
    """
    Combines several Zeeman outputs into single arrays for wavelength, signal,
    temperature, and atomic mass.
    """
    waves = np.array([])
    signals = np.array([])
    temps = np.array([])
    amus = np.array([])
    
    for scaler, obj in zip(scalelist, appendlist):
        waves = np.append(waves, obj['wave_air'])
        signals = np.append(signals, scaler * np.array(obj['signal']))
        
        # Capture metadata for each component to allow different T/amu during convolution
        # Defaults to 300K and 1 amu if not found 
        t_val = obj['Input'].get('Temp', 300)
        a_val = obj['Input'].get('amu', 1)
        
        temps = np.append(temps, np.full_like(obj['wave_air'], t_val))
        amus = np.append(amus, np.full_like(obj['wave_air'], a_val))
        
    return [waves, signals, temps, amus]

def Convol_Sticks(x0, xwind, Temp_in, amu_in, stepsize_nm, function="", Skew_consts=[1, 1], vel_ion=0, specres=10, bidirect=False):
    '''
    Assume that the inputs are all in units of nm.
    The difference in this function is that I'm assuming that this subroutine will be called after the sticks have already been limited to a larger window.
    This means that the full lineshape for a given convolution may not always be plotted, but then ensures that any given larger convolution will always end up the same length of array.
    '''
    # hstep = stepsize_nm # step size in nm!! (Want 10x the steps per pixe for better shape?)
    hstep = (1/specres)*stepsize_nm  # step size in nm!! (Want 1/specres times the steps per pixe for better shape)

    xmin = np.min(xwind)
    xmax = np.max(xwind)

    numwavesteps = int(round((xmax-xmin)/hstep))
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
    wavered = np.asarray(sticks_in)
    if wavered.size == 0:
        return {'binned_waves': [], 'binned_data': [[] for _ in data_arrays], 'bin_windows': []}
    srt = np.sort(wavered)
    breaks = np.where(np.diff(srt) > windsize)[0]
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

def Convol_Spect(waves_in, signal_in, spec_wind, plot_wind, atomic_weight_amu, resolution_nm, 
                 Temperature_in=300, functiontype="", wind_size=.1, 
                 Skewness_consts=[1, 1], ionvel=0, specres=10, bidir=False):

    # 1. Resolve inputs into arrays
    amus_arr = np.full_like(waves_in, atomic_weight_amu) if not hasattr(atomic_weight_amu, "__len__") else atomic_weight_amu
    temps_arr = np.full_like(waves_in, Temperature_in) if not hasattr(Temperature_in, "__len__") else Temperature_in

    # 2. Physics Reduction (spec_window) - used for the actual math
    phys_reducer = (np.array(signal_in) > (0.05 * np.max(signal_in)/len(signal_in))) & \
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
        twav = np.linspace(winds[0], winds[1], int(round(specres*(winds[1] - winds[0])/resolution_nm)))
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
