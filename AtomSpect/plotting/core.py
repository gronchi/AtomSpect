"""Plot grids, legends and PlotFunction."""
import numpy as np
from matplotlib import pyplot as plt


def plotarray(shape, figure_size=(10, 8),fontsize=18):
    """Create a shared-axis (rows, columns) plot grid and return (figure, axes_array).

    shape is a two-element sequence. figure_size is in inches and fontsize
    in points. The axes array is always two-dimensional, including for 1x1.
    Rows share x by column; all axes share y. The function sets wavelength
    and intensity labels and changes Matplotlib's global font defaults."""

    fig = plt.figure(figsize=figure_size)  #Make the main figure
    plotmat = np.zeros(shape, dtype='O') #Make an empty np array of datatype 'Object"

    font = {
            'weight' : 'normal',
            'size'   : fontsize}
    plt.rc('font', **font)


    for rows in range(shape[0]):
        for cols in range(shape[1]):
            if rows == 0:
                if cols == 0:
                    plotmat[rows, cols] = plt.subplot2grid(shape, (rows, cols))
                else:
                    plotmat[rows, cols] = plt.subplot2grid(shape, (rows, cols), sharey=plotmat[0, 0])

            else:
                plotmat[rows, cols] = plt.subplot2grid(shape, (rows, cols), sharey=plotmat[0, 0], sharex=plotmat[0, cols])

    for rows in range(shape[0]-1):
        for cols in range(shape[1]):
            plotmat[rows, cols].tick_params('x', labelbottom=False)
    for rows in range(shape[0]):
        for cols in range(1, shape[1]):
            plotmat[rows, cols].tick_params('y', labelleft=False)

    fig.supxlabel('Wavelength (nm)', weight='semibold')
    fig.supylabel('Intensity (arb)', weight='semibold')
    plt.subplots_adjust(hspace=0, wspace=0)

    return fig, plotmat


def parse_plot_data(Plotobject):

    """Wrap a single spectrum dictionary or [x,y] array into a list of spectra."""
    if isinstance(Plotobject, list):
        if len(Plotobject) > 0 and isinstance(Plotobject[0], (dict, list, np.ndarray)):
            if isinstance(Plotobject[0], (list, np.ndarray)) and np.ndim(Plotobject[0]) == 2:
                return Plotobject
            if isinstance(Plotobject[0], dict):
                return Plotobject

    if isinstance(Plotobject, dict):
        return [Plotobject]

    if isinstance(Plotobject, (list, np.ndarray)) and np.ndim(Plotobject) == 2:
        return [Plotobject]

    return Plotobject

def get_target_axis(taxs, position):
    """Return one axes from an axes object or array, using position=[row,column]."""
    if not isinstance(taxs, np.ndarray):
        return taxs

    # Standardize position to [row, col].
    # If only one index is provided, treat it as the column index.
    row = position[0] if len(position) > 0 else 0
    col = position[1] if len(position) > 1 else 0

    # plotarray typically returns a 2D array even for a single row.
    if taxs.ndim == 1:
        return taxs[col]
    else:
        return taxs[row, col]

def consolidate_legend(tfig, plotpol=True, plotnondip=False, legcols=None, fontsize=18):
    """Collect unique labels from all figure axes, including twins, into one legend.

    legcols sets the column count; otherwise plotpol and plotnondip select
    the default layout. fontsize is in points. Return the supplied figure."""
    tfig.legends = [] # Clear existing legends
    all_handles = []
    all_labels = []

    # Iterate through all axes in the figure (twins are stored in tfig.axes as well)
    for ax in tfig.axes:
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)

    # Filter for unique labels to prevent 'Spectrometer' etc. from repeating
    unique_map = {}
    for handle, label in zip(all_handles, all_labels):
        if label not in unique_map:
            unique_map[label] = handle

    if not unique_map:
        return tfig

    labels = list(unique_map.keys())
    handles = list(unique_map.values())

    if legcols:
        cols = legcols
        rect_adj = [0, 0, 1, 0.95]
    elif plotnondip:
        cols = 3
        rect_adj = [0, 0, 1, 0.9]
    elif plotpol:
        cols = 5
        rect_adj = [0, 0, 1, 0.95]
    else:
        cols = 2
        rect_adj = [0, 0, 1, 0.95]

    plt.tight_layout(rect=rect_adj)
    tfig.legend(handles, labels, bbox_to_anchor=(0.5, 1),
                loc='upper center', ncol=cols, fontsize=fontsize)
    return tfig


def PlotFunction(Plotobject, plot_vars, plottitle='', plotwind=None, SpectrumPlot=[], Shape=[1, 1],
                 position=[0, 0], axsin=[], NormalizeSig=False, NormalizeScale=1, makefig=True,
                 fig_size=(10, 8), plotpol=True, plotlabel='', plotnondip=False, dotwin=False,
                 marker=None, marker_count=None, marker_size = None, fontsize=None, ScaleSpectrometer=None,
                 SpectrometerLabel='Spectrometer', SpectrometerMarker=None, SpectrometerColor='black',
                 SpectrometerLS='solid', legcols=None, show=True):

    """Plot one or more spectra and return an array of Matplotlib axes.
    
    Plotobject is a SpectrumResult/dictionary, [x,y], or a list of these.
    plot_vars is [color, linewidth, linestyle, optional label, optional marker],
    shared by all spectra or supplied once per spectrum. SpectrumPlot is
    optional measured [wavelength_nm, signal] data. Shape and fig_size set
    the new axes array and figure size in inches; makefig=False draws on
    axsin at position=[row,column]. show controls display of a new figure.
    
    NormalizeSig scales calculated spectra to NormalizeScale (scalar or a
    list per spectrum). ScaleSpectrometer sets measured peak height; zero
    is allowed. Raw calculated intensities otherwise use automatic y limits.
    plotwind sets wavelength bounds in nm, falling back to Input.plot_window.
    plottitle and plotlabel add text. marker, marker_count and marker_size
    control calculated curve markers; the Spectrometer options style measured
    data. plotpol adds pi/sigma sticks; dotwin uses a separate stick axis.
    plotnondip displays other_sticks, a legacy name for forbidden E1 channels.
    fontsize and legcols control text and the consolidated figure legend."""

    # --- 1. Standardize Input and Initialization ---
    Plotobject = parse_plot_data(Plotobject)

    font_size = fontsize if fontsize else 18
    markcount = marker_count if marker_count else 50
    spectrometer_scale = ScaleSpectrometer if ScaleSpectrometer is not None else 1
    marker_size_ = marker_size if marker_size is not None else 10

    if len(plot_vars) >= 3 and np.isscalar(plot_vars[1]):
        working_vars = [plot_vars]
    else:
        working_vars = plot_vars
    if len(working_vars) == 1 and len(Plotobject) > 1:
        working_vars = working_vars * len(Plotobject)
    if len(working_vars) != len(Plotobject):
        raise ValueError('Provide one shared plot style or one style per spectrum')

    resolved_markers = []
    for v in working_vars:
        if len(v) > 4:
            resolved_markers.append(v[4])
        elif marker:
            resolved_markers.append(marker)
        else:
            resolved_markers.append('o')

    # --- 2. Figure and Axis Setup ---
    if makefig:
        tfig, taxs = plotarray(Shape, figure_size=fig_size, fontsize=font_size)
    else:
        taxs = axsin

    target_axis = get_target_axis(taxs, position)
    tfig = target_axis.get_figure()

    # --- 3. Data Extraction and Spectrometer Input Logic ---
    tPlotObj = []
    main_dict = None

    for x in Plotobject:
        if isinstance(x, dict):
            # Fallback to wave_air/signal if SpecOut isn't the key
            tPlotObj.append(x.get('SpecOut', [x.get('wave_air'), x.get('signal')]))
            if main_dict is None: main_dict = x
        else:
            tPlotObj.append(x)

    final_spec_data = SpectrumPlot
    if len(final_spec_data) == 0 and main_dict is not None:
        if 'Input' in main_dict and 'SpectrumData' in main_dict['Input']:
            final_spec_data = main_dict['Input']['SpectrumData']

    if len(final_spec_data) > 0:
        spec_y = np.asarray(final_spec_data[1])
        from ..lineshapes import Normalize
        spec_norm = Normalize(spec_y, spectrometer_scale)

        target_axis.plot(final_spec_data[0], spec_norm,
                         color=SpectrometerColor, marker=SpectrometerMarker, markevery=10,
                         linewidth=1.5, label=SpectrometerLabel, linestyle=SpectrometerLS)

    # --- 4. Plotting Calculated Spectra ---
    for i, (tplot_data, p_vars) in enumerate(zip(tPlotObj, working_vars)):
        y_raw = np.asarray(tplot_data[1])
        if NormalizeSig:
            scale_val = NormalizeScale[i] if isinstance(NormalizeScale, list) else NormalizeScale
            from ..lineshapes import Normalize
            y_data = Normalize(y_raw, scale_val)
        else:
            y_data = y_raw

        target_axis.plot(tplot_data[0], y_data, color=p_vars[0], linestyle=p_vars[2],
                         linewidth=p_vars[1], label=p_vars[3] if len(p_vars) > 3 else None, alpha=0.8,
                         marker=resolved_markers[i], markevery=markcount, markersize=marker_size_)

    # --- 5. Styling and Limits ---
    plottingwind = plotwind
    if main_dict and 'Input' in main_dict and plotwind is None:
        plottingwind = main_dict['Input'].get('plot_window', None)

    if plottingwind is not None and len(plottingwind) > 0:
        target_axis.set_xlim(np.min(plottingwind), np.max(plottingwind))
    if NormalizeSig:
        scales = np.asarray(NormalizeScale)
        top = max(1., float(np.max(scales)), float(spectrometer_scale))
        target_axis.set_ylim(-0.1*top, 1.1*top)

    if plottitle:
        target_axis.set_title(plottitle)
    if plotlabel:
        target_axis.text(0.05, 0.95, f'{plotlabel}', verticalalignment='top',
                         transform=target_axis.transAxes, fontsize=font_size-2)

    # --- 6. Polarization Stems ---
    if plotpol and main_dict and 'pi_sticks' in main_dict:
        stem_ax = target_axis.twinx() if dotwin else target_axis

        if dotwin and 'reduced_sticks' in main_dict and len(main_dict['reduced_sticks'][1]):
            peak = np.max(main_dict['reduced_sticks'][1])
            if peak > 0:
                stem_ax.set_ylim(-0.1*peak, 1.1*peak)

        stem_ax.stem(main_dict['pi_sticks'][0], main_dict['pi_sticks'][1],
                     linefmt='C0', basefmt=" ", label=r'$\pi$')
        stem_ax.stem(main_dict['sigmaminus_sticks'][0], main_dict['sigmaminus_sticks'][1],
                     linefmt='C2', markerfmt='_', basefmt=" ", label=r'$\sigma^-$')
        stem_ax.stem(main_dict['sigmaplus_sticks'][0], main_dict['sigmaplus_sticks'][1],
                     linefmt='C3', markerfmt='x', basefmt=" ", label=r'$\sigma^+$')

        if plotnondip and 'other_sticks' in main_dict:
            m, _, _ = stem_ax.stem(main_dict['other_sticks'][0], main_dict['other_sticks'][1],
                                   linefmt='C4', markerfmt='*', basefmt=" ", label='other components')
            plt.setp(m, markersize=10, markerfacecolor='none')

    # --- 7. Legend and Final Layout ---
    consolidate_legend(tfig, plotpol=plotpol, plotnondip=plotnondip, legcols=legcols, fontsize=font_size)

    if makefig:
        plt.subplots_adjust(hspace=0, wspace=0)
        if show:
            plt.show()

    return taxs
