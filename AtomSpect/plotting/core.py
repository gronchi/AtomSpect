"""Plot grids, legends and PlotFunction."""
import numpy as np
from matplotlib import pyplot as plt




def plotarray(shape, figure_size=(10, 8),fontsize=18):
    '''
    This is a general function which makes a blank array of plots with shape (rows,col)
    and returns the fig object and array of plots. This is used with PlotFunction to feed plot data
    towards a streamlined plotting experience.
    A generalized version of these functions would be ideal, but the scope here is for use
    in this code.

    Parameters
    ----------
    shape : tuple
        Dimensions of the plot (Rows, Colums)
    figure_size : tuple (Width, Height)
        Plot size in inches for the resulting figure. The default is (10, 8).
    fontsize : Int
        Size for the font in the figure. The default is 18.

    Returns
    -------
    fig : natplotlib figure object
        callable and assignable figure object.
    plotmat : array
        Array of plots where individual axes can be indexed for plotting on.

    '''
    
    fig = plt.figure(figsize=figure_size)  #Make the main figure
    plotmat = np.zeros(shape, dtype='O') #Make an empty np array of datatype 'Object"
    
    #Set the font size
    font = {
            'weight' : 'normal',
            'size'   : fontsize}
    plt.rc('font', **font)


    #Go through the array and define the plots in the subfigure.
    #Iterate through the range and set that the columns and rows have shared axes.
    
    for rows in range(shape[0]):
        for cols in range(shape[1]):
            #First row and column have unique instructions since they define what is shared.
            if rows == 0:
                if cols == 0:
                    plotmat[rows, cols] = plt.subplot2grid(shape, (rows, cols))
                else:
                    plotmat[rows, cols] = plt.subplot2grid(shape, (rows, cols), sharey=plotmat[0, 0])

            else:
                plotmat[rows, cols] = plt.subplot2grid(shape, (rows, cols), sharey=plotmat[0, 0], sharex=plotmat[0, cols])
   
    # Get rid fo the labels and ticks that are covered up for x axis.
    for rows in range(shape[0]-1):
        for cols in range(shape[1]):
            plotmat[rows, cols].tick_params('x', labelbottom=False)
    # Get rid of y axis labels, but only for the later columns.
    for rows in range(shape[0]):
        for cols in range(1, shape[1]):
            plotmat[rows, cols].tick_params('y', labelleft=False)
            
    #Set the x and y labels for standard use case of AtomSpect.
    fig.supxlabel('Wavelength (nm)', weight='semibold')
    fig.supylabel('Intensity (arb)', weight='semibold')
    #Adjust the spacing to be closer together
    plt.subplots_adjust(hspace=0, wspace=0)

    #Return a figure and plotmatrix objects that can be called for other functions.
    return fig, plotmat



def parse_plot_data(Plotobject):
    #This is a helper function which parses possible input structures for use in the PlotFunction.
    
    # Case 1: Already a list of objects
    # Check if first element is a dictionary or a nested list
    if isinstance(Plotobject, list):
        if len(Plotobject) > 0 and isinstance(Plotobject[0], (dict, list, np.ndarray)):
            # If it's [ [wavs, sigs] ], we are good.
            if isinstance(Plotobject[0], (list, np.ndarray)) and np.ndim(Plotobject[0]) == 2:
                return Plotobject
            # If it's [ {dict} ], we are good.
            if isinstance(Plotobject[0], dict):
                return Plotobject

    # Case 2: Single Dictionary (Zeeman output)
    if isinstance(Plotobject, dict):
        return [Plotobject]

    # Case 3: Single 2D Array [waves, signals]
    if isinstance(Plotobject, (list, np.ndarray)) and np.ndim(Plotobject) == 2:
        return [Plotobject]

    return Plotobject

def get_target_axis(taxs, position):
    '''
    Safely extracts a single target axis regardless of taxs array dimensions.
    This is for use with PlotFunction with multiple calls for overlapping plotting of different objects etc.
    '''
    # If taxs is already a single axis object (e.g. from a manual slice), return it
    if not isinstance(taxs, np.ndarray):
        return taxs
    
    # Standardize position to [row, col]. 
    # If only one index is provided, treat it as the column index.
    row = position[0] if len(position) > 0 else 0
    col = position[1] if len(position) > 1 else 0
    
    # Handle indexing based on the array dimensions returned by plotarray
    # plotarray typically returns a 2D array even for a single row.
    if taxs.ndim == 1:
        return taxs[col]
    else:
        return taxs[row, col]

def consolidate_legend(tfig, plotpol=True, plotnondip=False, legcols=None, fontsize=18):
    '''
    Gathers all handles and labels from EVERY axis in the figure (including twins)
    and generates a single, unique legend at the top.
    '''
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

    # Determine layout adjustment and column count based input flags
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
         
    '''
    Author note: This function was refactored by Gemini AI from a previous iteration. The changes add a pre-processing step
    which helps generalize usage and make it easier to parse what the plot function is doing. This code has been reviewed prior to publication
    
    Parameters
    ----------
    Plotobject : What is being plotted.
        This can be [x,y] data or a Zeeman Object dictionary output. It used to handle lists of possible inputs
        But I would suggest using the ability to define an existing axes or plotarray instead.
    plot_vars : list [color,linethickness, linestyle, label (optional), markerstyle (optional)]
        List of definable parameters for a given plot such as color, linestyle, label, line thickness, and marker style
    plottitle : str, optional
        String for the plot title
    plotwind : list or array, optional
        Sets the limits for upper and lower bound of the plot window. If not defined, will try to see if there is an 
        PlotObject['input'] which is the input dictionary for the main code which is contained in the output of the main function.
        This input can include the plot_window keyword.
        Otherwise, defaults to either the limits of the spectrometer data or the full range of the spectra convolution
    SpectrumPlot : Array, optional
        This is an array of spectrometer data [wavelength(nm),intensity]. Does not need to be normalized, can be using ScaleSpectrometer flag.
        This gets plotted in black, but can be changed.
    Shape : list or tuple, optional
        Shape of the plot array that is to be created if makefig=True is set. Ignored otherwise.
    position : index (list or int), optional [row,column]
        Position in the plot array for the given object to be plotted. Useful for making one plot then adding many overplots
    axsin : array of plot objects, optional
        This is the plotmatrix as output from PlotArray or returned as an otput of PlotFunction
        If this is fed into PlotFunction here, this is where the object will be plotted. position information is thus required.
    NormalizeSig : BOOL, optional
        Whether a pass is done on the data to normalize it (making the maximum height as 1).
        Note that the sums of the signal strenghts are already normalized to equal 1. This just uniformly scales the plot outputs.
        The default is False.
    NormalizeScale : INT, optional
        A given curve can be scaled so that the maximum height is different than 1.
        Useful for relative intensities of multiple species to match the observed spectra. The default is 1.
    makefig : BOOL, optional
        Whether a figure object should be created. Need Shape unless 1x1 plot is desired. The default is True.
        Set to FALSE 
    show : BOOL, optional
        Display a new figure immediately. Default True. Set False when adding
        overlays or saving the completed figure before displaying it.
    fig_size : tuple (Width, Height)
        Plot size in inches for the resulting figure. The default is (10, 8).
    plotpol : BOOL, optional
        Whether the polarization components are plotted as stems in addition to the spectra. The default is True.
    plotlabel : STR, optional
        Label for the plot. The default is ''.
    plotnondip : BOOL, optional
        Whether other_sticks should be plotted as purple stars. Default False.
        This is a legacy option name, not evidence for non-dipole emission.
    dotwin : BOOL, optional
        Whether a twin axis is plotted with a separate scale for the sticks than the main plot. The default is False.
    marker : INT, optional
        Index of markerlist. These are ['o','8','D','s','P','^','v','X']. Can also be 
    marker_count : INT, optional
        How many points between each marker on the plot. Allows for easier reading of the plot. The default is None.
    fontsize : TYPE, optional
       Size of the font
    ScaleSpectrometer : float, optional
       If provided, scales the spectrometer height so that the maximum value is that as provided. Default would be 1.
    SpectrometerLabel : STR, optional
        Optional label for spectrometer data. The default is 'Spectrometer'.
    SpectrometerMarker : STR, optional
        Ability to change the spectrometer data marker. The default is 'd'.
    SpectrometerColor : STR, optional
        Color for spectrometer data. The default is 'black'.
    SpectrometerLS : STR, optional
        Spectrometer Linestyle. The default is 'solid'.
    legcols : INT, optional
        How many columns the legend will have.. The default is None.
    
    Returns
    -------
    The plot array of axes objects that can be called and indexed for adding plot objects.
    After creation, this array can be fed in as axsin multiple times to keep adding data.
    From an index axes, it is possible to get additional information with axes operators.
   
   
    '''

    # --- 1. Standardize Input and Initialization --- 
    # Use the intput parser to ensure we always have a list of objects to loop over
    Plotobject = parse_plot_data(Plotobject)
    
    font_size = fontsize if fontsize else 18 
    markcount = marker_count if marker_count else 50
    spectrometer_scale = ScaleSpectrometer if ScaleSpectrometer else 1 
    marker_size_ = marker_size if marker_size else 10  
    
    if len(np.shape(plot_vars)) == 1: 
        working_vars = [plot_vars] 
    else: 
        working_vars = plot_vars 

    #Setting up the marker styles if input, otherwise default to circles.
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
     
    # Use the global helper to extract the correct axis from [1,m], [n,1], or [n,m] arrays
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

    # Resolve Spectrometer Data
    final_spec_data = SpectrumPlot 
    if len(final_spec_data) == 0 and main_dict is not None: 
        if 'Input' in main_dict and 'SpectrumData' in main_dict['Input']: 
            final_spec_data = main_dict['Input']['SpectrumData'] 

    if len(final_spec_data) > 0: 
        # Vectorized Normalize to prevent iteration errors
        spec_y = np.asarray(final_spec_data[1])
        spec_norm = (spec_y / np.max(spec_y)) * spectrometer_scale if np.max(spec_y) != 0 else spec_y
        
        target_axis.plot(final_spec_data[0], spec_norm,  
                         color=SpectrometerColor, marker=SpectrometerMarker, markevery=10,  
                         linewidth=1.5, label=SpectrometerLabel, linestyle=SpectrometerLS) 

    # --- 4. Plotting Calculated Spectra --- 
    for i, (tplot_data, p_vars) in enumerate(zip(tPlotObj, working_vars)): 
        y_raw = np.asarray(tplot_data[1])
        if NormalizeSig: 
            scale_val = NormalizeScale[i] if isinstance(NormalizeScale, list) else NormalizeScale 
            # Vectorized Normalization: Scaling * (Data / Max)
            y_data = (y_raw / np.max(y_raw)) * scale_val if np.max(y_raw) != 0 else y_raw
        else: 
            y_data = y_raw 
             
        target_axis.plot(tplot_data[0], y_data, color=p_vars[0], linestyle=p_vars[2],  
                         linewidth=p_vars[1], label=p_vars[3], alpha=0.8, 
                         marker=resolved_markers[i], markevery=markcount, markersize=marker_size_) 

    # --- 5. Styling and Limits --- 
    plottingwind = plotwind 
    if main_dict and 'Input' in main_dict and not plotwind: 
        plottingwind = main_dict['Input'].get('plot_window', None) 

    if plottingwind: 
        target_axis.set_xlim(np.min(plottingwind), np.max(plottingwind)) 
    target_axis.set_ylim(-0.1, 1.1) 
     
    if plottitle:  
        target_axis.set_title(plottitle) 
    if plotlabel: 
        target_axis.text(0.05, 0.95, f'{plotlabel}', verticalalignment='top',  
                         transform=target_axis.transAxes, fontsize=font_size-2) 

    # --- 6. Polarization Stems --- 
    if plotpol and main_dict and 'pi_sticks' in main_dict: 
        stem_ax = target_axis.twinx() if dotwin else target_axis 
         
        if dotwin and 'reduced_sticks' in main_dict: 
            stem_ax.set_ylim(-0.1 * np.max(main_dict['reduced_sticks'][1]),  
                             np.max(main_dict['reduced_sticks'][1])) 
         
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
