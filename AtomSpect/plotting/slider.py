"""Interactive slider GUI (MakeSlider)."""
import sys

import numpy as np
import tkinter as tk
from tkinter import simpledialog
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from ..lineshapes import Normalize


# This section defines the plot_slider which creates interactve plots where parameters can be changed. This is a proof of concept, but is already looking very promising. Further UI work is needed.


def _script_should_block():
    """True when the process would exit right after the figure is shown (a plain script)."""
    if hasattr(sys, 'ps1') or sys.flags.interactive:  # python -i or the plain interactive prompt
        return False
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # Spyder, Jupyter and IPython run their own event loop
            return False
    except ImportError:
        pass
    return True


def Zslider(B, thisTemp, thisRes, Input):
    """Recompute a spectrum after changing B (T), temperature (eV), and width (nm).

    Mutate Input in place, updating Temp in kelvin and any existing TempeV,
    then return AtomSpect_Main's SpectrumResult."""
    from .. import AtomSpect_Main  # late import: the driver still lives in the package root
    Input['Bmag'] = B
    Input['specstep'] = thisRes
    Input['Temp'] = thisTemp*11602
    if 'TempeV' in Input:
        Input['TempeV'] = thisTemp

    return AtomSpect_Main(Input)


def Zslider_Pol(B, thisTemp, thisRes, Bangle, PolAngle, Input):
    """Recompute after changing the field, temperature, width, and geometry.

    B is in T, thisTemp in eV, thisRes in nm, and Bangle/PolAngle in degrees.
    PolAngle=None preserves the existing filter setting; zero enables the
    analyzer at zero degrees. Mutate Input and return a SpectrumResult."""
    from .. import AtomSpect_Main  # late import: the driver still lives in the package root
    if PolAngle is not None:  # None: leave any filter angle in the deck untouched. 0 is a valid angle.
        Input['Pol_angle'] = PolAngle

    Input['Bmag'] = B
    Input['specstep'] = thisRes
    Input['Temp'] = thisTemp*11602
    if 'TempeV' in Input:
        Input['TempeV'] = thisTemp
    Input['b_angle'] = Bangle

    return AtomSpect_Main(Input)


def enable_manual_input(fig, slider):
    """Allow manual entry of slider values by double-clicking on the value text."""
    valtext = slider.valtext  # this is a matplotlib.text.Text object

    def on_double_click(event):
        if event.dblclick and valtext.contains(event)[0]:
            # popup Tk dialog for numeric input
            root = tk.Tk()
            root.withdraw()  # hide main window
            try:
                new_val = simpledialog.askfloat("Input", f"Enter {slider.label.get_text()} value:",
                                                initialvalue=slider.val)
                if new_val is not None:
                    slider.set_val(new_val)  # updates slider + fires callbacks
            finally:
                root.destroy()

    fig.canvas.mpl_connect("button_press_event", on_double_click)



def MakeSlider(Inputdict, Spectra, plotwind=None, do_sticks=True, do_Polplot=True, banglevary=False
               , polvary=False, do_other = False, Tmax = 30, Bmax = 5, block=None):
    """Display an interactive spectrum with field, temperature, and width sliders.
    
    Inputdict is mutated during updates. Spectra is measured [wavelength_nm,
    intensity] data; plotwind optionally supplies two wavelength bounds.
    do_sticks adds stems, do_Polplot splits them by polarization (and enables
    stems), and do_other includes forbidden E1 channels. banglevary and
    polvary add observation and analyzer angles in degrees. Tmax is the
    maximum temperature in eV and Bmax the maximum field in T.
    
    The GUI starts at hard-coded B=2.5 T, T=0.05 eV, and width=0.002 nm.
    It needs an interactive Matplotlib backend; manual-value dialogs use Tk.
    block=None blocks until the window is closed when run as a plain script and
    returns at once in IPython/Spyder; block=False always returns at once, so several
    windows can be opened and then shown together with plt.show(block=True).
    Returns the figure; its `sliders` attribute keeps the slider widgets alive."""
    from .. import _approximation_enabled  # late import: the driver still lives in the package root
    
    # Initial params
    init_B = 2.5  # Initial B field (T)
    init_Temp = 0.05  # Initial temp (300K)
    init_Res = 0.002  # Initial pixel size (nm)
    init_bang = 90  # Initially LoS is perpendicular to B
    init_polang = 90  # Initially block all circular light

    #Make the blank figure and set parameters
    figs, axs = plt.subplots(figsize=(16, 10))
    plt.rcParams.update({'font.size': 14})
    figs.subplots_adjust(left=0.2, bottom=0.2, right=0.8)

    #Plot the spectrometer data
    axs.plot(Spectra[0], Spectra[1], color="black", label="Spectrometer")

    #Check for polvary and/or banglevary flags and initilize with the respective cases.
    if polvary and not banglevary:
        #Assume bangle=90 by default (perpendicular to LoS)
        Mycalc0 = Zslider_Pol(init_B, init_Temp, init_Res, init_bang, init_polang, Inputdict)
    elif banglevary and not polvary:
        Mycalc0 = Zslider_Pol(init_B, init_Temp, init_Res, init_bang, None, Inputdict)
    elif polvary and banglevary:
        Mycalc0 = Zslider_Pol(init_B, init_Temp, init_Res, init_bang, init_polang, Inputdict)

    else:
        #Have no 
        Mycalc0 = Zslider(init_B, init_Temp, init_Res, Inputdict)
        Bang_Slider = None
        Polang_Slider = None
    if banglevary:
        axampBang = figs.add_axes([0.85, 0.25, 0.0225, 0.63])
        Bang_Slider = Slider(axampBang, "B_Angle to LoS", 0, 90, valstep=0.5, valinit=init_bang,
                             orientation="vertical")
    if polvary:
        axamppolang = figs.add_axes([0.93, 0.25, 0.0225, 0.63])
        Polang_Slider = Slider(axamppolang, "Pol Filter",  0, 90, valstep=0.5, valinit=init_polang,
                               orientation="vertical")
    # artists
    line, = axs.plot(Mycalc0['SpecOut'][0],
                     Normalize(Mycalc0['SpecOut'][1]), color="tab:brown",
                     lw=3.5, label="Intermediate", linestyle=':',marker='o', markevery=10,markersize=7, alpha=0.9)

    # line = axs.scatter(Mycalc0['SpecOut'][0],
    #                  Normalize(Mycalc0['SpecOut'][1]),color="tab:brown" ,
    #                  lw=1, label="Exact",marker = '.' , alpha = 0.75)
    numcols = 5
    line2 = line3 = None
    if _approximation_enabled(Inputdict.get('DoLowSig', False)):
        line2, = axs.plot(Mycalc0['SpecOutLow'][0],
                         Normalize(Mycalc0['SpecOutLow'][1]),
                         lw=2, alpha=.9, label="Lowfield", linestyle=(5, (5, 2, 1, 2)),
                         marker='D', markevery=25, markersize=5)
        numcols = 4
    if _approximation_enabled(Inputdict.get('DoHighSig', False)):
        line3, = axs.plot(Mycalc0['SpecOutHigh'][0],
                         Normalize(Mycalc0['SpecOutHigh'][1]),
                         lw=2, alpha=.9, label="Highfield", linestyle=(5, (5, 2, 1, 2)),
                         marker='x', markevery=25, markersize=5)
        numcols = 3 if line2 is not None else 4

    if plotwind is None:
        try:
            axs.set_xlim(Inputdict['plot_window'][0], Inputdict['plot_window'][1])
        except:
            pass
    else:
        try:
            axs.set_xlim(plotwind[0], plotwind[1])
        except:
            print('Something went wrong, setting plotwind range as min/max of spectra')
            axs.set_xlim(np.min(Mycalc0['SpecOut'][0]), np.max(Mycalc0['SpecOut'][0]))

    # axs.legend(loc="best")
    vstem = None
    vstem = None
    # vl_sp = vl_pi = vl_sm = vl_o = None
    if do_Polplot:
        do_sticks = True
    if do_sticks and not do_Polplot:
        # vl = axs.vlines(Mycalc0['reduced_sticks'][0], 0, Mycalc0['reduced_sticks'][1])
        vmarks, vstem, vbase = axs.stem(Mycalc0['reduced_sticks'][0], Mycalc0['reduced_sticks'][1], linefmt='C0', basefmt=" ")

    elif do_sticks and do_Polplot:
        # vl_sp = axs.vlines(Mycalc0['sigmaplus_sticks'][0], np.zeros_like(Mycalc0['sigmaplus_sticks'][1]), Mycalc0['sigmaplus_sticks'][1], lw=2,color="tab:purple", label="σ+")
        # vl_pi = axs.vlines(Mycalc0['pi_sticks'][0], np.zeros_like(Mycalc0['pi_sticks'][1]), Mycalc0['pi_sticks'][1], color="tab:green",lw=2, label="π")
        # vl_sm = axs.vlines(Mycalc0['sigmaminus_sticks'][0], np.zeros_like(Mycalc0['sigmaminus_sticks'][1]), Mycalc0['sigmaminus_sticks'][1],lw=2, color="tab:red", label="σ−")
        # vl_o = axs.vlines(Mycalc0['other_sticks'][0], np.zeros_like(Mycalc0['other_sticks'][1]), Mycalc0['other_sticks'][1],lw=2, color="tab:cyan", label="??")

        vmarksp, vstemsp, vbasesp = axs.stem(Mycalc0['pi_sticks'][0], Mycalc0['pi_sticks'][1], linefmt='C0', markerfmt='o', basefmt=" ", label=r'$\pi$ component')
        vmarkpi, vstempi, vbasepi = axs.stem(Mycalc0['sigmaminus_sticks'][0], Mycalc0['sigmaminus_sticks'][1], linefmt='C2', markerfmt='_', basefmt=" ", label=r'$\sigma^-$ component')
        vmarksm, vstemsm, vbasesm = axs.stem(Mycalc0['sigmaplus_sticks'][0], Mycalc0['sigmaplus_sticks'][1], linefmt='C3', markerfmt='+', basefmt=" ",  label=r'$\sigma^+$ component')

        propspi = vmarkpi.properties()
        propssp = vmarksp.properties()
        propssm = vmarksm.properties()
        
        if do_other:
            vmarko, vstemo, vbaseo = axs.stem(Mycalc0['other_sticks'][0], Mycalc0['other_sticks'][1], linefmt='C4', markerfmt='D', basefmt=" ",  label='other component')
            propso = vmarko.properties()

        # axs.legend(loc="best")
    # sliders

    # figs.legend(bbox_to_anchor=(.5, 0.15),loc='outside center')
    figs.legend(bbox_to_anchor=(.5, .125), loc='upper center', ncol=numcols)  # Excluding highfield and lowfield.

    axampB = figs.add_axes([0.025, 0.25, 0.0225, 0.63])
    B_Slider = Slider(axampB, "B (T)", 0, 5, valinit=init_B,
                      orientation="vertical", valstep=0.01)

    axampT = figs.add_axes([0.075, 0.25, 0.0225, 0.63])
    Te_Slider = Slider(axampT, "T (eV)", 0.01, 30, valstep = 0.01, valinit=init_Temp,
                       orientation="vertical")

    axampR = figs.add_axes([0.125, 0.25, 0.0225, 0.63])
    Res_Slider = Slider(axampR, r"$W_i$ (nm)", 0.002, 0.1, valstep=0.001, valinit=init_Res,
                        orientation="vertical")

    # reset button
    # resetax = figs.add_axes([0.8, 0.025, 0.1, 0.04])
    # button = Button(resetax, "Reset", hovercolor="0.975")
    plt.ion()

    artists = [line]
    if line2:
        artists.append(line2)

    if vstem:
        artists.append(vstem)

    def update(val):
        if polvary and not banglevary:
            Mycalc = Zslider_Pol(B_Slider.val, Te_Slider.val, Res_Slider.val, Inputdict['b_angle'], Polang_Slider.val, Inputdict)
        elif banglevary and not polvary:
            Mycalc = Zslider_Pol(B_Slider.val, Te_Slider.val, Res_Slider.val, Bang_Slider.val, None, Inputdict)
        elif polvary and banglevary:
            Mycalc = Zslider_Pol(B_Slider.val, Te_Slider.val, Res_Slider.val, Bang_Slider.val, Polang_Slider.val, Inputdict)
        else:
            Mycalc = Zslider(B_Slider.val, Te_Slider.val, Res_Slider.val, Inputdict)
        # update lines
        line.set_xdata(Mycalc['SpecOut'][0])
        line.set_ydata(Normalize(Mycalc['SpecOut'][1]))
        if line2:
            line2.set_xdata(Mycalc['SpecOutLow'][0])
            line2.set_ydata(Normalize(Mycalc['SpecOutLow'][1]))
        if line3:
            line3.set_xdata(Mycalc['SpecOutHigh'][0])
            line3.set_ydata(Normalize(Mycalc['SpecOutHigh'][1]))

        if do_sticks and not do_Polplot:
            x_new = Mycalc['reduced_sticks'][0]
            y_new = Mycalc['reduced_sticks'][1]

            vstem.set_paths([np.array([[xx, 0],
                                       [xx, yy]]) for (xx, yy) in zip(x_new, y_new)])

            vmarks.set_xdata(x_new)
            vmarks.set_ydata(y_new)

        elif do_sticks and do_Polplot:
            if do_other:
                for (vlstem, vlmarks, props, key, colorfmt, markers) in [
                    (vstempi, vmarkpi, propspi, "pi_sticks", "C0", 'o'),
                    (vstemsp, vmarksp, propssp, "sigmaplus_sticks", "C3", '+'),
                    (vstemsm, vmarksm, propssm, "sigmaminus_sticks", "C2", '_'),
                    (vstemo, vmarko, propso, "other_sticks", "C4", 'D'),
                ]:
                    
                    x_new = Mycalc[key][0]
                    y_new = Mycalc[key][1]
    
                    vlstem.set_paths([np.array([[xx, 0],
                                                [xx, yy]]) for (xx, yy) in zip(x_new, y_new)])
    
                    vlstem.set_color(colorfmt)
                    vlmarks.set_xdata(x_new)
                    vlmarks.set_ydata(y_new)
                    vlmarks.set_marker(markers)
                    vlmarks.set_color(colorfmt)
                    # vlmarks.update(props)
            else:
                for (vlstem, vlmarks, props, key, colorfmt, markers) in [
                    (vstempi, vmarkpi, propspi, "pi_sticks", "C0", 'o'),
                    (vstemsp, vmarksp, propssp, "sigmaplus_sticks", "C3", '+'),
                    (vstemsm, vmarksm, propssm, "sigmaminus_sticks", "C2", '_'),
                  #  (vstemo, vmarko, propso, "other_sticks", "C4", 'D'),
                ]:
                    
                    x_new = Mycalc[key][0]
                    y_new = Mycalc[key][1]
    
                    vlstem.set_paths([np.array([[xx, 0],
                                                [xx, yy]]) for (xx, yy) in zip(x_new, y_new)])
    
                    vlstem.set_color(colorfmt)
                    vlmarks.set_xdata(x_new)
                    vlmarks.set_ydata(y_new)
                    vlmarks.set_marker(markers)
                    vlmarks.set_color(colorfmt)

        figs.canvas.blit(axs.bbox)
    axs.set_ylim(-.1, 1.1)

    # slider connections
    B_Slider.on_changed(update)
    Te_Slider.on_changed(update)
    Res_Slider.on_changed(update)
    if banglevary:
        Bang_Slider.on_changed(update)
    if polvary:
        Polang_Slider.on_changed(update)

    #No clue why the reset button doesn't work anymore. =\
    # def reset(event): 
    #     B_Slider.reset()
    #     Te_Slider.reset()
    #     Res_Slider.reset()
        # if polvary:
        #     Bang_Slider.reset()
        #     Polang_Slider.reset()
    # button.on_clicked(reset)
    enable_manual_input(figs, B_Slider)
    enable_manual_input(figs, Te_Slider)
    enable_manual_input(figs, Res_Slider)
    plt.suptitle(Inputdict.get('plottitle', ''))

    axs.set_ylabel("Normalized Intensity (Arb units)", weight='semibold')
    axs.set_xlabel('Wavelength (nm)', weight='semibold')

    # With interactive mode on, show() returns at once and a plain script would exit, closing the
    # window. Keep the sliders alive in a script; a console that runs its own event loop needs no block.
    # Matplotlib holds widget callbacks weakly: keep the sliders referenced from the figure so they
    # stay responsive after this function returns.
    figs.sliders = {name: widget for name, widget in (('B', B_Slider), ('T', Te_Slider), ('Wi', Res_Slider),
                    ('angle', locals().get('Bang_Slider')), ('polarizer', locals().get('Polang_Slider')))
                    if widget is not None}
    plt.show(block=_script_should_block() if block is None else block)
    return figs
