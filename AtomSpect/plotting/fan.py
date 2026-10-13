"""Zeeman fan plots (plotZfan)."""
from matplotlib import pyplot as plt
from matplotlib import ticker


def plotZfan(Inputdeck, flags='', savefig=False, figname='placeholder',markercount = 15,
             markersize_ = 6,markersize_H = 6,markersize_L = 6):
    '''
    Takes the input deck (which should have Bmag as a range of values), and calculates the typical zeeman fan.
    Both exact solution and lowfield are plotted on the same figure if dolow=True.

    Parameters
    ----------
    Inputdeck : dict
        DESCRIPTION.
    dolow : Boolean, optional
        DESCRIPTION. The default is True.
    savefig : Boolean, optional
        DESCRIPTION. The default is False.
    flags : String
        Can be any combination of G,E,L,H. The default is ''.
        G: Fan plot Ground
        E: Fan Plot of Excited
        L: Include Lowfield Approx
        H: Include Highfield Approx (hyperfine Paschen-Back for HFS).

    figname : String, optional
        DESCRIPTION. The default is 'placeholder'.

    Returns
    -------
    Returns exact and lowfield zeeman fan incase different plotting is required.
    [Gfan,Efan,Gfan_low,Efan_low]

    '''
    from .. import ZeemanFan  # late import: the driver still lives in the package root
    # Takes the inputdeck and a range of magnetic fields to calculate a typical zeeman fan.
    # Exact and lowfield are plotted on the same figure if that option is requested.

    Zfantest = ZeemanFan(Inputdeck, flags=flags)  # Calculate
    # Pull energy arrays from Zfancalculation

    Brange = Inputdeck['Bmag']
    if markercount == 0:
        markersize_ = 0
        markersize_H = 0
        markersize_L = 0
    if 'G' in flags:
        # figfg, axsfg = plt.subplots(figsize = (16,8))
        figs, axs = plt.subplots(figsize=(12, 10))
        plt.rcParams.update({'font.size': 18})

        axs.plot(Brange[0], Zfantest['G'][0][0], color='blue',  linewidth=.75, label="Intermediate", 
                 marker='D', markevery=markercount,markersize=markersize_, alpha=0.8)  # This makes a single point for the purpose of the legend.
        for things in (Zfantest['G']):
            axs.plot(Brange, things, color='blue',  linewidth=.75,
                     marker='D', markevery=markercount,markersize=markersize_, alpha=0.8)  # , label = "Ground")

        if 'L' in flags:
            for thingsL in (Zfantest['LowG']):
                plt.plot(Brange, thingsL, color='maroon', linestyle=':', linewidth=.75,
                         marker='x', markevery=markercount-2,markersize=markersize_L, alpha=0.9)    # , label = "Ground_Low")
            axs.plot(Brange[0], Zfantest['LowG'][0][0], color='maroon', linestyle=':', label="Ground Weak",
                     marker='x', markevery=markercount-2,markersize=markersize_L, alpha=0.8)  # This makes a single point for the purpose of the legend.
        if 'H' in flags:
            for thingsH in (Zfantest['HighG']):
                plt.plot(Brange, thingsH, color='green', linestyle='-.', linewidth=.75,
                         marker='o', markevery=markercount+2,markersize=markersize_H, alpha=0.8)   # , label = "Ground_Low")
            axs.plot(Brange[0], Zfantest['HighG'][0][0], color='green', linestyle='-.', label="Ground Strong",linewidth=.75,
                     marker='o', markevery=markercount+2,markersize=markersize_H, alpha=0.8)  # This makes a single point for the purpose of the legend.
        # plt.ylim(np.min(things),np.max(things))
        axs.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
        plt.tight_layout()
    if 'E' in flags:
        # figfe, axsfe = plt.subplots(figsize = (16,8))

        figs, axs = plt.subplots(figsize=(12, 10))
        plt.rcParams.update({'font.size': 18})

        axs.plot(Brange[0], Zfantest['E'][0][0], color='blue',  linewidth=.75, label="Intermediate", 
                 marker='D', markevery=markercount,markersize=markersize_, alpha=0.8)  # This makes a single point for the purpose of the legend.
        for things2 in Zfantest['E']:
            axs.plot(Brange, things2, color='blue', linewidth=.75,
                     marker='D', markevery=markercount,markersize=markersize_, alpha=0.8)  # , label = "Ground")
        if 'L' in flags:
            for thingsL in (Zfantest['LowE']):
                axs.plot(Brange, thingsL, color='maroon', linestyle=':', linewidth=.75,
                         marker='x', markevery=markercount-2,markersize=markersize_L, alpha=0.8) # , label = "Ground_Low")
            axs.plot(Brange[0], Zfantest['LowE'][0][0], color='maroon', linestyle=':', label="Excited Weak",linewidth=.75,
                     marker='x', markevery=markercount-2,markersize=markersize_L, alpha=0.8) # This makes a single point for the purpose of the legend.
        if 'H' in flags:
            for thingsH in (Zfantest['HighE']):
                axs.plot(Brange, thingsH, color='green', linestyle='-.',linewidth=.75,
                         marker='o', markevery=markercount+2,markersize=markersize_H, alpha=0.8)   # , label = "Ground_Low")
            axs.plot(Brange[0], Zfantest['HighE'][0][0], color='green', linestyle='-.', label="Excited Strong",linewidth=.75,
                     marker='o', markevery=markercount+2,markersize=markersize_H, alpha=0.8)   # This makes a single point for the purpose of the legend.
        # plt.ylim(np.min(things2),np.max(things2))
        axs.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
        plt.tight_layout()
    # Setting plot parameters
    plt.xlabel('Magnetic Field (T)', weight='semibold')
    plt.ylabel('Energy (cm^-1)', weight='semibold')
    plt.legend(loc='best')
    # plt.title(Inputdeck['plottitle'])
    plt.tight_layout()
    if savefig==True:
        plt.savefig(f'{figname}_Excited.png',dpi=2000) #Save the figure if requested
    return Zfantest
