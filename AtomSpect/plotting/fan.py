"""Zeeman fan plots (plotZfan)."""
from matplotlib import pyplot as plt
from matplotlib import ticker


def plotZfan(Inputdeck, flags='', savefig=False, figname='placeholder',markercount = 15,
             markersize_ = 6,markersize_H = 6,markersize_L = 6):
    """Plot the energy fans selected by G/E/L/H flags and return ZeemanFan's dictionary.

    Inputdeck['Bmag'] contains field samples in T. Energies are in cm^-1.
    markercount controls marker spacing; zero hides markers. The marker-size
    arguments set exact, high- and weak-field marker sizes. savefig writes
    the current figure to figname + '_Excited.png' (legacy naming); if both
    G and E are selected, separate figures are opened and only E is saved."""
    from .. import ZeemanFan  # late import: the driver still lives in the package root

    Zfantest = ZeemanFan(Inputdeck, flags=flags)  # Calculate

    Brange = Inputdeck['Bmag']
    if markercount == 0:
        markersize_ = 0
        markersize_H = 0
        markersize_L = 0
    if 'G' in flags:
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
        axs.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
        plt.tight_layout()
    if 'E' in flags:

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
        axs.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
        plt.tight_layout()
    plt.xlabel('Magnetic Field (T)', weight='semibold')
    plt.ylabel('Energy (cm^-1)', weight='semibold')
    plt.legend(loc='best')
    plt.tight_layout()
    if savefig==True:
        plt.savefig(f'{figname}_Excited.png',dpi=2000) #Save the figure if requested
    return Zfantest
