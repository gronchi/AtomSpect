"""Redraw manual figures from the checked CSV data without rerunning AtomSpect."""
import csv
import shutil
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Latin Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 9,
    'axes.labelsize': 9,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': .6,
    'lines.linewidth': 1.,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': .02,
})
BLUE, ORANGE, AQUA, GREY = '#2a6fb0', '#d9822b', '#2f9e8f', '#555555'
SEQ = ['#0b3c6b', '#2a6fb0', '#5b9bd5', '#94c0e6', '#c9def2']


def numeric_csv(path):
    """Read named columns and require finite numerical values."""
    with path.open(newline='') as handle:
        rows = list(csv.reader(handle))
    values = np.asarray(rows[1:], dtype=float)
    if values.ndim != 2 or values.shape[1] != len(rows[0]) or not np.all(np.isfinite(values)):
        raise ValueError(f'Invalid numerical data: {path}')
    return rows[0], values


def generate(figures):
    """Draw all four PDFs in figures from its data/ subdirectory."""
    figures = Path(figures)
    for filename, output, prefix, title, ylabel, approximation in (
        ('zeeman_fan_he_3P.csv', 'fig_zeeman_fan.pdf', 'low', 'He I 1s2p $^3$P Zeeman fan',
         'Energy relative to mean $E_J$ (cm$^{-1}$)', 'low-field ($g_J$)'),
        ('hfs_fan_rb85_5S.csv', 'fig_hfs_fan.pdf', 'high', '$^{85}$Rb 5S$_{1/2}$ hyperfine fan',
         'Energy (cm$^{-1}$)', 'hyperfine Paschen-Back'),
    ):
        header, values = numeric_csv(figures/'data'/filename)
        fig, ax = plt.subplots(figsize=(3.3, 2.5))
        for column, name in enumerate(header[1:], 1):
            exact = name.startswith('exact')
            if not exact and not name.startswith(prefix):
                raise ValueError(f'Unexpected fan column: {name}')
            ax.plot(values[:, 0], values[:, column], color=BLUE if exact else GREY,
                    lw=1 if exact else .8, ls='-' if exact else '--')
        ax.plot([], [], color=BLUE, label='exact within the term')
        ax.plot([], [], color=GREY, ls='--', label=approximation)
        ax.set(xlabel='Magnetic field (T)', ylabel=ylabel, title=title)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures/output)
        plt.close(fig)

    with (figures/'data'/'polarization_sticks.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=True)
    for ax, angle in zip(axes, (90, 0)):
        angle_rows = [row for row in rows if float(row['b_angle_deg']) == angle]
        peak = max(float(row['strength']) for row in angle_rows)
        for key, label, color, linestyle in (('pi_sticks', r'$\pi$', BLUE, '-'),
                                              ('sigmaminus_sticks', r'$\sigma^-$', ORANGE, '--'),
                                              ('sigmaplus_sticks', r'$\sigma^+$', ORANGE, '-')):
            selected = [row for row in angle_rows if row['component'] == key]
            waves = np.asarray([float(row['wavelength_nm']) for row in selected])
            strengths = np.asarray([float(row['strength']) for row in selected])
            keep = strengths > 1e-9*peak
            ax.vlines(waves[keep], 0, strengths[keep]/peak, color=color, lw=.9,
                      linestyles=linestyle, label=label)
        ax.set(xlim=(447.1, 447.2), title=rf'B at {angle}$^\circ$ to the line of sight',
               xlabel='Air wavelength (nm)')
    axes[0].set_ylabel('Peak-scaled strength')
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures/'fig_polarization.pdf')
    plt.close(fig)

    header, values = numeric_csv(figures/'data'/'lineshapes.csv')
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    labels = {'Gaussian (Doppler)': 'Doppler', 'Gaussian + instrument': 'Doppler + Gaussian',
              'Voigt': 'Voigt', 'Lorentzian (FWHM 10 pm)': 'Lorentzian'}
    styles = {'Gaussian (Doppler)': (GREY, '--'), 'Gaussian + instrument': (SEQ[3], '-'),
              'Voigt': (SEQ[1], '-'), 'Lorentzian (FWHM 10 pm)': (SEQ[0], '-')}
    for column, name in enumerate(header[1:], 1):
        color, linestyle = styles[name]
        ax.plot((values[:, 0]-600)*1000, values[:, column]*1e-9,
                color=color, ls=linestyle, label=labels[name])
    ax.set(xlabel='Offset from line centre (pm)', ylabel='Profile (nm$^{-1}$)',
           title='Unit-area line profiles')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures/'fig_lineshapes.pdf')
    plt.close(fig)


def main():
    """Stage plots before replacing the existing figure PDFs."""
    destination = Path(__file__).resolve().parents[1]/'figures'
    with tempfile.TemporaryDirectory(prefix='atomspect-plots-') as temporary:
        staging = Path(temporary)
        shutil.copytree(destination/'data', staging/'data')
        generate(staging)
        for path in staging.glob('*.pdf'):
            shutil.copy2(path, destination/path.name)
    print(f'Four figures redrawn from CSV data in {destination}')


if __name__ == '__main__':
    main()
