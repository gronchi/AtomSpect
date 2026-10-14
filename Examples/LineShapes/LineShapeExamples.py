"""Compare line profiles and mono-/bidirectional Doppler shifts.

Run with --no-show for a headless calculation; --output-dir saves PDF and PNG
figures. Profile inputs use metres, kelvin and m/s; plots use nm and unit peaks.

Original example by Leo Nofs.
"""

import argparse
from functools import partial
from pathlib import Path
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from AtomSpect import Gaussian, GaussianInstrum, Lorrentzian, Normalize, Voigt
from AtomSpect.constants import c_light


def make_figures():
    """Return ten profile-comparison figures, keyed by output filename stem."""
    wavelength = np.linspace(428.95e-9, 429.05e-9, 1000)
    peak = 429e-9
    temperature = 10 * 11602
    mass = 183
    width = 0.010e-9
    velocities = [5000, 5500]
    profiles = {
        "Voigt": partial(Voigt, Temp_in=temperature, amu=mass, pixel_size=width),
        "Gaussian": partial(Gaussian, Temp_in=temperature, amu=mass),
        "GaussianInstrum": partial(
            GaussianInstrum, Temp_in=temperature, amu=mass, pixel_size=width
        ),
        "Lorrentzian": partial(Lorrentzian, pixel_size=width),
        "SkewedLorrentzian": partial(
            Lorrentzian, pixel_size=width, Skewness=[1, 0.3]
        ),
    }
    labels = {
        "Voigt": "Voigt",
        "Gaussian": "Doppler Gaussian",
        "GaussianInstrum": "Doppler + instrumental Gaussian",
        "Lorrentzian": "Lorentzian",
        "SkewedLorrentzian": "Skewed Lorentzian (1, 0.3)",
    }
    cases = {
        "NoDoppler": ("No bulk velocity", 0, False),
        "Vplus": ("v = +5000 m/s", velocities[0], False),
        "Vminus": ("v = -5500 m/s", -velocities[1], False),
        "Bidir": ("Equal streams: +5000 / -5500 m/s", velocities, True),
    }
    curves = {
        (name, case): Normalize(profile(wavelength, peak, v_ion=v, bidir=bidir))
        for name, profile in profiles.items()
        for case, (_, v, bidir) in cases.items()
    }
    figures = {}

    def axes_for(stem, title):
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        ax.set(title=title, xlabel="Wavelength (nm)", ylabel="Peak-normalized intensity")
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
        ax.tick_params(direction="in", top=True, right=True)
        figures[stem] = fig
        return ax

    def mark_centres(ax, case_names):
        ax.axvline(peak * 1e9, color="0.5", linestyle=":", label="Rest wavelength")
        if any(case in case_names for case in ("Vplus", "Bidir")):
            ax.axvline(peak * (1 + velocities[0] / c_light) * 1e9,
                       color="0.5", linestyle="--")
        if any(case in case_names for case in ("Vminus", "Bidir")):
            ax.axvline(peak * (1 - velocities[1] / c_light) * 1e9,
                       color="0.5", linestyle="--")
        ax.legend(fontsize=9)

    for name in profiles:
        ax = axes_for(f"{name}Example", labels[name])
        for case, (label, _, _) in cases.items():
            ax.plot(wavelength * 1e9, curves[name, case], label=label,
                    linestyle="--" if case == "Bidir" else "-")
        mark_centres(ax, cases)

    for case, (label, _, _) in cases.items():
        ax = axes_for(f"All_{case}", label)
        for name in profiles:
            ax.plot(wavelength * 1e9, curves[name, case], label=labels[name])
        mark_centres(ax, [case])

    ax = axes_for("Lineshapes", "Line profiles: T = 10 eV, mass = 183 u, width = 0.010 nm")
    for name in profiles:
        ax.plot(wavelength * 1e9, curves[name, "NoDoppler"], label=labels[name])
    ax.plot(wavelength * 1e9, curves["Gaussian", "Vplus"],
            linestyle="--", label="Doppler Gaussian, v = +5000 m/s")
    ax.plot(wavelength * 1e9, curves["SkewedLorrentzian", "Bidir"],
            linestyle="--", label="Skewed Lorentzian, equal streams")
    mark_centres(ax, ["NoDoppler"])
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Run without opening windows.")
    parser.add_argument("--output-dir", type=Path, help="Save PDF and PNG figures here.")
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend("Agg")
    figures = make_figures()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for stem, fig in figures.items():
            for extension in ("pdf", "png"):
                fig.savefig(args.output_dir / f"{stem}.{extension}", dpi=160)
    print(f"Calculated {len(figures)} line-shape comparisons.")
    if args.no_show:
        for fig in figures.values():
            plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
