"""Plot E1 polarization factors versus viewing and linear-analyzer angles.

Uses the package polarization function. Each curve is an intensity factor for
one component, before multiplication by its atomic E1 strength. Sigma branches
have equal factors in this geometry. Angles are in degrees.

Adapted from the polarization investigations by Diash.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from AtomSpect import polarization


def make_figures():
    """Return viewing-angle, analyzer-angle, and transmission comparisons."""
    angles = np.linspace(0, 180, 361)
    channels = ((0, r"$\pi$", "tab:blue", "-"),
                (1, r"$\sigma$ (each branch)", "tab:orange", "--"))
    figures = {}

    def panel(stem, title, xlabel, **kwargs):
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        for q, label, color, style in channels:
            amplitude, _ = polarization(q, 0, **kwargs)
            ax.plot(angles, np.asarray(amplitude) ** 2,
                    label=label, color=color, linestyle=style)
        ax.set(title=title, xlabel=xlabel, ylabel="Intensity factor per component",
               xlim=(0, 180), ylim=(0, 1.05))
        ax.set_xticks(np.arange(0, 181, 30))
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend()
        figures[stem] = fig

    panel("ViewingAngle", "No analyzer", "Viewing angle to magnetic field (degrees)",
          Btheta=angles)
    panel("AnalyzerAngle", "Linear analyzer; viewing angle = 90 degrees",
          "Analyzer angle (degrees)", Btheta=90, gamma=angles, Pol_Filter=True)
    panel("Transmission", "No analyzer; intensity transmissions: alpha = 0.25, beta = 1",
          "Viewing angle to magnetic field (degrees)", Btheta=angles, TAlpha=0.25, TBeta=1)
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
    print(f"Calculated {len(figures)} polarization comparisons.")
    if args.no_show:
        for fig in figures.values():
            plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
