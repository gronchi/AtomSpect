"""Regenerate the manual's data, analytic checks and figures from the AtomSpect package.

Run from the repository root:  python doc/scripts/make_figures.py
Writes doc/figures/data/*.csv, doc/figures/data/provenance.txt and doc/figures/*.pdf.
The analytic checks must pass before any file is written.
"""
import csv
import datetime
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("MPMATH_NOGMPY", "1")
os.environ.setdefault("SYMPY_GROUND_TYPES", "python")
import matplotlib
matplotlib.use("Agg")
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import AtomSpect  # noqa: E402

FIG = ROOT / "doc" / "figures"
DATA = FIG / "data"

# Zero-field energies in cm^-1 (the repository's He I and Rb examples; not independently validated).
HE_G = np.array([169087.8308131, 169086.8428979, 169086.7664725])   # 1s2p 3P J=0,1,2
HE_E = np.array([191444.49804915, 191444.47952984, 191444.47832914])  # 1s4d 3D J=1,2,3
RB = dict(s_ground=.5, s_excited=.5, l_ground=0, l_excited=1, E_ground=np.array([0.]),
          E_excited=np.array([12578.95, 12816.545]), I_spin=2.5, mu_I=1.35298,
          HFS_G=[1011.910813e6], HFS_E=[120.527e6, 25.0354e6], HFSB_E=[0., 25.898e6],
          b_angle=90, amu=85)
HE = dict(s_ground=1, s_excited=1, l_ground=1, l_excited=2, E_ground=HE_G, E_excited=HE_E,
          b_angle=90, amu=4, Temp=300, spec_window=[446.9, 447.4], plot_window=[447.0, 447.3],
          specstep=0.002222, specres=3, Convfxn="Skewed", Skewness=[1.24, 1.98], fxnwindow=0.1)
HZ_TO_CM = 1/(100*AtomSpect.c_light)


def write_csv(name, header, rows):
    """Write a header and rows into the active data directory."""
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / name, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


# ------------------------------------------------------------------ checks
def run_checks():
    """Return [(name, measured, tolerance)] and raise if a check fails."""
    checks = []

    # 1. Weak-field slope of the 3P2 sublevels: dE = gJ * m * muB * B (cm^-1). J=2 is the lowest
    #    manifold for these inputs; a small field keeps the J-mixing (second-order) term below 1e-9.
    b = 1e-5
    level = AtomSpect.stateprocess(1, 1, HE_G, b)
    energies = np.sort(AtomSpect.Zeeman_func(level, b)[0])[:5]
    gj = AtomSpect.lande_gj2(2, 1, 1)
    err = np.max(np.abs(energies - HE_G[2] - gj*AtomSpect.muBcm*b*np.array([-2, -1, 0, 1, 2])))
    checks.append(("weak_field_zeeman_3P2_cm-1", err, 1e-9))

    # 2. Hyperfine interval of 85Rb 5S1/2 at B = 0 equals A*(I + 1/2).
    deck = dict(RB, Bmag=np.array([0.0]))
    fan = AtomSpect.ZeemanFan(deck, "G")["G"][:, 0]
    interval = np.max(fan) - np.min(fan)
    expected = 1011.910813e6*(2.5 + .5)*HZ_TO_CM
    checks.append(("rb85_5S_hyperfine_interval_relative", abs(interval - expected)/expected, 1e-9))

    # 3. The relative strengths are normalized.
    result = AtomSpect.AtomSpect_Main(dict(HE, Bmag=0.5, b_angle=60))
    checks.append(("strength_normalization", abs(np.sum(result["signal"]) - 1), 1e-12))

    # 4. E1 has no |Delta M| > 1 component.
    checks.append(("no_forbidden_components", float(np.sum(result["other_sticks"][1])), 1e-14))

    # 5. Looking along the field only sigma light reaches the detector: pi strength vanishes.
    along = AtomSpect.AtomSpect_Main(dict(HE, Bmag=0.5, b_angle=0))
    checks.append(("pi_strength_along_field", float(np.sum(along["pi_sticks"][1])), 1e-12))

    # 6. Lineshapes carry unit area and the stated FWHM.
    x = np.linspace(600e-9 - 2e-9, 600e-9 + 2e-9, 400001)
    area = lambda y: np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)  # noqa: E731
    checks.append(("gaussian_unit_area", abs(area(AtomSpect.Gaussian(x, 600e-9, 300, 40)) - 1), 1e-3))
    checks.append(("voigt_unit_area", abs(area(AtomSpect.Voigt(x, 600e-9, 300, 40, pixel_size=5e-12)) - 1), 1e-3))
    lor = AtomSpect.Lorrentzian(x, 600e-9, pixel_size=5e-12)
    half = x[lor >= lor.max()/2]
    checks.append(("lorentzian_fwhm_m", abs((half[-1] - half[0]) - 5e-12), 2e-14))

    # 7. Compare weak-field gF*mF with direct projection of the exact operator.
    gi = AtomSpect.lande_gi(1.5, 1.)
    hfs_level = AtomSpect.stateprocess_HFS(0, .5, 1.5, {'HFSA': [1e6]}, gi, [0.], 0.)
    operator = AtomSpect.HZeeman_HFS(hfs_level, gi)['H_Z']
    errors = []
    for index, state in enumerate(hfs_level[-1]):
        F, mF, J, _, I, _, L, _, S, _ = state
        errors.append(abs(AtomSpect.lande_gf(F, I, J, L, S, gi)*mF - operator[index, index]))
    checks.append(('hfs_weak_field_projection', max(errors), 1e-12))

    # 8. Bins with overlapping padding must not duplicate component profiles.
    waves = np.array([600., 600.15, 600.16])
    spectrum, _, bins = AtomSpect.Convol_Spect(waves, np.ones(3), [599, 601], [599, 601], 40, .002)
    inside = (spectrum[0] >= bins[0][0]) & (spectrum[0] <= bins[0][1])
    grid = spectrum[0][inside]*1e-9
    expected = sum(AtomSpect.Lorrentzian(grid, wave*1e-9, pixel_size=.002e-9) for wave in waves)
    residual = np.max(np.abs(spectrum[1][inside] - expected))/np.max(expected)
    checks.append(('overlapping_bin_profile_sum', residual, 1e-12))

    # 9. An analyzer with zero transmission yields a dark spectrum, not NaN.
    dark = AtomSpect.AtomSpect_Main(dict(HE, Bmag=.5, TAlpha=0., TBeta=0.))
    checks.append(('dark_spectrum_strength', float(np.max(np.abs(dark.signal))), 0.))

    for name, value, tol in checks:
        if not value <= tol:
            raise AssertionError(f"analytic check {name} failed: {value:.3e} > {tol:.1e}")
    write_csv("validation_checks.csv", ["check", "measured", "tolerance"], [(n, f"{v:.3e}", f"{t:.0e}") for n, v, t in checks])
    latex_rows = []
    for n, v, t in checks:
        escaped = n.replace("_", "\\_")
        latex_rows.append("\\texttt{" + escaped + "} & " + f"{v:.2e} & {t:.0e}" + " \\\\")
    table = ["\\begin{tabular}{lcc}", "\\toprule", "Check & measured & tolerance \\\\", "\\midrule"]
    table += latex_rows + ["\\bottomrule", "\\end{tabular}"]
    (DATA / "validation_checks.tex").write_text("\n".join(table) + "\n", newline="\n")
    return checks


# ------------------------------------------------------------------ figures
def fig_zeeman_fan():
    """Write exact and weak-field He 3P fan data, relative to mean E_J."""
    fields = np.linspace(0, 1, 101)
    fan = AtomSpect.ZeemanFan(dict(HE, Bmag=fields), 'GL')
    offset = np.mean(HE_G)
    exact, low = fan['G']-offset, fan['LowG']-offset
    header = ['B_T'] + [f'exact{i}_cm-1' for i in range(len(exact))]
    header += [f'low{i}_cm-1' for i in range(len(low))]
    write_csv('zeeman_fan_he_3P.csv', header, np.column_stack([fields, exact.T, low.T]))


def fig_hfs_fan():
    """Write exact and hyperfine Paschen-Back Rb 5S fan data."""
    fields = np.linspace(0, .5, 101)
    fan = AtomSpect.ZeemanFan(dict(RB, Bmag=fields), 'GH')
    header = ['B_T'] + [f'exact{i}_cm-1' for i in range(len(fan['G']))]
    header += [f'high{i}_cm-1' for i in range(len(fan['HighG']))]
    write_csv('hfs_fan_rb85_5S.csv', header, np.column_stack([fields, fan['G'].T, fan['HighG'].T]))


def fig_polarization():
    """Write He 447 nm sticks for transverse and axial observation."""
    rows = []
    for angle in (90, 0):
        result = AtomSpect.AtomSpect_Main(dict(HE, Bmag=.5, b_angle=angle))
        for key in ('pi_sticks', 'sigmaminus_sticks', 'sigmaplus_sticks'):
            rows.extend((angle, key, w, s) for w, s in zip(*result[key]))
    write_csv('polarization_sticks.csv', ['b_angle_deg', 'component', 'wavelength_nm', 'strength'], rows)


def fig_lineshapes():
    """Write per-metre profile densities on a wavelength grid in nm."""
    x = np.linspace(600e-9 - 4e-11, 600e-9 + 4e-11, 2001)
    curves = {'Gaussian (Doppler)': AtomSpect.Gaussian(x, 600e-9, 3000, 40),
              'Gaussian + instrument': AtomSpect.GaussianInstrum(x, 600e-9, 3000, 40, pixel_size=1e-11),
              'Voigt': AtomSpect.Voigt(x, 600e-9, 3000, 40, pixel_size=1e-11),
              'Lorentzian (FWHM 10 pm)': AtomSpect.Lorrentzian(x, 600e-9, pixel_size=1e-11)}
    write_csv('lineshapes.csv', ['wavelength_nm'] + list(curves), np.column_stack([x*1e9] + list(curves.values())))


def write_provenance(checks):
    """Record runtime, source digest, settings and validation scope."""
    import hashlib
    import scipy
    import subprocess
    import sympy
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, capture_output=True)
    commit = result.stdout.strip() if result.returncode == 0 else 'unavailable'
    digest = hashlib.sha256()
    sources = sorted((ROOT/'AtomSpect').rglob('*.py')) + sorted((ROOT/'doc'/'scripts').glob('*.py'))
    for path in sources:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    lines = [f'Generated UTC: {datetime.datetime.now(datetime.timezone.utc).isoformat()}',
             f'Source commit: {commit}; SHA256 of package and documentation scripts: {digest.hexdigest()}',
             f'Python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}, '
             f'sympy {sympy.__version__}, matplotlib {matplotlib.__version__}',
             f'Platform: {platform.platform()}; SymPy backend: {os.environ.get("SYMPY_GROUND_TYPES")}',
             'Inputs: He I 1s2p 3P / 1s4d 3D and 85Rb 5S1/2 -> 5P energies and hyperfine constants from repository examples.',
             'Fan grids: 101 points, He 0..1 T and Rb 0..0.5 T. Polarization: B=0.5 T, angles 90 and 0 degrees.',
             'Profiles: 600 nm, 3000 K, mass 40, instrument FWHM 0.01 nm; 2001 samples over +/-0.04 nm.',
             f'Analytic checks passed: {len(checks)}',
             'Figures illustrate the model; they are not a validation against experiment.']
    (DATA/'provenance.txt').write_text('\n'.join(lines)+'\n')


def main():
    """Stage outputs until all checks, calculations and requested plots succeed."""
    import argparse
    import shutil
    import tempfile
    global FIG, DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-only', action='store_true', help='write checked data without drawing figures')
    args = parser.parse_args()
    destination = ROOT/'doc'/'figures'
    with tempfile.TemporaryDirectory(prefix='atomspect-doc-') as temporary:
        FIG = Path(temporary)
        DATA = FIG/'data'
        DATA.mkdir()
        checks = run_checks()
        fig_zeeman_fan()
        fig_hfs_fan()
        fig_polarization()
        fig_lineshapes()
        write_provenance(checks)
        if not args.data_only:
            from plot_figures import generate
            generate(FIG)
        for path in sorted(FIG.rglob('*')):
            if path.is_file():
                target = destination/path.relative_to(FIG)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    FIG, DATA = destination, destination/'data'
    print(f'{len(checks)} checks passed; outputs written to {destination}')


if __name__ == '__main__':
    main()
