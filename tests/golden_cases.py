"""Deterministic reference cases used to guard the refactor.

`python tests/golden_cases.py` regenerates tests/golden/reference.npz from the
current code. tests/test_golden.py compares fresh output to that file, so a
pure code move must leave every number unchanged.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MPMATH_NOGMPY", "1")
os.environ.setdefault("SYMPY_GROUND_TYPES", "python")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import numpy as np
import AtomSpect

GOLDEN = Path(__file__).parent / "golden" / "reference.npz"

HE = dict(s_ground=1, s_excited=1, l_ground=1, l_excited=2,
          E_ground=np.array([169087.8308131, 169086.8428979, 169086.7664725]),
          E_excited=np.array([191444.49804915, 191444.47952984, 191444.47832914]),
          Bmag=0.5, b_angle=70, amu=4, Temp=300, spec_window=[446.8, 447.6],
          plot_window=[447.0, 447.3], specstep=0.002222, specres=3,
          Skewness=[1.24, 1.98], fxnwindow=0.1)
RB = dict(s_ground=.5, s_excited=.5, l_ground=0, l_excited=1,
          E_ground=np.array([0.]), E_excited=np.array([12578.95, 12816.545]),
          I_spin=2.5, mu_I=1.35298, HFS_G=[1011.910813e6],
          HFS_E=[120.527e6, 25.0354e6], HFSB_E=[0., 25.898e6],
          Bmag=.1, b_angle=90, amu=85, Temp=300, spec_window=[780.0, 795.0],
          plot_window=[794.5, 795.5], specstep=.01, specres=2)


def flatten(obj, prefix, out):
    """Collect every numeric array/scalar reachable from obj into out."""
    if isinstance(obj, dict):
        for key, val in obj.items():
            flatten(val, f"{prefix}/{key}", out)
    elif isinstance(obj, (list, tuple)):
        try:
            arr = np.asarray(obj)
            if arr.dtype != object and np.issubdtype(arr.dtype, np.number):
                out[prefix] = arr
                return
        except Exception:
            pass
        for i, val in enumerate(obj):
            flatten(val, f"{prefix}[{i}]", out)
    elif isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.number):
            out[prefix] = obj
    elif isinstance(obj, (int, float, complex, np.number)) and not isinstance(obj, bool):
        out[prefix] = np.asarray(obj)


def cases():
    out = {}
    for name, base in (("he", HE), ("rb", RB)):
        for fxn in ("Skewed", "Lorrentzian", "Gaussian", "GaussianInstrum", "Voigt"):
            deck = dict(base, Convfxn=fxn)
            flatten(AtomSpect.AtomSpect_Main(deck), f"main/{name}/{fxn}", out)
    flatten(AtomSpect.AtomSpect_Main(dict(HE, Pol_angle=30, DoLowSig=True,
                                           DoHighSig=True, EtermG=169087., EtermE=191444.)), "main/he/filter", out)
    fan = dict(HE, Bmag=np.linspace(0, 1, 7), EtermG=169087., EtermE=191444.)
    flatten(AtomSpect.ZeemanFan(fan, "GEHL"), "fan/he", out)
    fan = dict(RB, Bmag=np.linspace(0, .5, 7))
    flatten(AtomSpect.ZeemanFan(fan, "GEH"), "fan/rb", out)

    # Direct calls into the level/Hamiltonian/signal layers (cover sortE, every
    # field regime, both PsiHFS settings and the unlabelled signal path).
    for sort in (False, True):
        for name, args in (("g", (1, 1, HE["E_ground"])), ("e", (2, 1, HE["E_excited"]))):
            lev = AtomSpect.stateprocess(*args, 0.5, sortE=sort)
            flatten(lev, f"levels/he{name}/sort{sort}", out)
        gi = AtomSpect.lande_gi(RB["I_spin"], RB["mu_I"])
        for name, L, E, A, B in (("g", 0, RB["E_ground"], RB["HFS_G"], [0.]),
                                 ("e", 1, RB["E_excited"], RB["HFS_E"], RB["HFSB_E"])):
            lev = AtomSpect.stateprocess_HFS(L, .5, RB["I_spin"], {"HFSA": A, "HFSB": B}, gi, E, 0.1, sortE=sort)
            flatten(lev, f"levels/rb{name}/sort{sort}", out)
    lg = AtomSpect.stateprocess(1, 1, HE["E_ground"], 0.5)
    le = AtomSpect.stateprocess(2, 1, HE["E_excited"], 0.5)
    for regime, kw in (("mid", {}), ("low", {"Lowfield": True}), ("high", {"Highfield": True})):
        eg, ee = ({"Eterm": 169087.}, {"Eterm": 191444.}) if kw.get("Highfield") else ({}, {})
        zg, ze = AtomSpect.Zeeman_func(lg, 0.5, **kw, **eg), AtomSpect.Zeeman_func(le, 0.5, **kw, **ee)
        flatten([zg[0], ze[0]], f"zfunc/he/{regime}/energies", out)
        if regime != "high":  # Exactly degenerate levels have no unique per-eigenpair labelling.
            flatten(AtomSpect.Zeeman_signal(lg, zg, le, ze, Bangle=70, gamma=20, Filter=True), f"zsig/he/{regime}", out)
        flatten(AtomSpect._zeeman_signal(lg, zg, le, ze, label_eigenpairs=False), f"zsig/he/{regime}/unlabelled", out)
    zg, ze = AtomSpect.Zeeman_func(lg, 0.5), AtomSpect.Zeeman_func(le, 0.5)
    flatten(AtomSpect.Zeeman_sig_norad(lg, zg, le, ze), "zsig/he/norad", out)
    flatten(AtomSpect.HZeeman(lg), "hzeeman/he", out)
    gi = AtomSpect.lande_gi(RB["I_spin"], RB["mu_I"])
    rg = AtomSpect.stateprocess_HFS(0, .5, 2.5, {"HFSA": RB["HFS_G"], "HFSB": [0.]}, gi, RB["E_ground"], 0.1)
    re_ = AtomSpect.stateprocess_HFS(1, .5, 2.5, {"HFSA": RB["HFS_E"], "HFSB": RB["HFSB_E"]}, gi, RB["E_excited"], 0.1)
    for psi in (True, False):
        zg, ze = AtomSpect.Zeeman_func_HFS(rg, 0.1, gi, psi), AtomSpect.Zeeman_func_HFS(re_, 0.1, gi, psi)
        flatten([zg[0], ze[0]], f"zfunc/rb/psi{psi}/energies", out)
        flatten(AtomSpect.Zeeman_signal_HFS(rg, zg, re_, ze, Bangle=70, gamma=20, Filter=True), f"zsig/rb/psi{psi}", out)
    flatten(AtomSpect.HZeeman_HFS(rg, gi), "hzeeman/rb", out)

    # Units and wavelength conversions.
    x = np.array([200., 450., 800., 1500.])
    for fn in ("Hz_to_nm", "nm_to_Hz", "ev_to_MHz", "cm_to_nm", "cm_to_ev", "ev_to_cm",
               "Vac_to_air", "air_to_vac"):
        flatten(getattr(AtomSpect, fn)(x), f"units/{fn}", out)
    flatten(AtomSpect.cm_to_Hz(x), "units/cm_to_Hz", out)

    # Lineshapes and convolution helpers.
    xs = np.linspace(599.9e-9, 600.1e-9, 501)
    flatten(AtomSpect.Gaussian(xs, 600e-9, 300, 40, v_ion=1e3), "shape/gauss", out)
    flatten(AtomSpect.GaussianInstrum(xs, 600e-9, 300, 40, pixel_size=5e-12, bidir=True), "shape/gaussi", out)
    flatten(AtomSpect.Voigt(xs, 600e-9, 300, 40, pixel_size=5e-12), "shape/voigt", out)
    flatten(AtomSpect.Lorrentzian(xs, 600e-9, pixel_size=5e-12, Skewness=[.6, 1.]), "shape/lorentz", out)
    flatten(AtomSpect.Normalize(np.array([1., 2., 4.]), 3.), "shape/normalize", out)
    waves = np.array([600.0, 600.01, 600.01, 601.0, 601.2])
    sig = np.array([1., .5, .2, .8, .1])
    flatten(AtomSpect.make_stickbins(waves, sig, windsize=.1), "conv/bins", out)
    flatten(AtomSpect.Convol_Sticks(600., [599.9, 600.1], 300, 40, .002, function="Voigt"), "conv/sticks", out)
    flatten(AtomSpect.Convol_Spect(waves, sig, [599, 602], [599.5, 601.5], 40, .002,
                                   Temperature_in=300, functiontype="Gaussian"), "conv/spect", out)
    flatten(AtomSpect.MultiSpec([AtomSpect.AtomSpect_Main(dict(HE, Convfxn="Skewed"))] * 2, [1, .5]),
            "conv/multi", out)
    return out


if __name__ == "__main__":
    data = cases()
    GOLDEN.parent.mkdir(exist_ok=True)
    np.savez_compressed(GOLDEN, **{k.replace("/", "|"): v for k, v in data.items()})
    print(len(data), "arrays,", GOLDEN.stat().st_size // 1024, "KiB")
