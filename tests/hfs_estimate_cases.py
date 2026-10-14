"""Deterministic cases that pin the numbers of the hyperfine-constant estimator.

`python tests/hfs_estimate_cases.py <module>` writes tests/golden/hfs_estimate.json from
`<module>` (default AtomSpect.hfs_estimate). tests/test_hfs_estimate.py compares fresh values
with that file, so moving or tidying the module must not change any estimate.
"""
import importlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPMATH_NOGMPY", "1")
os.environ.setdefault("SYMPY_GROUND_TYPES", "python")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GOLDEN = Path(__file__).parent / "golden" / "hfs_estimate.json"
TABLE = ROOT / "HFS Calcs" / "BulkHFSImport" / "HFSInputs.csv"
FLAGS = ["", "F", "H", "de", "ds", "FHdelepsds"]


def cases(module):
    out = {}
    rows = module.read_HFSInput(str(TABLE))
    for flags in FLAGS:
        for row in rows:
            row = dict(row)
            row["Corrflags"] = flags
            out[f"A/{flags or 'none'}/{row['Label']}"] = float(module.Ahfs(row)["A"])
    # Direct helper calls with fixed arguments.
    out["Ahfs_Kopf"] = module.Ahfs_Kopf(37, 5, 0.1, 1.35298, 2.5, 1, 1.5)
    out["Ahfs_Kopf2"] = module.Ahfs_Kopf2(37, 1.0, 1.35298, 2.5, 1, 1.5)
    out["Bhfs_kopf"] = module.Bhfs_kopf(25.0, 1, 1.5, 2.5, 1.35298, 0.277)
    out["Bhfsconst"] = module.Bhfsconst(37, 5, 1, 1.5, 0.277, 1.0)
    out["A_Bransden"] = module.A_Bransden(37, 5, 0.1, 1.35298, 2.5, 85, 1, 1.5)
    out["Qdefect"] = list(module.Qdefect(4.177128, 5))
    out["QdefCorr"] = module.QdefCorr(4.177128, 1.57923, 5, 2.496113, 6)
    out["A_ratio"] = module.A_ratio(1011.9, 1.35298, 2.5, 2.75124, 1.5)
    out["Isotope"] = module.Isotope(4.177128, 85, 87)
    out["Isotopediff"] = module.Isotopediff(4.177128, 85, 87)
    out["GenCouple"] = module.GenCouple(2.5, 0.5, 1.5, 1)
    out["cm_to_MHz"] = getattr(module, "cm_to_MHz", module.cm_to_Hz)(0.1)
    for name in ("ev_to_MHz", "cm_to_nm", "cm_to_ev", "ev_to_cm", "Hz_to_nm"):
        out[name] = getattr(module, name)(1.2345)
    for name, args in (("epscorr", (37, 85, 0.5)), ("deltacorr", (37, 85, 0.5)), ("Fcorrj", (37, 0.5)),
                       ("Rcorr", (37, 0, 0.5, 1)), ("Hcorr", (37, 0))):
        out[name] = getattr(module, name)(*args)
    return out


def jsonable(value):
    return [float(v) for v in value] if isinstance(value, (list, tuple)) else float(value)


if __name__ == "__main__":
    mod = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "AtomSpect.hfs_estimate")
    data = {k: jsonable(v) for k, v in cases(mod).items()}
    GOLDEN.write_text(json.dumps(data, indent=0, sort_keys=True))
    print(len(data), "values from", mod.__name__)
