"""The hyperfine-constant estimator keeps the numbers it had as a standalone script."""
import json
import math

import pytest

import hfs_estimate_cases as cases
from AtomSpect import hfs_estimate

REFERENCE = json.loads(cases.GOLDEN.read_text())


@pytest.fixture(scope="module")
def fresh():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {k: cases.jsonable(v) for k, v in cases.cases(hfs_estimate).items()}


def test_same_cases(fresh):
    assert set(fresh) == set(REFERENCE)


def test_values_unchanged(fresh):
    bad = []
    for key, ref in REFERENCE.items():
        got = fresh[key]
        pairs = zip(got, ref) if isinstance(ref, list) else [(got, ref)]
        for a, b in pairs:
            if math.isnan(b):
                ok = math.isnan(a)
            else:
                ok = math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-14)
            if not ok:
                bad.append(key)
    assert not bad, bad[:5]


def test_rubidium_p_levels_are_close_to_experiment():
    rows = {r["Label"]: r for r in hfs_estimate.read_HFSInput(str(cases.TABLE))}
    for label in ("Rb5P12", "Rb5P32"):
        row = rows[label]
        estimate = hfs_estimate.Ahfs(dict(row, Corrflags=""))["A"]
        assert hfs_estimate.pererror(row["A_Exp"], estimate) < 10


def test_module_reuses_package_units():
    from AtomSpect import units
    assert hfs_estimate.ev_to_MHz is units.ev_to_MHz
    assert hfs_estimate.cm_to_MHz(1.0) == pytest.approx(units.cm_to_Hz(1.0)/1e6)
