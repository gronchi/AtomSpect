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


# ---- physical checks of the corrected quadrupole routines -------------------
from AtomSpect import constants as K  # noqa: E402

H_PLANCK = 2*math.pi*K.hbar


def _radial_r3(Z, nstar, l, Z_a=1):
    return Z*Z_a**2/(K.a_o**3*nstar**3*l*(l + .5)*(l + 1))


def _angular_expectation(l, j):
    """<j,j| 3cos^2(theta) - 1 |j,j> for one electron, from the explicit |l m_l, s m_s> expansion."""
    def f(m):  # <l m| 3cos^2(theta)-1 |l m>
        return 2*(l*(l + 1) - 3*m*m)/((2*l - 1)*(2*l + 3))
    if math.isclose(j, l + .5):
        return f(l)
    weight_up = (l - j + .5)/(2*l + 1)      # |m_l = j-1/2, up>
    weight_down = (l + j + .5)/(2*l + 1)    # |m_l = j+1/2, down>
    return weight_up*f(j - .5) + weight_down*f(j + .5)


@pytest.mark.parametrize("l,j", [(1, 1.5), (2, 2.5), (2, 1.5), (3, 3.5), (3, 2.5)])
def test_quadrupole_angular_factor_matches_explicit_expectation(l, j):
    assert -_angular_expectation(l, j) == pytest.approx((2*j - 1)/(2*j + 2), rel=1e-12)


def test_bhfsconst_is_the_si_quadrupole_energy():
    Z, nstar, l, J, Q = 33, 2.3, 1, 1.5, 0.277
    expected = (K.q_charge**2*Q*1e-28*_radial_r3(Z, nstar, l)*(2*J - 1)/(2*J + 2)
                / (4*math.pi*K.eps0)/H_PLANCK/1e6)
    assert hfs_estimate.Bhfsconst(Z, nstar, l, J, Q, 1.0) == pytest.approx(expected, rel=1e-12)
    assert 15 < expected < 35   # the right order for Rb 5P3/2 (measured 25.9 MHz)
    assert hfs_estimate.Bhfsconst(Z, nstar, l, J, Q, 1.1) == pytest.approx(1.1*expected, rel=1e-12)


def test_bhfsconst_vanishes_without_a_quadrupole():
    assert hfs_estimate.Bhfsconst(33, 2.3, 0, 0.5, 0.277, 1.0) == 0
    assert hfs_estimate.Bhfsconst(33, 2.3, 1, 0.5, 0.277, 1.0) == 0


def test_bhfs_kopf_agrees_with_the_direct_constant():
    """Both routes must give the same B when A comes from the same <r^-3>."""
    Z, nstar, l, j, I, mu_I, Q = 33, 2.3, 1, 1.5, 2.5, 1.35298, 0.277
    g_i = (mu_I/I)*(K.m_elec/K.m_prot)
    A_hz = K.mu_0/(4*math.pi)*2*K.muB**2*g_i*_radial_r3(Z, nstar, l)*l*(l + 1)/(j*(j + 1))/H_PLANCK
    from_a = hfs_estimate.Bhfs_kopf(A_hz/1e6, l, j, I, mu_I, Q)
    direct = hfs_estimate.Bhfsconst(Z, nstar, l, j, Q, 1.0)
    assert from_a == pytest.approx(direct, rel=1e-9)


def test_removed_unreliable_helpers():
    assert not hasattr(hfs_estimate, "qe")
    assert not hasattr(hfs_estimate, "gI_kopf")
