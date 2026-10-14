"""Numerical output must not change during pure code moves (see golden_cases.py)."""
import numpy as np
import pytest

import golden_cases as gc

REFERENCE = np.load(gc.GOLDEN)

# Eigenvectors and quantities built from them depend on the BLAS/LAPACK build
# (sign, and mixing inside degenerate subspaces), so only gauge-invariant
# outputs are compared: energies, wavelengths, strengths, labels and spectra.
GAUGE_DEPENDENT = ("EigGround[1]", "EigExcited[1]", "afcf", "Red_Dipole", "RedDipoleOp",
                   "Zsig_norad", "A_Ein_polavg")


def comparable(key):
    return not key.endswith(GAUGE_DEPENDENT)


@pytest.fixture(scope="module")
def fresh():
    return gc.cases()


def test_same_set_of_outputs(fresh):
    assert {k.replace("/", "|") for k in fresh} == set(REFERENCE.files)
    assert sum(comparable(k) for k in fresh) > 400


def test_outputs_match_reference(fresh):
    bad = []
    for key, value in fresh.items():
        if not comparable(key):
            continue
        ref = REFERENCE[key.replace("/", "|")]
        atol = 1e-9*max(1.0, np.max(np.abs(ref))) if ref.size else 0
        if np.shape(value) != ref.shape or not np.allclose(value, ref, rtol=1e-7, atol=atol, equal_nan=True):
            bad.append(key)
    assert not bad, bad[:10]
