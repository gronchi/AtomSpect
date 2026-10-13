"""Numerical output must not change during pure code moves (see golden_cases.py)."""
import numpy as np
import pytest

import golden_cases as gc

REFERENCE = np.load(gc.GOLDEN)


@pytest.fixture(scope="module")
def fresh():
    return gc.cases()


def test_same_set_of_outputs(fresh):
    assert {k.replace("/", "|") for k in fresh} == set(REFERENCE.files)


def test_outputs_match_reference(fresh):
    bad = []
    for key, value in fresh.items():
        ref = REFERENCE[key.replace("/", "|")]
        if np.shape(value) != ref.shape or not np.allclose(value, ref, rtol=1e-9, atol=1e-10, equal_nan=True):
            bad.append(key)
    assert not bad, bad[:10]
