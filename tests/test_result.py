"""The result of AtomSpect_Main: dict behaviour plus attribute access."""
import copy
import os
import pickle

os.environ.setdefault("MPMATH_NOGMPY", "1")
os.environ.setdefault("SYMPY_GROUND_TYPES", "python")
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
import AtomSpect
from AtomSpect.result import SpectrumResult

FINE = dict(s_ground=0, s_excited=0, l_ground=0, l_excited=1, E_ground=np.array([0.]),
            E_excited=np.array([12500.]), Bmag=.1, b_angle=90, amu=40,
            spec_window=[799, 801], plot_window=[799, 801], specstep=.01, specres=1,
            DoLowSig=True, DoHighSig=True, EtermG=0., EtermE=12500.)
HFS = dict(s_ground=.5, s_excited=.5, l_ground=0, l_excited=1, E_ground=np.array([0.]),
           E_excited=np.array([12578.95, 12816.545]), I_spin=2.5, mu_I=1.35298,
           HFS_G=[1011.910813e6], HFS_E=[120.527e6, 25.0354e6], HFSB_E=[0., 25.898e6],
           Bmag=.1, b_angle=90, amu=85, spec_window=[775, 799], plot_window=[775, 799],
           specstep=.01, specres=1)


@pytest.fixture(scope="module")
def results():
    return AtomSpect.AtomSpect_Main(dict(FINE)), AtomSpect.AtomSpect_Main(dict(HFS))


def test_is_a_dict_with_attribute_access(results):
    for result in results:
        assert isinstance(result, dict) and isinstance(result, SpectrumResult)
        assert result.signal is result['signal']
        assert result.Input is result['Input']


def test_missing_attribute_lists_available_keys(results):
    with pytest.raises(AttributeError, match="no entry 'nonsense'.*signal"):
        results[0].nonsense
    assert not hasattr(results[0], 'nonsense')


def test_set_delete_dir_and_copy(results):
    r = results[0].copy()
    assert type(r) is SpectrumResult
    r.extra = 3
    assert r['extra'] == 3 and 'extra' in dir(r)
    del r.extra
    assert 'extra' not in r
    assert 'extra' not in results[0]


def test_every_key_is_documented(results):
    for result in results:
        assert [k for k, v in result.describe().items() if v == '(undocumented)'] == []


def test_pickle_copy_and_plain_dict(results):
    r = results[1]
    assert type(copy.deepcopy(r)) is SpectrumResult
    assert type(pickle.loads(pickle.dumps(r))) is SpectrumResult
    assert type(r.to_dict()) is dict and set(r.to_dict()) == set(r)


def test_save_and_load_roundtrip_is_plain_dict(results, tmp_path):
    path = str(tmp_path / "saved")
    AtomSpect.Savenpy_Dict(path, results[0])
    loaded = AtomSpect.Loadnpy_Dict(path)
    loaded = loaded[0] if isinstance(loaded, tuple) else loaded
    assert type(loaded) is dict
    np.testing.assert_array_equal(loaded['signal'], results[0]['signal'])
