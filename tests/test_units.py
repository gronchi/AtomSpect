"""Air wavelength conversions against published refractive indices."""
import os

os.environ.setdefault("MPMATH_NOGMPY", "1")
os.environ.setdefault("SYMPY_GROUND_TYPES", "python")

import numpy as np
import pytest

from AtomSpect.units import Vac_to_air, air_to_vac


@pytest.mark.parametrize('wavelength, index', [
    (780.241209686, 1.000267308),
    (794.97885098, 1.000267215),
])
def test_steck_laboratory_indices(wavelength, index):
    # Steck Rb-87 Tables 3/4 and Eq. (3): 23 C, 1 atm, 40% RH.
    # Saturation vapor pressure at 23 C is approximately 2809 Pa.
    air = Vac_to_air(wavelength, temperature_c=23.0,
                     water_vapor_pressure_pa=0.4 * 2809.0)
    assert wavelength / air == pytest.approx(index, rel=0, abs=1e-9)


@pytest.mark.parametrize('conditions', [
    {},
    dict(pressure_pa=90000.0, temperature_c=23.0,
         water_vapor_pressure_pa=1123.6),
])
def test_air_vacuum_round_trip(conditions):
    vacuum = np.array([[350., 450., 500.], [650., 780., 795.]])
    air = Vac_to_air(vacuum, **conditions)
    assert np.all(air < vacuum)
    np.testing.assert_allclose(air_to_vac(air, **conditions), vacuum,
                               rtol=2e-15, atol=0)
    assert air_to_vac(Vac_to_air(500., **conditions), **conditions) == pytest.approx(500., abs=1e-12)


def test_zero_pressure_has_no_refraction():
    wavelengths = [350., 500., 795.]
    np.testing.assert_array_equal(Vac_to_air(wavelengths, pressure_pa=0), wavelengths)
    np.testing.assert_array_equal(air_to_vac(wavelengths, pressure_pa=0), wavelengths)


def test_default_conditions_are_dry_air_at_20_c():
    assert Vac_to_air(500.) == Vac_to_air(
        500., pressure_pa=101325., temperature_c=20., water_vapor_pressure_pa=0.)
    assert air_to_vac(500.) == air_to_vac(
        500., pressure_pa=101325., temperature_c=20., water_vapor_pressure_pa=0.)
