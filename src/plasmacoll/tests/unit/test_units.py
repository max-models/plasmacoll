"""plasmacoll.units: astropy quantities in, plain numbers in the declared unit out."""

from __future__ import annotations

import dataclasses
import math
import subprocess
import sys
from typing import Any

import astropy.units as u
import numpy as np
import pytest

from plasmacoll import constants
from plasmacoll.units import (
    convert_units,
    dataclass_to_dict,
    describe,
    is_quantity,
    parse_quantity,
    rate_to_si,
    to_array,
    to_unit,
    unit_field,
    unit_metadata,
    with_unit,
)

KELVIN_PER_EV = constants.ELEMENTARY_CHARGE / constants.BOLTZMANN


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (0.1 * u.ns, "s", 1e-10),
        ("0.1 ns", "s", 1e-10),
        ("1e-10", "s", 1e-10),  # YAML reads 1e-10 as a string
        (3, "m", 3.0),
        (np.float32(2.0), "m", 2.0),
        (np.array(4.0), "m", 4.0),
        ([5, 3] * u.mm, "m", (5e-3, 3e-3)),
        (["5 mm", "3 mm"], "m", (5e-3, 3e-3)),
        ("[5, 3] mm", "m", (5e-3, 3e-3)),
        ("[[0, 1], [2, 3]] mm", "m", ((0.0, 1e-3), (2e-3, 3e-3))),
        ([1 * u.mm, 2.0], "m", (1e-3, 2.0)),
        (1.0 * u.eV, "J", constants.ELEMENTARY_CHARGE),
        (KELVIN_PER_EV * u.K, "eV", 1.0),
        ("1e11 cm-3", "m^-3", 1e17),
        (None, "m", None),
    ],
)
def test_to_unit(value, unit, expected) -> None:
    result = to_unit(value, unit)
    if expected is None:
        assert result is None
    else:
        assert np.allclose(result, expected, rtol=1e-6, atol=0.0)
        assert type(result) is type(expected)


def test_to_unit_keeps_mapping_keys() -> None:
    result = to_unit({"y_min": "300 K"}, "eV")
    assert result == pytest.approx({"y_min": 300.0 / KELVIN_PER_EV})


def test_to_unit_rejects_wrong_dimensions_and_mappings() -> None:
    with pytest.raises(ValueError, match="dt must be in units of s"):
        to_unit(3 * u.m, "s", "dt")
    with pytest.raises(ValueError, match="must be a plain number"):
        to_unit(3 * u.m, "", "ratio")
    with pytest.raises(ValueError, match="as a string such as"):
        to_unit({"value": 1.0, "unit": "mm"}, "m", "radius")
    with pytest.raises(ValueError, match="cannot read"):
        to_unit("five metres", "m")
    with pytest.raises(ValueError, match="cannot read"):
        parse_quantity("[1, 2")
    with pytest.raises(TypeError, match="number or a quantity"):
        to_unit(object(), "m")


def test_to_array_keeps_arrays() -> None:
    profile = np.linspace(0.0, 1.0, 5)
    assert to_array(profile, "eV") is profile
    assert to_array(2.0, "eV") == 2.0
    temperatures = to_array([300, 600] * u.K, "eV")
    assert isinstance(temperatures, np.ndarray)
    assert temperatures == pytest.approx(np.array([300.0, 600.0]) / KELVIN_PER_EV)
    assert to_array("2 kV", "V") == pytest.approx(2000.0)
    assert to_array(["1 mm", 2 * u.mm], "m") == pytest.approx(np.array([1e-3, 2e-3]))
    with pytest.raises(ValueError, match="in units of m"):
        to_array(1 * u.s, "m")


def test_with_unit_and_describe() -> None:
    assert with_unit(1e-10, "s") == "1e-10 s"
    assert with_unit((0.0, 0.032), "m") == ["0 m", "0.032 m"]
    assert with_unit({"y_min": 0.1}, "eV") == {"y_min": "0.1 eV"}
    assert with_unit(math.inf, "s") == "inf s"
    assert with_unit(-math.inf, "s") == "-inf s"
    assert with_unit(None, "s") is None
    assert with_unit(2.0, "") == "2"
    assert describe((0.0, 0.032), "m") == "[0, 0.032] m"
    assert describe(1e-10, "s") == "1e-10 s"
    assert describe(None, "m") == "None"
    # Numbers are written so that they read back exactly.
    for value in (1 / 3, 2.4032649509999997e-19, 0.8000000000000002):
        assert to_unit(with_unit(value, "m"), "m") == value


def test_rate_to_si() -> None:
    assert rate_to_si(None) is None
    assert rate_to_si(2.0) == 2.0
    assert rate_to_si("1e18 1/s") == pytest.approx(1e18)
    assert rate_to_si(1e14 / (u.cm**2 * u.s)) == pytest.approx(1e18)
    with pytest.raises(ValueError, match="particles per second"):
        rate_to_si(1 * u.A)


@dataclasses.dataclass(frozen=True)
class _Gas:
    name: str
    density: float = unit_field("m^-3")
    temperature: float = unit_field("eV", 0.025)
    drift: tuple[float, ...] = unit_field("m/s", (0.0, 0.0, 0.0))
    profile: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        convert_units(self)
        object.__setattr__(self, "profile", tuple(self.profile))


def test_dataclass_fields_convert_and_round_trip() -> None:
    gas = _Gas(
        "Ar", density="1e21 m^-3", temperature=300 * u.K, drift=[1, 0, 0] * u.km / u.s
    )
    assert gas.density == pytest.approx(1e21)
    assert gas.temperature == pytest.approx(300.0 / KELVIN_PER_EV)
    assert gas.drift == pytest.approx((1000.0, 0.0, 0.0))
    params = dataclass_to_dict(gas)
    assert params["density"] == "1e+21 m^-3"
    assert params["drift"] == ["1000 m/s", "0 m/s", "0 m/s"]
    assert params["profile"] == []
    assert _Gas(**params) == gas
    with pytest.raises(ValueError, match="density must be in units of m\\^-3"):
        _Gas("Ar", density=1 * u.m)


def test_is_quantity() -> None:
    assert is_quantity(1 * u.m)
    assert not is_quantity(1.0)
    assert not is_quantity("1 m")


def test_numbers_do_not_import_astropy() -> None:
    """A bare number needs no astropy: the module imports it only for a quantity."""
    code = (
        "import sys\n"
        "from plasmacoll.units import convert_units, to_array, to_unit, with_unit\n"
        "assert to_unit([1, 2], 'm') == (1.0, 2.0)\n"
        "assert to_array(3.0, 'eV') == 3.0\n"
        "assert with_unit(1e-10, 's') == '1e-10 s'\n"
        "assert 'astropy' not in sys.modules, 'astropy was imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@dataclasses.dataclass(frozen=True)
class _Plasma:
    density: Any = dataclasses.field(metadata=unit_metadata("m^-3", array=True))
    name: str = "plasma"
    temperature: Any = unit_field("eV", 1.0, array=True)
    length: float = dataclasses.field(default=1.0, metadata=unit_metadata("m"))

    def __post_init__(self) -> None:
        convert_units(self)


def test_unit_metadata_and_array_fields() -> None:
    """A required field through ``unit_metadata``; array fields stay arrays."""
    density = np.array([1e17, 2e17])
    plasma = _Plasma(density, temperature=[300, 600] * u.K, length="2 mm")
    assert plasma.density is density
    assert isinstance(plasma.temperature, np.ndarray)
    assert plasma.temperature == pytest.approx(np.array([300.0, 600.0]) / KELVIN_PER_EV)
    assert plasma.length == pytest.approx(2e-3)
    assert _Plasma("1e11 cm-3").density == pytest.approx(1e17)
    assert unit_metadata("m") == {"unit": "m", "array": False}
    with pytest.raises(ValueError, match="density must be in units"):
        _Plasma(1 * u.s)


def test_to_array_on_mappings_and_nested_strings() -> None:
    profile = np.array([1.0, 2.0])
    fixed = to_array(
        {"e": profile, "Ar+": [1, 2] * u.cm**-3, "Ar": "1e21 m^-3"}, "m^-3"
    )
    assert fixed["e"] is profile
    assert fixed["Ar+"] == pytest.approx(np.array([1e6, 2e6]))
    assert fixed["Ar"] == pytest.approx(1e21)
    with pytest.raises(ValueError, match="as a string such as"):
        to_array({"e": {"value": 1.0, "unit": "m^-3"}}, "m^-3")
    # the configuration-file form of a 2D array reads back as an array
    positions = np.array([[0.0, 1e-3], [2e-3, 3e-3]])
    written = with_unit(positions, "m")
    assert written == [["0 m", "0.001 m"], ["0.002 m", "0.003 m"]]
    assert np.array_equal(to_array(written, "m"), positions)
    assert np.array_equal(to_array([["1 mm", 2e-3]], "m"), np.array([[1e-3, 2e-3]]))
    # a list of plain numbers is not converted
    plain = [[1.0, 2.0]]
    assert to_array(plain, "m") is plain
