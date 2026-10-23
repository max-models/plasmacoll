"""The Lotz formula and the metal ionization cross sections built on it."""

from __future__ import annotations

import math

import cunumpy as xp
import numpy as np
import pytest

from plasmacoll import LotzSubshell, lotz_cross_section, metal_ionization_cross_section
from plasmacoll.metals import LOTZ_CONSTANT, METAL_OUTER_SUBSHELLS, lotz_table


def test_lotz_constant_is_4_5e_14_cm2_ev2() -> None:
    assert LOTZ_CONSTANT == pytest.approx(4.5e-14 * 1e-4)


def test_lotz_cross_section_follows_the_formula() -> None:
    shell = LotzSubshell(binding_energy=10.0, electrons=2, b=0.5, c=0.3)
    energy = 40.0
    expected = (
        LOTZ_CONSTANT
        * 2
        * math.log(4.0)
        / (40.0 * 10.0)
        * (1.0 - 0.5 * math.exp(-0.3 * 3.0))
    )
    assert lotz_cross_section(energy, [shell])[0] == pytest.approx(expected)


def test_lotz_cross_section_is_zero_below_threshold_and_adds_subshells() -> None:
    outer = LotzSubshell(7.0, 1)
    inner = LotzSubshell(20.0, 10)
    energy = np.asarray([5.0, 7.0, 15.0, 50.0])
    both = lotz_cross_section(energy, [outer, inner])
    assert both[0] == 0.0 and both[1] == 0.0
    assert both[2] == lotz_cross_section(15.0, [outer])[0]
    assert both[3] == pytest.approx(
        lotz_cross_section(50.0, [outer])[0] + lotz_cross_section(50.0, [inner])[0]
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"binding_energy": 0.0, "electrons": 1},
        {"binding_energy": 1.0, "electrons": 0},
        {"binding_energy": 1.0, "electrons": 1, "b": 1.5},
    ],
)
def test_lotz_subshell_rejects_bad_parameters(kwargs) -> None:
    with pytest.raises(ValueError):
        LotzSubshell(**kwargs)


def test_lotz_table_starts_at_the_lowest_binding_energy() -> None:
    table = lotz_table([LotzSubshell(20.0, 10), LotzSubshell(7.0, 1)], num=50)
    assert table.threshold == 7.0
    assert float(table.energy[0]) == 7.0
    assert bool(xp.any(table.energy == 20.0))
    assert float(table(xp.asarray([6.9]))[0]) == 0.0
    with pytest.raises(ValueError, match="empty"):
        lotz_table([])
    with pytest.raises(ValueError, match="max_energy"):
        lotz_table([LotzSubshell(20.0, 1)], max_energy=10.0)


@pytest.mark.parametrize(
    ("element", "charge", "threshold"),
    [("Cu", 0, 7.726380), ("Cu", 1, 20.29239), ("Ti", 0, 6.828120), ("Ti", 1, 13.5755)],
)
def test_metal_ionization_starts_at_the_nist_ionization_energy(
    element: str, charge: int, threshold: float
) -> None:
    table = metal_ionization_cross_section(element, charge)
    assert table.threshold == threshold
    _, electrons, _ = METAL_OUTER_SUBSHELLS[(element, charge)]
    energy = 3.0 * threshold
    assert float(table(xp.asarray([energy]))[0]) == pytest.approx(
        LOTZ_CONSTANT * electrons * math.log(3.0) / (energy * threshold), rel=1e-3
    )
    assert bool(xp.all(table.sigma >= 0.0))


def test_metal_ionization_needs_a_known_element() -> None:
    with pytest.raises(KeyError, match="Al"):
        metal_ionization_cross_section("Al")
