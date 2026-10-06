"""Tabulated cross sections and the LXCat reader."""

from __future__ import annotations

import math
from pathlib import Path

import cunumpy as xp
import numpy as np
import pytest

from plasmacoll import CrossSection, find_lxcat_process, read_lxcat
from plasmacoll.constants import ELECTRON_MASS, ELEMENTARY_CHARGE

# A synthetic file in LXCat format; the numbers are not physical data.
LXCAT_EXAMPLE = """\
Some database header text that the reader must skip.
DATABASE: synthetic

ELASTIC
Gas
 1.0e-4
SPECIES: e / Gas
PROCESS: E + Gas -> E + Gas, Elastic
COLUMNS: Energy (eV) | Cross section (m2)
-----------------------------
 0.0e+0\t1.0e-20
 1.0e+1\t2.0e-20
-----------------------------

EXCITATION
Gas -> Gas*
 1.5e+1  1.0
PROCESS: E + Gas -> E + Gas*, Excitation
-----------------------------
 1.5e+1\t0.0
 3.0e+1\t4.0e-21
-----------------------------

IONIZATION
Gas -> Gas^+
 2.0e+1
-----------------------------
 2.0e+1\t0.0
 1.0e+2\t3.0e-20
-----------------------------

ATTACHMENT
Gas -> Gas^-
-----------------------------
 0.0\t5.0e-22
 5.0\t0.0
-----------------------------

EFFECTIVE
Gas <-> Gas
 2.0e-4
-----------------------------
 0.0\t1.0e-20
 1.0\t1.0e-20
-----------------------------
xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""


def test_cross_section_interpolates_linearly_and_extends_last_value() -> None:
    table = CrossSection(energy=[0.0, 10.0, 20.0], sigma=[1.0, 3.0, 2.0])
    assert bool(xp.allclose(table(xp.asarray([5.0, 15.0, 1000.0])), [2.0, 2.5, 2.0]))
    assert table.max_energy == 20.0


def test_cross_section_is_zero_below_threshold() -> None:
    table = CrossSection(energy=[0.0, 30.0], sigma=[1.0, 1.0], threshold=10.0)
    assert bool(xp.allclose(table(xp.asarray([9.99, 10.0, 20.0])), [0.0, 1.0, 1.0]))


@pytest.mark.parametrize(
    ("energy", "sigma", "threshold", "message"),
    [
        ([0.0], [1.0], 0.0, "at least two"),
        ([1.0, 0.0], [1.0, 1.0], 0.0, "non-decreasing"),
        ([0.0, 1.0], [1.0, -1.0], 0.0, "non-negative"),
        ([0.0, 1.0], [1.0], 0.0, "equal length"),
        ([0.0, 1.0], [1.0, 1.0], -1.0, "threshold"),
    ],
)
def test_cross_section_rejects_invalid_tables(
    energy: list[float], sigma: list[float], threshold: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CrossSection(energy=energy, sigma=sigma, threshold=threshold)


def test_constant_cross_section() -> None:
    table = CrossSection.constant(2.0e-19, threshold=3.0)
    assert float(table(1.0)) == 0.0
    assert float(table(5.0)) == pytest.approx(2.0e-19)
    assert float(table(1.0e9)) == pytest.approx(2.0e-19)
    assert "constant" in table.label


def test_maxwell_molecule_has_speed_independent_frequency() -> None:
    rate = 3.0e-15
    table = CrossSection.maxwell_molecule(rate, ELECTRON_MASS)
    energies = np.geomspace(1.0e-3, 1.0e4, 101)
    speeds = np.sqrt(2.0 * energies * ELEMENTARY_CHARGE / ELECTRON_MASS)
    products = xp.to_numpy(table(xp.asarray(energies))) * speeds
    assert np.allclose(products, rate, rtol=1e-4)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rate_coefficient": -1.0, "mass": 1.0}, "rate_coefficient"),
        ({"rate_coefficient": 1.0, "mass": 0.0}, "mass"),
        (
            {
                "rate_coefficient": 1.0,
                "mass": 1.0,
                "min_energy": 2.0,
                "max_energy": 1.0,
            },
            "min_energy",
        ),
    ],
)
def test_maxwell_molecule_rejects_invalid_arguments(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CrossSection.maxwell_molecule(**kwargs)


def test_from_function_samples_on_the_grid() -> None:
    table = CrossSection.from_function(
        lambda energy: 1.0e-20 * np.sqrt(energy), np.linspace(0.0, 4.0, 5), label="sqrt"
    )
    assert bool(xp.allclose(table.sigma, 1.0e-20 * xp.sqrt(xp.linspace(0.0, 4.0, 5))))
    assert float(table(2.5)) == pytest.approx(
        1.0e-20 * (math.sqrt(2) + math.sqrt(3)) / 2
    )
    assert table.label == "sqrt"


def test_with_threshold_keeps_the_table() -> None:
    table = CrossSection(energy=[0.0, 10.0], sigma=[1.0, 1.0], label="t")
    shifted = table.with_threshold(4.0)
    assert shifted.threshold == 4.0
    assert shifted.label == "t"
    assert float(shifted(3.0)) == 0.0
    assert float(table(3.0)) == 1.0


def test_from_table_reads_semicolon_separated_file(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    path.write_text("# comment\n\nenergy;sigma\n0.0;1.0e-20\n10.0;3.0e-20\n")
    table = CrossSection.from_table(path, delimiter=";")
    assert bool(xp.allclose(table.energy, [0.0, 10.0]))
    assert float(table(5.0)) == pytest.approx(2.0e-20)
    assert table.label == "table"


def test_from_table_converts_units(tmp_path: Path) -> None:
    path = tmp_path / "cm2.dat"
    path.write_text("1.0 2.0e-16\n2.0 4.0e-16\n")
    table = CrossSection.from_table(
        path, energy_unit=1.0e3, sigma_unit=1.0e-4, threshold=500.0, label="keV, cm^2"
    )
    assert bool(xp.allclose(table.energy, [1.0e3, 2.0e3]))
    assert bool(xp.allclose(table.sigma, [2.0e-20, 4.0e-20]))
    assert table.threshold == 500.0
    assert table.label == "keV, cm^2"


def test_read_lxcat_parses_all_process_kinds(tmp_path: Path) -> None:
    path = tmp_path / "gas.txt"
    path.write_text(LXCAT_EXAMPLE)

    processes = read_lxcat(path)

    assert [process.kind for process in processes] == [
        "ELASTIC",
        "EXCITATION",
        "IONIZATION",
        "ATTACHMENT",
        "EFFECTIVE",
    ]
    elastic, excitation, ionization, attachment, effective = processes
    assert elastic.mass_ratio == pytest.approx(1.0e-4)
    assert elastic.threshold == 0.0
    assert elastic.target == "Gas"
    assert elastic.product == ""
    assert elastic.comments["PROCESS"] == "E + Gas -> E + Gas, Elastic"
    assert elastic.cross_section.label == "E + Gas -> E + Gas, Elastic"
    assert float(elastic.cross_section(5.0)) == pytest.approx(1.5e-20)
    assert excitation.target == "Gas"
    assert excitation.product == "Gas*"
    assert excitation.threshold == pytest.approx(15.0)
    assert excitation.mass_ratio is None
    assert ionization.threshold == pytest.approx(20.0)
    assert ionization.cross_section.label == "IONIZATION Gas -> Gas^+"
    assert float(ionization.cross_section(60.0)) == pytest.approx(1.5e-20)
    assert attachment.parameter == 0.0
    assert float(attachment.cross_section(2.5)) == pytest.approx(2.5e-22)
    assert effective.mass_ratio == pytest.approx(2.0e-4)
    assert effective.product == "Gas"


def test_find_lxcat_process_by_kind_or_process_comment(tmp_path: Path) -> None:
    path = tmp_path / "gas.txt"
    path.write_text(LXCAT_EXAMPLE)
    processes = read_lxcat(path)

    assert find_lxcat_process(processes, "IONIZATION").threshold == pytest.approx(20.0)
    found = find_lxcat_process(processes, "E + Gas -> E + Gas*, Excitation")
    assert found.kind == "EXCITATION"
    with pytest.raises(KeyError, match="DETACHMENT"):
        find_lxcat_process(processes, "DETACHMENT")
