"""Argon process sets: the LXCat reader's ion blocks, Phelps' Ar+ + Ar fits and the
electron-argon processes of a downloaded set.

The electron-argon file below is self-made in the layout of the SIGLO set on LXCat;
its numbers are synthetic, not physical data. The reference values of the Phelps fits
are those of the LXCat Phelps database (Ar+ + Ar backscatter) and of the fits as
published; the optional test against a downloaded SIGLO file quotes three points of
its ionization table (Rapp and Schram).
"""

from __future__ import annotations

import os
from pathlib import Path

import cunumpy as xp
import numpy as np
import pytest

from plasmacoll import (
    MonteCarloCollisions,
    NeutralBackground,
    argon_electron_processes,
    argon_ion_cross_sections,
    argon_ion_processes,
    find_lxcat_process,
    read_lxcat,
)
from plasmacoll.argon import (
    ARGON_IONIZATION_ENERGY,
    ARGON_MASS,
    phelps_argon_ion_backscatter,
    phelps_argon_ion_isotropic,
)
from plasmacoll.constants import ELECTRON_MASS

from ._helpers import speed_for

# Self-made, in the layout of the LXCat SIGLO argon set (EFFECTIVE, one lumped
# EXCITATION level, IONIZATION); the cross sections are synthetic.
ELECTRON_ARGON = """\
LXCat, www.lxcat.net
DATABASE:         synthetic test set

********************************************************** Ar **********************************************************

EFFECTIVE
Ar
 1.360000e-5
SPECIES: e / Ar
PROCESS: E + Ar -> E + Ar, Effective
PARAM.:  m/M = 0.0000136, complete set
COLUMNS: Energy (eV) | Cross section (m2)
-----------------------------
 0.000000e+0\t5.0e-20
 1.000000e+1\t5.0e-20
 1.000000e+2\t5.0e-20
-----------------------------

EXCITATION
Ar -> Ar*(11.5eV)
 1.150000e+1
SPECIES: e / Ar
PROCESS: E + Ar -> E + Ar*(11.5eV), Excitation
PARAM.:  E = 11.5 eV, complete set
COLUMNS: Energy (eV) | Cross section (m2)
-----------------------------
 1.150000e+1\t0.0
 1.000000e+2\t1.0e-20
-----------------------------

IONIZATION
Ar -> Ar^+
 1.576000e+1
SPECIES: e / Ar
PROCESS: E + Ar -> E + E + Ar+, Ionization
PARAM.:  E = 15.76 eV, complete set
COLUMNS: Energy (eV) | Cross section (m2)
-----------------------------
 1.576000e+1\t0.0
 2.000000e+1\t4.0e-21
 5.000000e+1\t2.0e-20
 1.000000e+2\t3.0e-20
-----------------------------
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""

ION_ENERGIES = (0.01, 1.0, 100.0)


def _ion_file() -> str:
    """Self-made in the layout of the LXCat Phelps Ar+ + Ar blocks, from the fits."""
    blocks = []
    for kind, function in (
        ("Backscat", phelps_argon_ion_backscatter),
        ("Isotropic", phelps_argon_ion_isotropic),
    ):
        rows = "\n".join(
            f" {e:.6e}\t{s:.6e}"
            for e, s in zip(
                ION_ENERGIES, function(np.asarray(ION_ENERGIES)), strict=True
            )
        )
        blocks.append(
            f"""\
SPECIES: Ar^+ / Ar
PROCESS: Ar+ + Ar -> , {kind}
PARAM.:  Mi = 39.948, Mi/M = 1, complete set
COMMENT: Ar+ in Ar: Phelps, J. Appl. Phys. 76, 747 (1994).
COLUMNS: Energy (eV) | Cross section (m2)
-----------------------------
{rows}
-----------------------------
"""
        )
    return "LXCat, www.lxcat.net\nDATABASE: synthetic test\n\n" + "\n".join(blocks)


@pytest.fixture
def electron_file(tmp_path: Path) -> Path:
    path = tmp_path / "Ar_electrons.txt"
    path.write_text(ELECTRON_ARGON)
    return path


@pytest.fixture
def ion_file(tmp_path: Path) -> Path:
    path = tmp_path / "Ar_ions.txt"
    path.write_text(_ion_file())
    return path


def test_argon_ionization_energy_is_15_76_ev() -> None:
    assert round(ARGON_IONIZATION_ENERGY, 2) == 15.76


def test_read_lxcat_reads_the_ion_blocks_without_keywords(ion_file: Path) -> None:
    processes = read_lxcat(ion_file)
    assert [p.kind for p in processes] == ["BACKSCAT", "ISOTROPIC"]
    backscatter = find_lxcat_process(processes, "BACKSCAT")
    assert backscatter.target == "Ar"
    assert backscatter.parameter == 1.0
    assert backscatter.threshold == 0.0
    assert backscatter.comments["SPECIES"] == "Ar^+ / Ar"
    assert backscatter.cross_section.label == "Ar+ + Ar -> , Backscat"


def test_read_lxcat_ion_block_without_param_line(tmp_path: Path) -> None:
    path = tmp_path / "ions.txt"
    path.write_text(
        "SPECIES: X^+ / X\nPROCESS: X+ + X -> , Isotropic\n-----\n 1.0 2.0e-19\n"
        " 2.0 1.0e-19\n-----\n"
    )
    (process,) = read_lxcat(path)
    assert process.kind == "ISOTROPIC"
    assert process.parameter == 0.0


def test_phelps_1994_backscatter_matches_the_lxcat_table() -> None:
    # LXCat Phelps database, Ar+ + Ar Backscat, at centre-of-mass energies (eV).
    reference = {0.01: 4.93367e-19, 1.0: 4.22679e-19, 100.0: 3.36634e-19}
    energy = np.asarray(list(reference))
    assert np.allclose(
        phelps_argon_ion_backscatter(energy), list(reference.values()), rtol=1e-5
    )


def test_phelps_1994_fits_follow_the_published_formulas() -> None:
    eps = 20.0  # laboratory energy, eV
    isotropic = 2e-19 / (eps**0.5 * (1 + eps)) + 3e-19 * eps / (1 + eps / 3) ** 2.3
    momentum = 1.15e-18 * eps**-0.1 * (1 + 0.015 / eps) ** 0.6
    q_i = phelps_argon_ion_isotropic(np.asarray([eps / 2]))
    q_b = phelps_argon_ion_backscatter(np.asarray([eps / 2]))
    assert np.allclose(q_i, isotropic)
    assert np.allclose(q_i + 2.0 * q_b, momentum)


def test_phelps_2005_fits_follow_the_published_formulas() -> None:
    assert np.allclose(phelps_argon_ion_isotropic(np.asarray([4.0]), "2005"), 10.8e-20)
    q_b = phelps_argon_ion_backscatter(np.asarray([1.0]), "2005")
    assert np.allclose(q_b, 52e-20 / 1.08 / 1.001**0.3)


@pytest.mark.parametrize("revision", ["1994", "2005"])
def test_ion_cross_sections_and_momentum_transfer_are_positive(revision: str) -> None:
    tables = argon_ion_cross_sections(revision)
    energy = xp.asarray(np.geomspace(1e-4, 1e4, 200))
    q_i = tables["isotropic"](energy)
    q_b = tables["backscatter"](energy)
    assert bool(xp.all(q_i > 0.0))
    assert bool(xp.all(q_b > 0.0))
    assert bool(xp.all(q_i + 2.0 * q_b > 0.0))


def test_phelps_fits_reject_bad_input() -> None:
    with pytest.raises(ValueError, match="revision"):
        argon_ion_cross_sections("1991")
    with pytest.raises(ValueError, match="energies > 0"):
        phelps_argon_ion_isotropic(np.asarray([0.0]))
    with pytest.raises(ValueError, match="min_energy"):
        argon_ion_cross_sections(min_energy=10.0, max_energy=1.0)


def test_argon_ion_processes_from_fits_lxcat_and_tables(ion_file: Path) -> None:
    fits = argon_ion_processes(neutral="Ar_fast")
    assert [p.kind for p in fits] == ["elastic", "backscatter"]
    assert all(p.energy_frame == "center_of_mass" for p in fits)
    assert fits[1].products == {"neutral": "Ar_fast"}
    from_file = argon_ion_processes(ion_file)
    from_list = argon_ion_processes(read_lxcat(ion_file))
    energy = xp.asarray(list(ION_ENERGIES))
    for a, b, c in zip(fits, from_file, from_list, strict=True):
        assert bool(xp.allclose(a.cross_section(energy), b.cross_section(energy)))
        assert bool(xp.allclose(b.cross_section(energy), c.cross_section(energy)))
    tables = argon_ion_cross_sections()
    assert argon_ion_processes(tables)[0].cross_section is tables["isotropic"]


def test_argon_ion_processes_need_both_tables(electron_file: Path) -> None:
    with pytest.raises(KeyError, match="backscatter"):
        argon_ion_processes({"isotropic": argon_ion_cross_sections()["isotropic"]})
    with pytest.raises(KeyError, match="ISOTROPIC"):
        argon_ion_processes(electron_file)


def test_argon_electron_processes_from_a_downloaded_set(electron_file: Path) -> None:
    processes = argon_electron_processes(electron_file)
    assert [p.kind for p in processes] == ["elastic", "excitation", "ionization"]
    assert all(p.energy_frame == "lab" for p in processes)
    elastic, excitation, ionization = processes
    # units: energies in eV and cross sections in m^2, as in the file
    assert ionization.cross_section.threshold == 15.76
    assert ionization.loss == 15.76
    assert ionization.products == {"electron": "e", "ion": "Ar+"}
    assert ionization.energy_sharing == "opal"
    assert bool(
        xp.allclose(
            ionization.cross_section(xp.asarray([15.0, 20.0, 50.0, 100.0])),
            [0.0, 4.0e-21, 2.0e-20, 3.0e-20],
        )
    )
    assert excitation.loss == 11.5
    # the effective cross section less the inelastic ones
    energy = xp.asarray([1.0, 30.0, 100.0])
    expected = (
        5.0e-20 - excitation.cross_section(energy) - ionization.cross_section(energy)
    )
    assert bool(xp.allclose(elastic.cross_section(energy), expected))
    assert bool(xp.all(elastic.cross_section(xp.asarray(np.geomspace(1e-3, 1e3))) > 0))


def test_argon_electron_processes_need_an_ionization_block(tmp_path: Path) -> None:
    path = tmp_path / "no_ionization.txt"
    path.write_text(ELECTRON_ARGON.split("IONIZATION")[0])
    with pytest.raises(KeyError, match="IONIZATION"):
        argon_electron_processes(path)


def test_argon_processes_run_in_monte_carlo_collisions(electron_file: Path) -> None:
    mcc = MonteCarloCollisions(
        species_masses={"e": ELECTRON_MASS, "Ar+": ARGON_MASS},
        backgrounds=[NeutralBackground("Ar", 1.0e21, 0.026, ARGON_MASS)],
        processes={
            "e": argon_electron_processes(electron_file),
            "Ar+": argon_ion_processes(),
        },
        seed=3,
    )
    num = 2000
    velocities = xp.zeros((num, 3))
    velocities[:, 0] = speed_for(60.0, ELECTRON_MASS)
    result = mcc.collide("e", xp.zeros((num, 1)), velocities, xp.ones(num), dt=1e-9)
    assert result.diagnostics.counts["ionization"] > 0
    assert "Ar+" in result.created
    velocities[:, 0] = speed_for(5.0, ARGON_MASS)
    result = mcc.collide("Ar+", xp.zeros((num, 1)), velocities, xp.ones(num), dt=2e-8)
    assert result.diagnostics.counts["ar_ion_backscatter"] > 0


SIGLO_FILE = os.environ.get("PLASMACOLL_LXCAT_SIGLO_AR")


@pytest.mark.skipif(
    not SIGLO_FILE, reason="set PLASMACOLL_LXCAT_SIGLO_AR to a SIGLO argon file"
)
def test_downloaded_siglo_set_matches_its_source() -> None:  # pragma: no cover
    processes = argon_electron_processes(str(SIGLO_FILE))
    ionization = processes[-1]
    # SIGLO's Ar ionization (Rapp and Schram) starts at 15.8 eV
    assert ionization.cross_section.threshold == pytest.approx(15.8)
    assert bool(
        xp.allclose(
            ionization.cross_section(xp.asarray([20.0, 50.0, 100.0])),
            [6.30e-21, 2.53e-20, 2.85e-20],
        )
    )
    energy = xp.asarray(np.geomspace(1e-3, 1e4, 300))
    assert bool(xp.all(processes[0].cross_section(energy) > 0.0))
