"""The proton-hydrogen process set; the cross sections are synthetic constants."""

from __future__ import annotations

import cunumpy as xp
import pytest

from pymcc import (
    CrossSection,
    MonteCarloCollisions,
    NeutralBackground,
    proton_hydrogen_processes,
)
from pymcc.hydrogen import PROTON_HYDROGEN_PROCESS_KEYS

from ._helpers import M_P, speed_for


def _tables() -> dict[str, CrossSection]:
    return {
        key: CrossSection.constant(1.0e-19, threshold=0.1)
        for key in PROTON_HYDROGEN_PROCESS_KEYS
    }


def test_proton_hydrogen_processes_follow_the_model() -> None:
    processes = {p.name: p for p in proton_hydrogen_processes(_tables(), h2_ion="H2+")}
    assert list(processes) == list(PROTON_HYDROGEN_PROCESS_KEYS)
    assert processes["mt_h"].energy_frame == "center_of_mass"
    assert all(
        processes[key].energy_frame == "lab" for key in PROTON_HYDROGEN_PROCESS_KEYS[1:]
    )
    assert processes["cx_h"].kind == "backscatter"
    assert processes["cx_h"].background == "H"
    assert processes["cx_h2"].kind == "charge_transfer"
    assert processes["cx_h2"].products == {"ion": "H2+"}
    assert processes["rot_h2"].loss == pytest.approx(0.0453)
    assert processes["vib_h2"].loss == pytest.approx(0.516)
    assert processes["mt_h2"].loss == 0.0
    # the thresholds of the tables are replaced
    assert processes["mt_h"].cross_section.threshold == 0.0
    assert processes["vib_h2"].cross_section.threshold == pytest.approx(0.516)


def test_proton_hydrogen_processes_name_the_backgrounds() -> None:
    processes = proton_hydrogen_processes(_tables(), atom="atoms", molecule="molecules")
    assert {p.background for p in processes} == {"atoms", "molecules"}
    assert next(p for p in processes if p.name == "cx_h2").products == {}


def test_proton_hydrogen_processes_need_every_table() -> None:
    tables = _tables()
    del tables["vib_h2"]
    with pytest.raises(KeyError, match="vib_h2"):
        proton_hydrogen_processes(tables)


def test_proton_hydrogen_processes_run_together() -> None:
    """All six processes act on one population."""
    mcc = MonteCarloCollisions(
        species_masses={"H+": M_P},
        backgrounds=[
            NeutralBackground("H", 1.0e20, 300.0, M_P),
            NeutralBackground("H2", 1.0e20, 300.0, 2 * M_P),
        ],
        processes={"H+": proton_hydrogen_processes(_tables())},
        seed=5,
    )
    num = 20_000
    velocities = xp.zeros((num, 3))
    velocities[:, 0] = speed_for(5.0, M_P)
    result = mcc.collide("H+", xp.zeros((num, 1)), velocities, xp.ones(num), dt=2.0e-6)
    counts = result.diagnostics.counts
    assert set(counts) == set(PROTON_HYDROGEN_PROCESS_KEYS)
    assert all(count > 0 for count in counts.values())
