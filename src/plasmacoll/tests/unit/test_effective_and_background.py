"""LXCat effective cross sections, and flowing, heated and profiled backgrounds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cunumpy as xp
import numpy as np
import pytest

from plasmacoll import (
    CollisionProcess,
    CrossSection,
    FunctionProfile,
    MonteCarloCollisions,
    NeutralBackground,
    ParticleArrays,
    Profile,
    elastic_from_lxcat,
    make_rng,
    read_lxcat,
)
from plasmacoll.constants import BOLTZMANN

from ._helpers import M_HE
from .test_cross_sections import LXCAT_EXAMPLE


def _host(array) -> np.ndarray:
    return np.asarray(xp.to_numpy(array))


# --------------------------------------------------------------------- #
# Effective cross sections
# --------------------------------------------------------------------- #
def test_elastic_from_effective_subtracts_the_inelastic_cross_sections() -> None:
    effective = CrossSection.constant(1.0e-19, label="eff")
    excitation = CrossSection(
        xp.asarray([10.0, 20.0]), xp.asarray([0.0, 2.0e-20]), 10.0
    )
    ionization = CrossSection.constant(1.0e-20, threshold=15.0)
    elastic = CrossSection.elastic_from_effective(effective, [excitation, ionization])
    energies = xp.asarray([5.0, 9.99, 15.0, 20.0, 100.0])
    expected = [1.0e-19, 1.0e-19, 8.0e-20, 7.0e-20, 7.0e-20]
    np.testing.assert_allclose(_host(elastic(energies)), expected, rtol=1e-6)
    assert elastic.label == "elastic from eff"


def test_elastic_from_effective_warns_for_an_inconsistent_set() -> None:
    effective = CrossSection.constant(1.0e-20)
    with pytest.warns(RuntimeWarning, match="clipped"):
        elastic = CrossSection.elastic_from_effective(
            effective, [CrossSection.constant(2.0e-20)], label="x"
        )
    assert float(xp.max(elastic.sigma)) == 0.0


def test_elastic_from_lxcat(tmp_path: Path) -> None:
    path = tmp_path / "gas.txt"
    path.write_text(LXCAT_EXAMPLE)
    processes = read_lxcat(path)
    # The set has an ELASTIC block, which is returned as is.
    assert elastic_from_lxcat(processes) is processes[0].cross_section
    # Without it, the EFFECTIVE block minus the inelastic ones. The synthetic set
    # is inconsistent above the ionization threshold (20 eV), so check below it.
    effective_only = [p for p in processes if p.kind != "ELASTIC"]
    with pytest.warns(RuntimeWarning, match="clipped"):
        elastic = elastic_from_lxcat(effective_only, target="Gas")
    energy = xp.asarray([18.0])
    inelastic = sum(
        _host(p.cross_section(energy))[0]
        for p in effective_only
        if p.kind in ("EXCITATION", "IONIZATION", "ATTACHMENT")
    )
    assert inelastic > 0.0
    assert _host(elastic(energy))[0] == pytest.approx(1.0e-20 - inelastic, rel=1e-9)
    with pytest.raises(KeyError, match="No ELASTIC or EFFECTIVE"):
        elastic_from_lxcat(effective_only, target="Ar")


# --------------------------------------------------------------------- #
# Backgrounds
# --------------------------------------------------------------------- #
def test_drifting_background_samples_its_flow_velocity() -> None:
    gas = NeutralBackground("gas", 1.0e20, 300.0, M_HE, drift=(1000.0, 0.0, -500.0))
    assert gas.drift_speed == pytest.approx(np.hypot(1000.0, 500.0))
    velocities = _host(gas.sample_velocities(make_rng(1), 200_000))
    np.testing.assert_allclose(
        velocities.mean(axis=0), [1000.0, 0.0, -500.0], atol=10.0
    )
    two_components: Any = (1.0, 2.0)
    with pytest.raises(ValueError, match="three components"):
        NeutralBackground("gas", 1.0e20, 300.0, M_HE, drift=two_components)


def test_temperature_profile_scales_the_sampled_temperature() -> None:
    hot_right = FunctionProfile(
        lambda x: xp.where(x[:, 0] > 0.0, 4.0, 1.0), max_factor=4.0
    )
    assert isinstance(hot_right, Profile)
    gas = NeutralBackground("gas", 1.0e20, 300.0, M_HE, temperature_profile=hot_right)
    assert gas.max_thermal_speed == pytest.approx(2.0 * gas.thermal_speed)
    num = 200_000
    positions = xp.asarray(np.repeat([[-1.0], [1.0]], num // 2, axis=0))
    velocities = _host(gas.sample_velocities(make_rng(2), num, positions))
    cold = M_HE * velocities[: num // 2].var(axis=0).mean() / BOLTZMANN
    hot = M_HE * velocities[num // 2 :].var(axis=0).mean() / BOLTZMANN
    assert cold == pytest.approx(300.0, rel=0.02)
    assert hot == pytest.approx(1200.0, rel=0.02)
    with pytest.raises(ValueError, match="positions"):
        gas.sample_velocities(make_rng(2), 10)


def test_function_profile_validates_its_bound() -> None:
    with pytest.raises(ValueError, match="max_factor"):
        FunctionProfile(lambda x: x[:, 0], max_factor=-1.0)


def test_ions_relax_to_a_flowing_heated_gas() -> None:
    """Ions colliding elastically with a gas end up at its flow velocity and temperature."""
    gas = NeutralBackground(
        "gas",
        1.0e21,
        300.0,
        M_HE,
        drift=(2000.0, 0.0, 0.0),
        temperature_profile=FunctionProfile(lambda x: xp.full(x.shape[0], 2.0), 2.0),
        profile=FunctionProfile(lambda x: xp.full(x.shape[0], 0.5), 1.0),
    )
    rate = 1.0e-15
    mcc = MonteCarloCollisions(
        {"ion": M_HE},
        [gas],
        {
            "ion": [
                CollisionProcess(
                    "elastic", "gas", CrossSection.maxwell_molecule(rate, M_HE / 2)
                )
            ]
        },
        seed=3,
    )
    ions = ParticleArrays.maxwellian(20_000, M_HE, 300.0, seed=4)
    nu = 0.5 * 1.0e21 * rate
    for _ in range(400):
        mcc.collide_species({"ion": ions}, 0.05 / nu)
    mean = _host(xp.sum(ions.velocities, axis=0)) / len(ions)
    assert mean[0] == pytest.approx(2000.0, rel=0.05)
    assert ions.temperature(M_HE) == pytest.approx(600.0, rel=0.05)
