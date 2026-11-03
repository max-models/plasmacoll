"""Caesium in a hydrogen plasma: ionization by background electrons, Cs and Cs+ on H2."""

from __future__ import annotations

import math

import cunumpy as xp
import numpy as np
import pytest

from plasmacoll import (
    FunctionProfile,
    NeutralBackground,
    caesium_collisions,
    caesium_ionization_cross_section,
    electron_background,
    rate_coefficient,
)
from plasmacoll.caesium import (
    CAESIUM_IONIZATION_ENERGY,
    CAESIUM_MASS,
    CS_H2_HARD_SPHERE_CROSS_SECTION,
    H2_MASS,
    H2_POLARIZABILITY,
    caesium_ion_h2_elastic_process,
    caesium_ionization_process,
    langevin_rate_coefficient,
)
from plasmacoll.constants import ELECTRON_MASS


def test_ionization_cross_section_starts_at_the_ionization_energy() -> None:
    sigma = caesium_ionization_cross_section()
    assert sigma.threshold == CAESIUM_IONIZATION_ENERGY
    assert float(sigma(xp.asarray([3.0]))[0]) == 0.0


def test_ionization_cross_section_peaks_at_e_times_the_threshold() -> None:
    sigma = caesium_ionization_cross_section(num=2000)
    energy = np.geomspace(4.0, 100.0, 2000)
    values = xp.to_numpy(sigma(xp.asarray(energy)))
    peak = energy[np.argmax(values)]
    assert peak == pytest.approx(math.e * CAESIUM_IONIZATION_ENERGY, rel=0.01)
    assert values.max() == pytest.approx(
        4.5e-18 / (math.e * CAESIUM_IONIZATION_ENERGY**2), rel=1e-3
    )


def test_ionization_process_turns_cs_into_cs_plus() -> None:
    process = caesium_ionization_process()
    assert process.kind == "stripping"
    assert process.products == {"projectile": "Cs+"}
    assert process.charge_change == 1


def test_electron_background_has_the_electron_mass_and_profiles() -> None:
    profile = FunctionProfile(lambda x: x[:, 0], 1.0)
    electrons = electron_background(1e17, 2.0, profile, profile)
    assert electrons.mass == ELECTRON_MASS
    assert electrons.name == "e"
    assert electrons.profile is profile
    assert electrons.temperature_profile is profile


def test_langevin_rate_is_of_order_1e_15() -> None:
    mu = CAESIUM_MASS * H2_MASS / (CAESIUM_MASS + H2_MASS)
    rate = langevin_rate_coefficient(H2_POLARIZABILITY, mu)
    assert 1e-15 < rate < 3e-15
    with pytest.raises(ValueError):
        langevin_rate_coefficient(0.0, mu)


def test_cs_plus_h2_frequency_is_speed_independent() -> None:
    process = caesium_ion_h2_elastic_process()
    sigma = process.cross_section
    mu = CAESIUM_MASS * H2_MASS / (CAESIUM_MASS + H2_MASS)
    energy = xp.asarray([0.01, 1.0, 100.0])
    speed = xp.sqrt(2.0 * energy * 1.602176634e-19 / mu)
    products = xp.to_numpy(sigma(energy) * speed)
    assert np.allclose(products, products[0], rtol=1e-6)


def test_hard_sphere_cross_section_is_geometric() -> None:
    assert CS_H2_HARD_SPHERE_CROSS_SECTION == pytest.approx(7.47e-19, rel=0.01)


def _uniform(gas_density: float = 1e19):
    electrons = electron_background(1e17, 2.0)
    gas = NeutralBackground("H2", gas_density, 0.03, H2_MASS)
    return electrons, gas


def test_cs_markers_ionize_at_n_e_times_the_rate_coefficient() -> None:
    electrons, gas = _uniform(gas_density=0.0)
    mcc = caesium_collisions(electrons, gas, seed=1)
    num = 1_000_000
    positions = xp.zeros((num, 1))
    velocities = xp.zeros((num, 3))
    weights = xp.full(num, 3.0)
    rate = float(
        rate_coefficient(caesium_ionization_cross_section(), 2.0, ELECTRON_MASS)
    )
    dt = 0.003 / (electrons.density * rate)
    result = mcc.collide("Cs", positions, velocities, weights, dt)
    ionized = int(xp.to_numpy(result.removed).sum())
    expected = num * (1.0 - math.exp(-electrons.density * rate * dt))
    assert ionized == pytest.approx(expected, rel=0.06)
    ions = result.created["Cs+"]
    assert sum(int(new.weights.shape[0]) for new in ions) == ionized
    assert all(np.all(xp.to_numpy(new.weights) == 3.0) for new in ions)


def test_the_gas_bound_does_not_take_the_electron_speeds() -> None:
    electrons, gas = _uniform()
    mcc = caesium_collisions(electrons, gas, seed=4)
    num = 1000
    result = mcc.collide(
        "Cs", xp.zeros((num, 1)), xp.zeros((num, 3)), xp.ones(num), 1e-7
    )
    # With the electrons' relative speed for H2 as well, nu_max dt would be ~4.
    assert result.diagnostics.collision_probability < 0.05


def test_ionization_follows_the_electron_density_profile() -> None:
    profile = FunctionProfile(lambda x: (x[:, 0] > 0.5).astype(float), 1.0)
    electrons = electron_background(1e17, 2.0, density_profile=profile)
    gas = NeutralBackground("H2", 0.0, 0.03, H2_MASS)
    mcc = caesium_collisions(electrons, gas, seed=2)
    num = 20_000
    positions = xp.asarray(np.repeat([[0.0], [1.0]], num // 2, axis=0))
    rate = float(
        rate_coefficient(caesium_ionization_cross_section(), 2.0, ELECTRON_MASS)
    )
    result = mcc.collide(
        "Cs", positions, xp.zeros((num, 3)), xp.ones(num), 0.05 / (1e17 * rate)
    )
    removed = xp.to_numpy(result.removed)
    assert removed[: num // 2].sum() == 0
    assert removed[num // 2 :].sum() > 0


def test_cs_plus_scatters_on_h2() -> None:
    electrons, gas = _uniform()
    mcc = caesium_collisions(electrons, gas, seed=3)
    num = 10_000
    velocities = xp.zeros((num, 3))
    velocities[:, 0] = 2000.0
    result = mcc.collide("Cs+", xp.zeros((num, 1)), velocities, xp.ones(num), 1e-5)
    assert not bool(xp.to_numpy(result.removed).any())
    assert result.diagnostics.real_collisions > 0
