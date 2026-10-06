"""Neutral backgrounds, density profiles and process definitions."""

from __future__ import annotations

import math

import cunumpy as xp
import pytest

from pymcc import CollisionProcess, DensityProfile, NeutralBackground, make_rng
from pymcc.constants import BOLTZMANN

from ._helpers import M_P, constant


def test_density_profile_interpolates_and_bounds() -> None:
    profile = DensityProfile(axis=1, positions=[0.0, 1.0, 2.0], factors=[0.0, 1.0, 0.5])
    positions = xp.asarray([[9.0, -1.0], [9.0, 0.5], [9.0, 1.5], [9.0, 5.0]])
    assert bool(xp.allclose(profile(positions), [0.0, 0.5, 0.75, 0.5]))
    assert profile.max_factor == 1.0
    background = NeutralBackground("H", 1.0e19, 300.0, M_P, profile=profile)
    assert background.max_density_factor == 1.0
    assert bool(xp.allclose(background.density_factor(positions), profile(positions)))
    uniform = NeutralBackground("H", 1.0e19, 300.0, M_P)
    assert uniform.max_density_factor == 1.0
    assert uniform.density_factor(positions) == 1.0


@pytest.mark.parametrize(
    ("axis", "positions", "factors", "message"),
    [
        (0, [0.0], [1.0], "two points"),
        (0, [0.0, 0.0], [1.0, 1.0], "increasing"),
        (0, [0.0, 1.0], [1.0, -0.1], ">= 0"),
        (0, [0.0, 1.0], [1.0], "equal length"),
        (-1, [0.0, 1.0], [1.0, 1.0], "axis"),
    ],
)
def test_density_profile_rejects_bad_tables(
    axis: int, positions: list, factors: list, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DensityProfile(axis=axis, positions=positions, factors=factors)


@pytest.mark.parametrize(
    ("density", "temperature", "mass", "message"),
    [
        (-1.0, 300.0, 1.0, "density"),
        (1.0, -1.0, 1.0, "temperature"),
        (1.0, 300.0, 0.0, "mass"),
    ],
)
def test_background_rejects_unphysical_parameters(
    density: float, temperature: float, mass: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        NeutralBackground("gas", density, temperature, mass)


def test_background_samples_its_maxwellian() -> None:
    background = NeutralBackground("H", 1.0e19, 1000.0, M_P)
    assert background.thermal_speed == pytest.approx(
        math.sqrt(BOLTZMANN * 1000.0 / M_P)
    )
    velocities = background.sample_velocities(make_rng(0), 200_000)
    assert velocities.shape == (200_000, 3)
    spread = xp.std(velocities, axis=0) / background.thermal_speed
    assert bool(xp.allclose(spread, 1.0, atol=0.01))
    assert (
        float(xp.max(xp.abs(xp.mean(velocities, axis=0))))
        < 0.01 * background.thermal_speed
    )


def test_process_name_and_energy_loss_defaults() -> None:
    excitation = CollisionProcess("excitation", "Ar", constant(1e-20, threshold=11.5))
    assert excitation.name == "excitation:Ar"
    assert excitation.loss == 11.5
    lossy = CollisionProcess(
        "excitation", "Ar", constant(1e-20, 11.5), energy_loss=12.0
    )
    assert lossy.loss == 12.0
    elastic = CollisionProcess("elastic", "Ar", constant(1e-20, 11.5), energy_loss=12.0)
    assert elastic.loss == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "recombination"}, "Unknown process kind"),
        ({"kind": "elastic", "energy_frame": "ion"}, "energy_frame"),
        ({"kind": "elastic", "energy_sharing": "random"}, "energy_sharing"),
        ({"kind": "ionization"}, "needs product species"),
        ({"kind": "charge_transfer", "products": {"electron": "e"}}, "no product role"),
        ({"kind": "excitation", "energy_loss": -1.0}, "energy_loss"),
    ],
)
def test_process_rejects_invalid_definitions(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CollisionProcess(background="gas", cross_section=constant(1e-20), **kwargs)


def test_charge_transfer_may_name_an_ion() -> None:
    process = CollisionProcess(
        "charge_transfer", "H2", constant(1e-19), products={"ion": "H2+"}
    )
    assert process.products == {"ion": "H2+"}
    assert CollisionProcess("charge_transfer", "H2", constant(1e-19)).products == {}
