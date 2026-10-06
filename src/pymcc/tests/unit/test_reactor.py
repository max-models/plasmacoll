"""The 0D reactor against closed-form solutions for Maxwell-molecule cross sections.

With ``sigma = K / g`` every marker collides at the constant frequency
``nu = n K``. The null-collision method then collides a marker in one step with
probability ``p = (1 - exp(-s nu dt)) / s`` (``s`` is the bound safety factor)
and at most once per step, so counts follow ``(1 - p)**steps`` and
``(1 + p)**steps`` exactly in expectation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pymcc import (
    CollisionProcess,
    CrossSection,
    MonteCarloCollisions,
    NeutralBackground,
    ParticleArrays,
    ReactorHistory,
    ZeroDReactor,
)
from pymcc.constants import BOLTZMANN, ELEMENTARY_CHARGE, ev_to_kelvin

from ._helpers import M_E, M_HE

DENSITY = 1.0e20
RATE = 1.0e-15
NU = DENSITY * RATE
SAFETY = 1.02
GAS_TEMPERATURE = 300.0


def _step_probability(dt: float) -> float:
    return (1.0 - math.exp(-SAFETY * NU * dt)) / SAFETY


def _operator(process: CollisionProcess, masses: dict[str, float], incident: str):
    return MonteCarloCollisions(
        species_masses=masses,
        backgrounds=[NeutralBackground("gas", DENSITY, GAS_TEMPERATURE, M_HE)],
        processes={incident: [process]},
        seed=3,
        bound_safety=SAFETY,
    )


def test_resonant_charge_exchange_relaxes_temperature_and_drift() -> None:
    """Collided ions carry the gas temperature, the others keep T0 and the drift."""
    process = CollisionProcess(
        "backscatter", "gas", CrossSection.maxwell_molecule(RATE, M_HE / 2)
    )
    temperature, drift = 3000.0, 2000.0
    ions = ParticleArrays.maxwellian(
        200_000, M_HE, temperature, drift=(drift, 0, 0), seed=1
    )
    dt = 0.02 / NU
    reactor = ZeroDReactor(_operator(process, {"ion": M_HE}, "ion"), {"ion": ions})
    history = reactor.run(dt, 200, every=10)

    f = (1.0 - _step_probability(dt)) ** (history.time / dt)
    mixing = M_HE * drift**2 * f * (1.0 - f) / (3.0 * BOLTZMANN)
    expected = GAS_TEMPERATURE + (temperature - GAS_TEMPERATURE) * f + mixing
    assert np.allclose(history.temperature["ion"], expected, rtol=0.02)
    assert history.collisions["backscatter:gas"][0] == 0
    assert np.all(np.diff(history.collisions["backscatter:gas"]) > 0)


def test_equal_mass_elastic_energy_relaxes_at_half_the_collision_frequency() -> None:
    """For Maxwell molecules d<E>/dt = -(2 m M / (m + M)^2) nu (<E> - 3/2 k T_g)."""
    process = CollisionProcess(
        "elastic", "gas", CrossSection.maxwell_molecule(RATE, M_HE / 2)
    )
    ions = ParticleArrays.maxwellian(
        200_000, M_HE, GAS_TEMPERATURE, drift=(5000.0, 0, 0), seed=2
    )
    dt = 0.02 / NU
    reactor = ZeroDReactor(_operator(process, {"ion": M_HE}, "ion"), {"ion": ions})
    history = reactor.run(dt, 200, every=20)

    gas_energy = 1.5 * BOLTZMANN * GAS_TEMPERATURE / ELEMENTARY_CHARGE
    initial = history.mean_energy_ev["ion"][0]
    effective_rate = -math.log(1.0 - _step_probability(dt)) / dt
    expected = gas_energy + (initial - gas_energy) * np.exp(
        -0.5 * effective_rate * history.time
    )
    assert np.allclose(history.mean_energy_ev["ion"], expected, rtol=0.02)


def test_ionization_grows_the_density_geometrically() -> None:
    process = CollisionProcess(
        "ionization",
        "gas",
        CrossSection.maxwell_molecule(RATE, M_E),
        energy_frame="lab",
        products={"electron": "e", "ion": "He+"},
    )
    num = 20_000
    electrons = ParticleArrays.maxwellian(num, M_E, ev_to_kelvin(5.0), seed=5)
    dt = 0.01 / NU
    reactor = ZeroDReactor(
        _operator(process, {"e": M_E, "He+": M_HE}, "e"), {"e": electrons}, volume=2.0
    )
    history = reactor.run(dt, 300, every=30)

    growth = (1.0 + _step_probability(dt)) ** (history.time / dt)
    assert np.allclose(history.density["e"], num / 2.0 * growth, rtol=0.02)
    # every ionization makes one electron and one ion
    assert np.allclose(history.density["e"] - history.density["He+"], num / 2.0)
    assert history.num_markers["He+"][-1] == history.collisions["ionization:gas"][-1]
    assert reactor.iteration == 300
    assert reactor.time == pytest.approx(300 * dt)
    assert reactor.volume == 2.0


def test_attachment_decays_the_electron_density() -> None:
    process = CollisionProcess(
        "attachment",
        "gas",
        CrossSection.maxwell_molecule(RATE, M_E),
        energy_frame="lab",
        products={"negative_ion": "He-"},
    )
    num = 100_000
    electrons = ParticleArrays.maxwellian(num, M_E, ev_to_kelvin(1.0), seed=6)
    dt = 0.01 / NU
    reactor = ZeroDReactor(
        _operator(process, {"e": M_E, "He-": M_HE}, "e"), {"e": electrons}
    )
    history = reactor.run(dt, 300, every=30)

    survival = (1.0 - _step_probability(dt)) ** (history.time / dt)
    assert np.allclose(history.density["e"], num * survival, rtol=0.02)
    assert np.allclose(history.density["e"] + history.density["He-"], num)
    # negative ions start at the (thermal) neutral velocity
    assert history.temperature["He-"][-1] == pytest.approx(GAS_TEMPERATURE, rel=0.05)


def test_reactor_validates_its_input() -> None:
    process = CollisionProcess("elastic", "gas", CrossSection.constant(1e-19))
    mcc = _operator(process, {"e": M_E}, "e")
    with pytest.raises(ValueError, match="volume"):
        ZeroDReactor(mcc, {}, volume=0.0)
    with pytest.raises(KeyError, match="mass"):
        ZeroDReactor(mcc, {"Ar+": ParticleArrays.empty()})
    two = _operator(process, {"e": M_E, "i": M_HE}, "e")
    with pytest.raises(ValueError, match="position components"):
        ZeroDReactor(two, {"e": ParticleArrays.empty(1), "i": ParticleArrays.empty(2)})
    reactor = ZeroDReactor(mcc, {})
    assert len(reactor.species["e"]) == 0
    assert reactor.collisions is mcc
    with pytest.raises(ValueError, match="every"):
        reactor.run(1.0, 1, every=0)


def test_history_stacks_states_with_missing_keys(tmp_path: Path) -> None:
    process = CollisionProcess("elastic", "gas", CrossSection.constant(1e-19))
    electrons = ParticleArrays.maxwellian(100, M_E, 1.0e4, seed=0)
    reactor = ZeroDReactor(_operator(process, {"e": M_E}, "e"), {"e": electrons})
    states = [reactor.state(), reactor.step(1.0e-9)]
    history = ReactorHistory.from_states(states)
    assert history.time.shape == (2,)
    assert list(history.collisions["elastic:gas"])[0] == 0
    assert history.num_markers["e"].tolist() == [100, 100]
