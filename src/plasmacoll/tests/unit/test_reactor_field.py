"""The 0D reactor in an electric field, and its marker-number control.

For a Maxwell-molecule elastic cross section (constant collision frequency
``nu``) with isotropic scattering, the drift velocity in a field ``E`` is
``q E / (mu nu)`` and the mean energy obeys Wannier's relation
``3/2 k T + (m + M) v_d^2 / 2``, both exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from plasmacoll import (
    CollisionProcess,
    CrossSection,
    MonteCarloCollisions,
    NeutralBackground,
    ParticleArrays,
    ZeroDReactor,
)
from plasmacoll.constants import BOLTZMANN, ELEMENTARY_CHARGE

from ._helpers import M_E, M_HE

DENSITY = 1.0e21
RATE = 1.0e-15
NU = DENSITY * RATE
TEMPERATURE = 300.0


def _swarm_operator(seed: int = 1) -> MonteCarloCollisions:
    reduced = M_HE / 2.0
    return MonteCarloCollisions(
        {"He+": M_HE, "e": M_E},
        [NeutralBackground("He", DENSITY, TEMPERATURE, M_HE)],
        {
            "He+": [
                CollisionProcess(
                    "elastic", "He", CrossSection.maxwell_molecule(RATE, reduced)
                )
            ]
        },
        seed=seed,
    )


def test_drift_velocity_and_mean_energy_match_the_maxwell_model() -> None:
    field = 2.0e3
    ions = ParticleArrays.maxwellian(10_000, M_HE, TEMPERATURE, seed=2)
    reactor = ZeroDReactor(
        _swarm_operator(),
        {"He+": ions},
        electric_field=(field, 0.0, 0.0),
        charges={"He+": 1, "e": -1},
    )
    assert reactor.electric_field == (field, 0.0, 0.0)
    history = reactor.run(dt=0.02 / NU, num_steps=1500, every=10)
    late = history.time > 10.0 / NU
    drift = history.mean_velocity["He+"][late].mean(axis=0)
    energy = history.mean_energy_ev["He+"][late].mean()

    expected_drift = ELEMENTARY_CHARGE * field / (M_HE / 2.0 * NU)
    expected_energy = (
        1.5 * BOLTZMANN * TEMPERATURE + M_HE * expected_drift**2
    ) / ELEMENTARY_CHARGE
    assert drift[0] == pytest.approx(expected_drift, rel=0.03)
    assert abs(drift[1]) < 0.03 * expected_drift
    assert energy == pytest.approx(expected_energy, rel=0.03)
    # The empty electron species has no mean velocity.
    np.testing.assert_allclose(history.mean_velocity["e"], 0.0)
    assert history.mean_velocity["He+"].shape == (history.time.shape[0], 3)


def test_reactor_merges_species_above_max_markers() -> None:
    ions = ParticleArrays.maxwellian(3000, M_HE, TEMPERATURE, seed=3)
    weight = ions.total_weight
    reactor = ZeroDReactor(_swarm_operator(), {"He+": ions}, max_markers=1000, seed=4)
    state = reactor.step(0.01 / NU)
    assert state.num_markers["He+"] <= 1000
    assert state.density["He+"] == pytest.approx(weight)

    per_species = ZeroDReactor(
        _swarm_operator(),
        {"He+": ParticleArrays.maxwellian(3000, M_HE, TEMPERATURE, seed=3)},
        max_markers={"He+": 2000, "missing": 10},
    )
    assert per_species.step(0.01 / NU).num_markers["He+"] <= 2000


def test_reactor_validates_field_charges_and_limits() -> None:
    ions = {"He+": ParticleArrays.maxwellian(10, M_HE, TEMPERATURE, seed=3)}
    with pytest.raises(KeyError, match="Charge given"):
        ZeroDReactor(_swarm_operator(), ions, charges={"Ar+": 1})
    with pytest.raises(ValueError, match="three components"):
        ZeroDReactor(_swarm_operator(), ions, electric_field=(1.0, 2.0))
    with pytest.raises(ValueError, match="max_markers"):
        ZeroDReactor(_swarm_operator(), ions, max_markers=3)


def test_field_does_not_push_neutral_or_empty_species() -> None:
    ions = ParticleArrays.maxwellian(100, M_HE, 0.0, seed=3)
    reactor = ZeroDReactor(
        _swarm_operator(),
        {"He+": ions},
        electric_field=(0.0, 0.0, 1.0e3),
        charges={"He+": 0.0, "e": -1.0},
    )
    state = reactor.step(1.0e-12)
    np.testing.assert_allclose(state.mean_velocity["He+"], 0.0, atol=1e-6)
