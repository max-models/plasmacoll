"""Coulomb collisions and charged-charged reactions.

Takizuka-Abe scattering conserves momentum and energy pair by pair and has a
known drag on a fast beam; the reactions follow the closed-form solutions of
their rate equations.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import cunumpy as xp
import numpy as np
import pytest

from plasmacoll import (
    ChargedCollisions,
    ChargedReaction,
    CrossSection,
    ParticleArrays,
    coulomb_logarithm,
)
from plasmacoll.charged import VACUUM_PERMITTIVITY
from plasmacoll.constants import ATOMIC_MASS, ELECTRON_MASS, ELEMENTARY_CHARGE

M_H = 1.00728 * ATOMIC_MASS
M_HE = 4.0026 * ATOMIC_MASS


def _host(array) -> np.ndarray:
    return np.asarray(xp.to_numpy(array))


def _markers(
    num: int, mass: float, temperature_ev: float, seed: int, weight=1.0, ndim=1
):
    rng = np.random.default_rng(seed)
    velocities = rng.standard_normal((num, 3)) * math.sqrt(
        temperature_ev * ELEMENTARY_CHARGE / mass
    )
    return ParticleArrays(
        xp.zeros((num, ndim)), xp.asarray(velocities), xp.full(num, float(weight))
    )


def _momentum_energy(species, masses):
    momentum = sum(
        masses[name] * _host(xp.sum(m.weights[:, None] * m.velocities, axis=0))
        for name, m in species.items()
    )
    energy = sum(
        0.5
        * masses[name]
        * float(_host(xp.sum(m.weights * xp.sum(m.velocities**2, axis=1))))
        for name, m in species.items()
    )
    return momentum, energy


# --------------------------------------------------------------------- #
# Coulomb scattering
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("counts", [(1000, 1000), (1001, 333), (300, 1201)])
def test_coulomb_collisions_conserve_momentum_and_energy(counts) -> None:
    masses = {"a": M_H, "b": M_HE}
    species = {
        "a": _markers(counts[0], M_H, 2.0, 1),
        "b": _markers(counts[1], M_HE, 1.0, 2),
    }
    collisions = ChargedCollisions(
        masses, {"a": 1, "b": 1}, coulomb_pairs="all", seed=3
    )
    assert collisions.coulomb_pairs == [("a", "a"), ("a", "b"), ("b", "b")]
    before = _momentum_energy(species, masses)
    diagnostics = collisions.collide_species(species, 1.0e-7, cell_volume=1.0e-17)
    after = _momentum_energy(species, masses)
    scale = M_HE * 1.0e4 * sum(counts)
    np.testing.assert_allclose(after[0], before[0], atol=1e-12 * scale)
    if counts[0] == counts[1]:
        # No marker is reused, so energy is conserved exactly.
        assert after[1] == pytest.approx(before[1], rel=1e-12)
    else:
        assert after[1] == pytest.approx(before[1], rel=1e-3)
    assert diagnostics.coulomb_pairs["a:b"] == max(counts)
    assert diagnostics.coulomb_pairs["a:a"] == counts[0] // 2 + (counts[0] % 2) * 2


def test_like_collisions_isotropize_an_anisotropic_temperature() -> None:
    """Collisions within one species relax T_x != T_y towards one temperature."""
    rng = np.random.default_rng(1)
    velocities = rng.standard_normal((2001, 3)) * np.array([3.0e4, 1.0e4, 1.0e4])
    markers = ParticleArrays(xp.zeros((2001, 1)), xp.asarray(velocities), xp.ones(2001))
    collisions = ChargedCollisions(
        {"p": M_H}, {"p": 1}, coulomb_pairs=[("p", "p")], seed=2
    )
    before = _momentum_energy({"p": markers}, {"p": M_H})
    for _ in range(20):
        collisions.collide_species({"p": markers}, 1.0e-5, cell_volume=1.0e-16)
    after = _momentum_energy({"p": markers}, {"p": M_H})
    np.testing.assert_allclose(after[0], before[0], rtol=1e-10, atol=1e-30)
    assert after[1] == pytest.approx(before[1], rel=1e-10)
    spread = _host(xp.std(markers.velocities, axis=0))
    assert spread[0] < 2.8e4
    assert spread[1] > 1.2e4


def test_drag_on_a_fast_beam_matches_the_slowing_down_rate() -> None:
    """``dv/dt = -n e^4 lnL / (4 pi eps0^2 m mu v^2)`` on cold, heavy partners."""
    num, speed, density, log = 50_000, 1.0e5, 1.0e20, 10.0
    heavy = 1.0e6 * ATOMIC_MASS
    species = {
        "a": ParticleArrays(
            xp.zeros((num, 1)),
            xp.asarray(np.tile([speed, 0.0, 0.0], (num, 1))),
            xp.ones(num),
        ),
        "b": ParticleArrays(xp.zeros((num, 1)), xp.zeros((num, 3)), xp.ones(num)),
    }
    collisions = ChargedCollisions(
        {"a": M_H, "b": heavy},
        {"a": 1, "b": 1},
        coulomb_pairs=[("a", "b")],
        coulomb_logarithm=log,
        seed=4,
    )
    reduced = M_H * heavy / (M_H + heavy)
    drag = (
        density
        * ELEMENTARY_CHARGE**4
        * log
        / (4.0 * math.pi * VACUUM_PERMITTIVITY**2 * M_H * reduced * speed**2)
    )
    dt = 0.002 * speed / drag
    collisions.collide_species(species, dt, cell_volume=num / density)
    slowing = (speed - float(_host(xp.mean(species["a"].velocities[:, 0])))) / dt
    assert slowing == pytest.approx(drag, rel=0.03)


def test_temperature_relaxation_approaches_the_spitzer_rate() -> None:
    """Takizuka-Abe relaxes two temperatures at the Spitzer rate up to its time-step error.

    The binary model's error falls slowly with ``nu dt``: at ``nu_eps dt = 1e-3``
    it relaxes at about 0.93 of the Spitzer rate (see the validation guide).
    """
    masses = {"a": M_H, "b": M_HE}
    num, density, log = 40_000, 1.0e20, 10.0
    rng = np.random.default_rng(5)

    def exact(mass, temperature_ev):
        velocities = rng.standard_normal((num, 3))
        velocities -= velocities.mean(axis=0)
        velocities *= (
            math.sqrt(temperature_ev * ELEMENTARY_CHARGE / mass) / velocities.std()
        )
        return ParticleArrays(xp.zeros((num, 1)), xp.asarray(velocities), xp.ones(num))

    species = {"a": exact(M_H, 2.0), "b": exact(M_HE, 1.0)}
    collisions = ChargedCollisions(
        masses, {"a": 1, "b": 1}, coulomb_pairs="all", coulomb_logarithm=log, seed=6
    )

    def spitzer(temperature_a, temperature_b):
        return (
            8.0
            * math.sqrt(2.0 * math.pi)
            * density
            * ELEMENTARY_CHARGE**4
            * log
            * math.sqrt(M_H * M_HE)
            / (
                3.0
                * (4.0 * math.pi * VACUUM_PERMITTIVITY) ** 2
                * ELEMENTARY_CHARGE**1.5
                * (M_H * temperature_b + M_HE * temperature_a) ** 1.5
            )
        )

    rate = spitzer(2.0, 1.0)
    dt = 1.0e-3 / rate
    for _ in range(50):
        collisions.collide_species(species, dt, cell_volume=num / density)
    temperature = (
        M_H * float(_host(xp.mean(xp.sum(species["a"].velocities ** 2, axis=1)))) / 3.0
    ) / ELEMENTARY_CHARGE
    expected_change = -rate * 50 * dt  # (T_b - T_a) = -1 eV, nearly constant here
    assert (temperature - 2.0) / expected_change == pytest.approx(0.93, abs=0.12)


def test_cells_keep_pairs_apart() -> None:
    """Markers only pair with markers of their own cell."""
    fast = ParticleArrays(
        xp.asarray(np.full((500, 1), 0.5)),
        xp.asarray(np.tile([1e5, 0, 0], (500, 1))),
        xp.ones(500),
    )
    partners = ParticleArrays(
        xp.asarray(np.full((500, 1), 1.5)), xp.zeros((500, 3)), xp.ones(500)
    )
    collisions = ChargedCollisions(
        {"a": M_H, "b": M_HE}, {"a": 1, "b": 1}, coulomb_pairs=[("a", "b")], seed=7
    )
    diagnostics = collisions.collide_species(
        {"a": fast, "b": partners}, 1.0e-6, cell_volume=1e-16, cell_size=1.0
    )
    assert diagnostics.coulomb_pairs["a:b"] == 0
    np.testing.assert_allclose(_host(fast.velocities[:, 0]), 1e5)
    np.testing.assert_allclose(_host(partners.velocities), 0.0)


def test_cells_in_several_dimensions() -> None:
    rng = np.random.default_rng(8)
    species = {
        "a": ParticleArrays(
            xp.asarray(rng.uniform(0, 2, (600, 2))),
            xp.asarray(rng.normal(0, 1e4, (600, 3))),
            xp.ones(600),
        ),
        "b": ParticleArrays(
            xp.asarray(rng.uniform(0, 2, (300, 2))),
            xp.asarray(rng.normal(0, 1e4, (300, 3))),
            xp.ones(300),
        ),
    }
    masses = {"a": M_H, "b": M_HE}
    collisions = ChargedCollisions(
        masses, {"a": 1, "b": 1}, coulomb_pairs="all", seed=9
    )
    before = _momentum_energy(species, masses)
    diagnostics = collisions.collide_species(
        species, 1e-7, cell_volume=1e-16, cell_size=(1.0, 1.0)
    )
    np.testing.assert_allclose(
        _momentum_energy(species, masses)[0], before[0], atol=1e-30
    )
    assert diagnostics.coulomb_pairs["a:b"] > 0
    with pytest.raises(ValueError, match="cell_size"):
        collisions.collide_species(species, 1e-7, cell_volume=1e-16, cell_size=0.0)


def test_missing_species_and_markers_at_rest() -> None:
    collisions = ChargedCollisions(
        {"a": M_H, "b": M_HE}, {"a": 1, "b": 1}, coulomb_pairs="all", seed=10
    )
    resting = ParticleArrays(xp.zeros((4, 1)), xp.zeros((4, 3)), xp.ones(4))
    diagnostics = collisions.collide_species({"a": resting}, 1e-6, cell_volume=1e-16)
    assert diagnostics.coulomb_pairs == {"a:a": 2, "a:b": 0, "b:b": 0}
    np.testing.assert_allclose(_host(resting.velocities), 0.0)


# --------------------------------------------------------------------- #
# Reactions
# --------------------------------------------------------------------- #
def test_mutual_neutralization_follows_the_rate_equation_with_unequal_weights() -> None:
    rate, volume = 4.0e-13, 1.0e-6
    n_minus, n_plus = 1.0e17, 2.0e17
    masses = {"H-": M_H, "H+": M_H, "H": M_H}
    species = {
        "H-": _markers(20_000, M_H, 0.1, 11, weight=n_minus * volume / 20_000),
        "H+": _markers(30_000, M_H, 0.1, 12, weight=n_plus * volume / 30_000),
        "H": ParticleArrays.empty(),
    }
    reaction = ChargedReaction(
        "mutual_neutralization",
        ("H-", "H+"),
        rate_coefficient=rate,
        products={"neutral_1": "H", "neutral_2": "H"},
    )
    assert reaction.name == "mutual_neutralization:H-:H+"
    collisions = ChargedCollisions(
        masses, {"H-": -1, "H+": 1}, reactions=[reaction], seed=13
    )
    dt = 0.01 / (rate * n_plus)
    for _ in range(100):
        diagnostics = collisions.collide_species(species, dt, cell_volume=volume)
    t = 100 * dt
    delta = n_plus - n_minus
    expected = delta / ((n_plus / n_minus) * math.exp(rate * delta * t) - 1.0)
    density_minus = species["H-"].total_weight / volume
    density_plus = species["H+"].total_weight / volume
    assert density_minus == pytest.approx(expected, rel=0.02)
    assert density_plus - density_minus == pytest.approx(delta, rel=1e-9)
    neutrals = species["H"].total_weight / volume
    assert neutrals == pytest.approx(2.0 * (n_minus - density_minus), rel=1e-9)
    assert diagnostics.reactions[reaction.name] > 0
    assert diagnostics.max_reaction_probability < 0.1


def test_recombination_with_a_cross_section_conserves_momentum() -> None:
    # The neutral carries the mass of the electron and the ion.
    masses = {"e": ELECTRON_MASS, "H+": M_H, "H": M_H + ELECTRON_MASS}
    species = {
        "e": _markers(5000, ELECTRON_MASS, 1.0, 14),
        "H+": _markers(5000, M_H, 0.1, 15),
        "H": ParticleArrays.empty(),
    }
    reaction = ChargedReaction(
        "recombination",
        ("e", "H+"),
        cross_section=CrossSection.constant(1.0e-16),
        products={"neutral": "H"},
    )
    collisions = ChargedCollisions(
        masses,
        {"e": -1, "H+": 1},
        reactions=[reaction],
        seed=16,
        max_reaction_probability=1.0,
    )
    before = _momentum_energy(species, masses)[0]
    diagnostics = collisions.collide_species(species, 1.0e-9, cell_volume=5000 / 1e18)
    after = _momentum_energy(species, masses)[0]
    count = diagnostics.reactions[reaction.name]
    assert count > 0
    assert len(species["e"]) == 5000 - count == len(species["H+"])
    assert len(species["H"]) == count
    np.testing.assert_allclose(after, before, rtol=1e-9)


def test_electron_detachment_adds_two_electrons_with_the_residual_energy() -> None:
    masses = {"e": ELECTRON_MASS, "H-": M_H, "H": M_H}
    species = {
        "e": _markers(4000, ELECTRON_MASS, 20.0, 17),
        "H-": _markers(4000, M_H, 0.0, 18),
        "H": ParticleArrays.empty(),
    }
    reaction = ChargedReaction(
        "electron_detachment",
        ("e", "H-"),
        cross_section=CrossSection.constant(5.0e-19, threshold=0.75),
        products={"electron": "e", "neutral": "H"},
    )
    assert reaction.loss == 0.75
    collisions = ChargedCollisions(
        masses, {"e": -1, "H-": -1}, reactions=[reaction], seed=19
    )
    with pytest.warns(RuntimeWarning, match="reaction probability"):
        diagnostics = collisions.collide_species(
            species, 1.0e-7, cell_volume=4000 / 1e18
        )
    count = diagnostics.reactions[reaction.name]
    assert count > 0
    assert len(species["e"]) == 4000 + count
    assert len(species["H-"]) == 4000 - count
    assert diagnostics.weighted_reactions[reaction.name] == pytest.approx(count)
    # A second large step does not warn again.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        collisions.collide_species(species, 1.0e-7, cell_volume=4000 / 1e18)


def test_detachment_loss_defaults_and_reaction_without_partners() -> None:
    plain = ChargedReaction(
        "electron_detachment",
        ("e", "H-"),
        rate_coefficient=1e-14,
        products={"electron": "e"},
    )
    assert plain.loss == 0.0
    given = ChargedReaction(
        "electron_detachment",
        ("e", "H-"),
        rate_coefficient=1e-14,
        products={"electron": "e"},
        energy_loss=0.75,
    )
    assert given.loss == 0.75
    assert (
        ChargedReaction("recombination", ("e", "H+"), rate_coefficient=1e-19).loss
        == 0.0
    )
    collisions = ChargedCollisions(
        {"e": ELECTRON_MASS, "H-": M_H}, {"e": -1, "H-": -1}, reactions=[plain], seed=20
    )
    diagnostics = collisions.collide_species(
        {"e": _markers(10, ELECTRON_MASS, 1.0, 21), "H-": ParticleArrays.empty()},
        1e-9,
        cell_volume=1e-12,
    )
    assert diagnostics.reactions == {plain.name: 0}
    assert collisions.reactions == (plain,)
    # Partners present, but a zero rate: no pair reacts.
    never = ChargedReaction(
        "recombination", ("e", "H-"), rate_coefficient=0.0, name="never"
    )
    collisions = ChargedCollisions(
        {"e": ELECTRON_MASS, "H-": M_H}, {"e": -1, "H-": -1}, reactions=[never], seed=21
    )
    diagnostics = collisions.collide_species(
        {"e": _markers(10, ELECTRON_MASS, 1.0, 22), "H-": _markers(10, M_H, 0.1, 23)},
        1e-9,
        cell_volume=1e-12,
    )
    assert diagnostics.reactions == {"never": 0}


def test_missing_product_species_raises() -> None:
    reaction = ChargedReaction(
        "recombination", ("e", "H+"), rate_coefficient=1e-6, products={"neutral": "H"}
    )
    collisions = ChargedCollisions(
        {"e": ELECTRON_MASS, "H+": M_H, "H": M_H},
        {"e": -1, "H+": 1},
        reactions=[reaction],
        seed=22,
        max_reaction_probability=1.0,
    )
    species = {
        "e": _markers(100, ELECTRON_MASS, 1.0, 23),
        "H+": _markers(100, M_H, 0.1, 24),
    }
    with (
        pytest.warns(RuntimeWarning, match="reaction probability"),
        pytest.raises(KeyError, match="not simulated"),
    ):
        collisions.collide_species(species, 1.0, cell_volume=1e-12)


def test_definitions_are_validated() -> None:
    with pytest.raises(ValueError, match="Unknown charged reaction"):
        ChargedReaction("fusion", ("d", "t"), rate_coefficient=1.0)
    with pytest.raises(ValueError, match="two different species"):
        ChargedReaction("recombination", ("e", "e"), rate_coefficient=1.0)
    with pytest.raises(ValueError, match="exactly one"):
        ChargedReaction("recombination", ("e", "H+"))
    with pytest.raises(ValueError, match=">= 0"):
        ChargedReaction("recombination", ("e", "H+"), rate_coefficient=-1.0)
    with pytest.raises(ValueError, match="needs product"):
        ChargedReaction("electron_detachment", ("e", "H-"), rate_coefficient=1.0)
    with pytest.raises(ValueError, match="no product role"):
        ChargedReaction(
            "recombination", ("e", "H+"), rate_coefficient=1.0, products={"ion": "x"}
        )
    masses = {"e": ELECTRON_MASS, "H+": M_H}
    with pytest.raises(ValueError, match='"all"'):
        ChargedCollisions(masses, {"e": -1, "H+": 1}, coulomb_pairs="every")
    one_species: Any = [("e",)]
    with pytest.raises(ValueError, match="two species"):
        ChargedCollisions(masses, {"e": -1, "H+": 1}, coulomb_pairs=one_species)
    with pytest.raises(KeyError, match="No mass"):
        ChargedCollisions(masses, {"e": -1, "H+": 1}, coulomb_pairs=[("e", "Ar+")])
    with pytest.raises(KeyError, match="No charge"):
        ChargedCollisions(masses, {"e": -1}, coulomb_pairs=[("e", "H+")])
    with pytest.raises(ValueError, match="no charge"):
        ChargedCollisions(masses, {"e": -1, "H+": 0}, coulomb_pairs=[("e", "H+")])
    duplicate = ChargedReaction("recombination", ("e", "H+"), rate_coefficient=1.0)
    with pytest.raises(ValueError, match="unique"):
        ChargedCollisions(masses, {"e": -1, "H+": 1}, reactions=[duplicate, duplicate])
    with pytest.raises(KeyError, match="No mass"):
        ChargedCollisions(
            masses,
            {"e": -1, "H+": 1},
            reactions=[
                ChargedReaction(
                    "recombination",
                    ("e", "H+"),
                    rate_coefficient=1.0,
                    products={"neutral": "H"},
                )
            ],
        )
    with pytest.raises(ValueError, match="coulomb_logarithm"):
        ChargedCollisions(masses, {"e": -1, "H+": 1}, coulomb_logarithm=0.0)
    collisions = ChargedCollisions(masses, {"e": -1, "H+": 1})
    assert collisions.charges == {"e": -1.0, "H+": 1.0}
    with pytest.raises(ValueError, match="cell_volume"):
        collisions.collide_species({}, 1e-9, cell_volume=0.0)


def test_coulomb_logarithm_of_the_nrl_formulary() -> None:
    # T < 10 Z^2 eV: 23 - ln(n^1/2 Z T^-3/2), n in cm^-3.
    assert coulomb_logarithm(1.0e18, 2.0) == pytest.approx(
        23.0 - math.log(math.sqrt(1.0e12) * 2.0**-1.5)
    )
    assert coulomb_logarithm(1.0e18, 100.0) == pytest.approx(
        24.0 - math.log(math.sqrt(1.0e12) / 100.0)
    )
    with pytest.raises(ValueError, match="> 0"):
        coulomb_logarithm(0.0, 1.0)
