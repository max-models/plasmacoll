"""Stripping of a fast beam marker by marker against the fractions; synthetic cross sections."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest

from plasmacoll import (
    ChargeStateTransitions,
    CollisionProcess,
    CrossSection,
    MonteCarloCollisions,
    NeutralBackground,
    ParticleArrays,
    charge_state_fractions,
    stripping_processes,
)
from plasmacoll.constants import ATOMIC_MASS, ELECTRON_MASS

from ._helpers import beam

M_H = 1.00782503 * ATOMIC_MASS - ELECTRON_MASS  # the proton
MASSES = {
    "H-": M_H + 2.0 * ELECTRON_MASS,
    "H": M_H + ELECTRON_MASS,
    "H+": M_H,
    "e": ELECTRON_MASS,
    "H2+": 2.01565 * ATOMIC_MASS - ELECTRON_MASS,
}
SPECIES = {"-": "H-", "0": "H", "+": "H+"}
GAS = 1.0e20  # m^-3
SIGMA = {
    ("-", "0"): 1.2e-20,
    ("-", "+"): 1.5e-21,
    ("0", "+"): 6.0e-21,
    ("+", "0"): 1.0e-21,
}
ENERGY = 1.0e5  # eV


def operator(transitions, seed=3, **options) -> MonteCarloCollisions:
    return MonteCarloCollisions(
        MASSES,
        [NeutralBackground("gas", GAS, 0.0, 2.01565 * ATOMIC_MASS)],
        stripping_processes(transitions, SPECIES, **options),
        seed=seed,
    )


def h_minus_beam(num: int) -> dict[str, ParticleArrays]:
    velocities = beam(num, ENERGY, MASSES["H-"])
    species = {name: ParticleArrays.empty(ndim=1) for name in MASSES}
    species["H-"].add(xp.zeros((num, 1)), velocities, xp.ones(num))
    return species


def test_the_processes_follow_the_transitions() -> None:
    transitions = ChargeStateTransitions({"gas": SIGMA})
    processes = stripping_processes(
        transitions, SPECIES, electron="e", ions={"gas": "H2+"}
    )
    assert sorted(processes) == ["H", "H+", "H-"]
    single, double = processes["H-"]
    assert (single.name, single.charge_change, single.kind) == (
        "-->0:gas",
        1,
        "stripping",
    )
    assert single.products == {"projectile": "H", "electron": "e"}
    assert double.charge_change == 2 and double.products["projectile"] == "H+"
    (capture,) = processes["H+"]
    assert capture.charge_change == -1
    assert capture.products == {"projectile": "H", "ion": "H2+"}
    assert capture.energy_frame == "lab"
    assert float(capture.cross_section(1e5)) == pytest.approx(1.0e-21)
    # states without a species drop out with their transitions
    only = stripping_processes(transitions, {"-": "H-", "0": "H"}, {"gas": "H2"})
    assert list(only) == ["H-"] and only["H-"][0].background == "H2"


def test_the_populations_follow_the_fractions_along_the_path() -> None:
    """Over many markers the share in each state is ``charge_state_fractions``."""
    num = 20000
    transitions = ChargeStateTransitions({"gas": SIGMA})
    mcc = operator(transitions)
    species = h_minus_beam(num)
    speed = float(beam(1, ENERGY, MASSES["H-"])[0, 0])
    dt = 2.0e-9  # collision probability ~1% per step
    steps = np.arange(0, 301, 20)
    line_density = GAS * speed * dt * steps
    expected = charge_state_fractions(transitions, {"gas": line_density})
    names = ("H-", "H", "H+")
    for k in range(1, steps[-1] + 1):
        mcc.collide_species(species, dt, incident=names)
        if k in steps:
            row = int(np.searchsorted(steps, k))
            shares = np.array([species[n].total_weight for n in names]) / num
            assert shares.sum() == pytest.approx(1.0, abs=1e-12)
            spread = np.sqrt(expected[row] * (1.0 - expected[row]) / num)
            assert np.all(np.abs(shares - expected[row]) < 5.0 * spread), k
    # the beam is well into its evolution: H- mostly gone, H+ grown
    assert expected[-1][0] < 0.4 and expected[-1][2] > 0.1


def test_every_event_conserves_particles_and_momentum() -> None:
    num = 4000
    transitions = ChargeStateTransitions(
        {"gas": {("-", "0"): 1e-19, ("-", "+"): 5e-20}}
    )
    mcc = operator(transitions, electron="e")
    velocities = beam(num, ENERGY, MASSES["H-"])
    velocities[:, 1] = xp.asarray(1.0e4 * np.linspace(-1.0, 1.0, num))  # any direction
    weights = xp.asarray(np.linspace(0.5, 2.0, num))
    result = mcc.collide("H-", xp.zeros((num, 1)), velocities, weights, dt=2e-8)
    assert result.diagnostics.counts["-->0:gas"] > 100
    assert result.diagnostics.counts["-->+:gas"] > 50
    removed = xp.to_numpy(result.removed)
    before = xp.to_numpy(velocities)
    w = xp.to_numpy(weights)
    # the survivors keep their velocity
    assert xp.to_numpy(result.velocities)[~removed] == pytest.approx(before[~removed])
    products = {
        name: [(xp.to_numpy(b.weights), xp.to_numpy(b.velocities)) for b in batches]
        for name, batches in result.created.items()
    }
    heavy = sum(float(bw.sum()) for n in ("H", "H+") for bw, _ in products[n])
    assert heavy == pytest.approx(float(w[removed].sum()), rel=1e-14)
    momentum_after = sum(
        MASSES[name] * (bw[:, None] * bv).sum(axis=0)
        for name, batches in products.items()
        for bw, bv in batches
    )
    momentum_removed = MASSES["H-"] * (w[removed, None] * before[removed]).sum(axis=0)
    assert momentum_after == pytest.approx(momentum_removed, rel=1e-12)
    # one electron per single stripping, two per double: the charge balances
    charge_after = sum(
        q * sum(float(bw.sum()) for bw, _ in products[n])
        for n, q in (("H", 0), ("H+", 1), ("e", -1))
    )
    assert charge_after == pytest.approx(-float(w[removed].sum()), rel=1e-14)
    for _, bv in products["H"] + products["H+"] + products["e"]:
        assert np.all(bv[:, 0] == before[0, 0])  # at the projectile's velocity


def test_capture_leaves_a_target_ion_at_rest() -> None:
    transitions = ChargeStateTransitions({"gas": {("+", "0"): 1e-19}})
    mcc = operator(transitions, electron="e", ions={"gas": "H2+"})
    num = 2000
    velocities = beam(num, ENERGY, MASSES["H+"])
    result = mcc.collide("H+", xp.zeros((num, 1)), velocities, xp.ones(num), dt=2e-8)
    count = result.diagnostics.counts["+->0:gas"]
    assert count > 100 and int(xp.count_nonzero(result.removed)) == count
    assert "e" not in result.created  # a capture releases no electron
    (ion,) = result.created["H2+"]
    (neutral,) = result.created["H"]
    assert ion.weights.size == neutral.weights.size == count
    assert bool(xp.all(ion.velocities == 0.0))  # a cold gas
    assert bool(xp.all(neutral.velocities == velocities[0]))  # at its own velocity


def test_the_kind_is_checked() -> None:
    sigma = CrossSection.constant(1e-20)
    with pytest.raises(ValueError, match="projectile"):
        CollisionProcess("stripping", "gas", sigma)
    with pytest.raises(ValueError, match="non-zero integer"):
        CollisionProcess(
            "stripping", "gas", sigma, products={"projectile": "H"}, charge_change=0
        )
    with pytest.raises(ValueError, match="releases no electron"):
        CollisionProcess(
            "stripping",
            "gas",
            sigma,
            products={"projectile": "H", "electron": "e"},
            charge_change=-1,
        )
    process = CollisionProcess("stripping", "gas", sigma, products={"projectile": "H"})
    assert process.loss == 0.0 and process.name == "stripping:gas"
    transitions = ChargeStateTransitions(
        {"gas": {("a", "b"): 1e-20}}, states=("a", "b")
    )
    with pytest.raises(ValueError, match="no charge"):
        stripping_processes(transitions, {"a": "A", "b": "B"})
    with pytest.raises(ValueError, match="one charge"):
        stripping_processes(transitions, {"a": "A", "b": "B"}, charges={"a": 1, "b": 1})
