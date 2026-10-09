"""Gas coupling, speed classes, the reactor's fields and swarm diagnostics, and fits."""

from __future__ import annotations

import math
from typing import Any

import cunumpy as xp
import numpy as np
import pytest

from plasmacoll import (
    BackgroundTransfer,
    ChargedCollisions,
    ChargedReaction,
    CollisionProcess,
    CrossSection,
    MonteCarloCollisions,
    NeutralBackground,
    ParticleArrays,
    ZeroDReactor,
    cross_section_from_log_polynomial,
    double_log_polynomial,
    log_polynomial,
)
from plasmacoll.constants import ELEMENTARY_CHARGE
from plasmacoll.fits import CM2, CM3_PER_S

from ._helpers import M_E, M_HE, beam, constant, operator

RATE = 1.0e-15
GAS = 1.0e21
NU = GAS * RATE


def _host(array) -> np.ndarray:
    return np.asarray(xp.to_numpy(array))


def _maxwell(**options) -> MonteCarloCollisions:
    return MonteCarloCollisions(
        {"He+": M_HE},
        [NeutralBackground("He", GAS, 300.0, M_HE, **options)],
        {
            "He+": [
                CollisionProcess(
                    "elastic", "He", CrossSection.maxwell_molecule(RATE, M_HE / 2)
                )
            ]
        },
        seed=1,
    )


# --------------------------------------------------------------------- #
# Transfer to the gas
# --------------------------------------------------------------------- #
def test_transfer_closes_momentum_and_energy_of_every_process() -> None:
    """Plasma momentum and energy lost per step are exactly what the gas received."""
    processes = [
        CollisionProcess("elastic", "gas", constant(1e-19)),
        CollisionProcess("excitation", "gas", constant(1e-19, 5.0)),
        CollisionProcess(
            "ionization",
            "gas",
            constant(1e-19, 15.0),
            products={"electron": "e", "ion": "ion"},
            energy_frame="lab",
        ),
        CollisionProcess(
            "attachment", "gas", constant(1e-20), products={"negative_ion": "neg"}
        ),
    ]
    masses = {"e": M_E, "ion": M_HE, "neg": M_HE}
    mcc = operator(processes, masses, neutral_mass=M_HE, temperature=300.0)
    num = 20_000
    velocities = beam(num, 40.0, M_E)
    result = mcc.collide(
        "e",
        xp.zeros((num, 1)),
        velocities,
        xp.full(num, 2.0),
        dt=1.0,
        record_events=True,
    )
    removed = _host(result.removed)
    after = _host(result.velocities)
    p_before = M_E * 2.0 * _host(velocities).sum(axis=0)
    e_before = 0.5 * M_E * 2.0 * float((_host(velocities) ** 2).sum())
    p_after = M_E * 2.0 * after[~removed].sum(axis=0)
    e_after = 0.5 * M_E * 2.0 * float((after[~removed] ** 2).sum())
    for name, batches in result.created.items():
        for batch in batches:
            v = _host(batch.velocities)
            w = _host(batch.weights)
            p_after = p_after + masses[name] * (w[:, None] * v).sum(axis=0)
            e_after += 0.5 * masses[name] * float((w * (v**2).sum(axis=1)).sum())
    transfer = result.diagnostics.transfer["gas"]
    internal = (
        2.0
        * ELEMENTARY_CHARGE
        * (
            5.0 * result.diagnostics.counts["excitation:gas"]
            + 15.0 * result.diagnostics.counts["ionization:gas"]
        )
    )
    # The incident momentum and energy, minus what the markers kept, went to the gas.
    np.testing.assert_allclose(
        transfer.momentum, p_before - p_after, rtol=1e-9, atol=1e-30
    )
    assert transfer.energy == pytest.approx(e_before - e_after - internal, rel=1e-9)
    consumed = (
        result.diagnostics.counts["ionization:gas"]
        + result.diagnostics.counts["attachment:gas"]
    )
    assert transfer.particles == pytest.approx(2.0 * consumed)
    weighted = result.diagnostics.weighted_counts
    assert weighted["elastic:gas"] == pytest.approx(
        2.0 * result.diagnostics.counts["elastic:gas"]
    )
    events = result.diagnostics.events["gas"]
    assert (
        sum(int(batch.energy.shape[0]) for batch in events)
        == result.diagnostics.real_collisions
    )
    assert float(
        sum(float(_host(xp.sum(batch.energy))) for batch in events)
    ) == pytest.approx(transfer.energy)


def test_tracked_fast_neutral_consumes_a_gas_particle() -> None:
    process = CollisionProcess(
        "backscatter", "gas", constant(1e-18), products={"neutral": "He"}
    )
    mcc = operator(
        [process], {"He+": M_HE, "He": M_HE}, neutral_mass=M_HE, incident="He+"
    )
    result = mcc.collide(
        "He+", xp.zeros((100, 1)), beam(100, 10.0, M_HE), xp.ones(100), dt=1.0
    )
    transfer = result.diagnostics.transfer["gas"]
    assert transfer.particles == pytest.approx(100.0)
    untracked = operator(
        [CollisionProcess("backscatter", "gas", constant(1e-18))],
        {"He+": M_HE},
        neutral_mass=M_HE,
        incident="He+",
    )
    result = untracked.collide(
        "He+", xp.zeros((100, 1)), beam(100, 10.0, M_HE), xp.ones(100), dt=1.0
    )
    assert result.diagnostics.transfer["gas"].particles == 0.0


def test_background_transfer_accumulates() -> None:
    total = BackgroundTransfer()
    total.add(BackgroundTransfer(np.array([1.0, 0.0, 0.0]), 2.0, 3.0))
    total.add(BackgroundTransfer(np.array([1.0, 1.0, 0.0]), 1.0, 1.0))
    np.testing.assert_allclose(total.momentum, [2.0, 1.0, 0.0])
    assert (total.energy, total.particles) == (3.0, 4.0)


def test_reactor_heats_and_entrains_a_closed_gas() -> None:
    """A dense, hot, drifting ion population shares its energy and momentum with the gas."""
    volume = 1.0e-12
    weight = GAS * volume / 20_000
    ions = ParticleArrays.maxwellian(
        20_000, M_HE, 3000.0, weight=weight, drift=(5e3, 0, 0), seed=2
    )
    reactor = ZeroDReactor(
        _maxwell(), {"He+": ions}, volume=volume, evolve_backgrounds=True
    )
    history = reactor.run(0.02 / NU, 600, every=100)
    gas = reactor.collisions.backgrounds["He"]
    assert gas.temperature == pytest.approx(ions.temperature(M_HE), rel=0.05)
    assert gas.drift[0] == pytest.approx(2500.0, rel=0.05)
    assert history.gas_temperature["He"][-1] == pytest.approx(gas.temperature)
    assert reactor.transfer["He"].energy > 0.0


def _attachment_reactor(gas_density: float, electron_density: float, seed: int):
    volume = 1.0e-12
    attachment = CollisionProcess(
        "attachment",
        "He",
        CrossSection.maxwell_molecule(RATE, M_E),
        products={"negative_ion": "He-"},
        energy_frame="lab",
    )
    mcc = MonteCarloCollisions(
        {"e": M_E, "He-": M_HE},
        [NeutralBackground("He", gas_density, 300.0, M_HE)],
        {"e": [attachment]},
        seed=seed,
    )
    electrons = ParticleArrays.maxwellian(
        10_000, M_E, 3000.0, weight=electron_density * volume / 10_000, seed=seed + 1
    )
    return ZeroDReactor(mcc, {"e": electrons}, volume=volume, evolve_backgrounds=True)


def test_reactor_depletes_the_gas_by_attachment() -> None:
    """Each attachment uses up a gas atom: n_gas - n_e stays constant."""
    reactor = _attachment_reactor(1.0e16, 0.5e16, seed=3)
    history = reactor.run(0.01 / (1.0e16 * RATE), 300, every=100)
    difference = history.gas_density["He"] - history.density["e"]
    np.testing.assert_allclose(difference, 0.5e16, rtol=1e-9)
    assert history.gas_density["He"][-1] < 0.8e16


def test_a_used_up_gas_stays_empty() -> None:
    reactor = _attachment_reactor(1.0e16, 2.0e16, seed=5)
    # A deliberately huge step: every electron attaches, more than there is gas.
    reactor.step(10.0 / (1.0e16 * RATE))
    assert reactor.collisions.backgrounds["He"].density == 0.0


# --------------------------------------------------------------------- #
# Speed classes
# --------------------------------------------------------------------- #
def test_speed_classes_draw_fewer_candidates_without_changing_the_rates() -> None:
    """Slow markers among a few fast ones: far fewer null candidates, same real rate."""
    num, fast = 200_000, 100
    velocities = xp.concatenate([beam(num, 1.0, M_E), beam(fast, 1.0e4, M_E)])
    # nu_max dt = 0.1 for the fast markers; the slow ones are 100 times slower.
    dt = 0.1 / (1.0e20 * 1.0e-19 * math.sqrt(2.0e4 * ELEMENTARY_CHARGE / M_E))
    counts = {}
    for classes in (1, 8):
        mcc = operator(
            [CollisionProcess("elastic", "gas", constant(1e-19))],
            {"e": M_E},
            num_speed_classes=classes,
        )
        result = mcc.collide(
            "e", xp.zeros((num + fast, 1)), velocities, xp.ones(num + fast), dt=dt
        )
        rotated = int(
            xp.count_nonzero(xp.abs(result.velocities[:num, 1]) > 0.0)
        )  # slow markers that collided
        counts[classes] = (result.diagnostics.candidates, rotated)
    assert counts[8][0] < 0.1 * counts[1][0]
    expected = num * (1.0 - math.exp(-0.1)) / 100.0
    for _, collided in counts.values():
        assert collided == pytest.approx(expected, abs=5.0 * math.sqrt(expected))
    # One class: the alive mask and a zero frequency.
    single = operator(
        [CollisionProcess("elastic", "gas", constant(1e-19))],
        {"e": M_E},
        num_speed_classes=1,
    )
    alive = xp.asarray(np.arange(1000) % 2 == 0)
    result = single.collide(
        "e", xp.zeros((1000, 1)), beam(1000, 10.0, M_E), xp.ones(1000), 1.0, alive=alive
    )
    moved = _host(xp.abs(result.velocities[:, 1]) > 0.0)
    assert moved[::2].any() and not moved[1::2].any()
    silent = operator(
        [CollisionProcess("elastic", "gas", constant(0.0))],
        {"e": M_E},
        num_speed_classes=1,
    )
    quiet = silent.collide("e", xp.zeros((10, 1)), beam(10, 1.0, M_E), xp.ones(10), 1.0)
    assert quiet.diagnostics.candidates == 0
    with pytest.raises(ValueError, match="num_speed_classes"):
        operator([], {"e": M_E}, num_speed_classes=0)


# --------------------------------------------------------------------- #
# Fields and swarm diagnostics
# --------------------------------------------------------------------- #
def test_crossed_fields_give_the_e_cross_b_drift() -> None:
    none = MonteCarloCollisions({"e": M_E}, [], {}, seed=1)
    electrons = ParticleArrays.maxwellian(500, M_E, 0.0, seed=5)
    field, magnetic = 1.0e3, 0.01
    reactor = ZeroDReactor(
        none,
        {"e": electrons},
        electric_field=(field, 0.0, 0.0),
        magnetic_field=(0.0, 0.0, magnetic),
        charges={"e": -1},
    )
    assert reactor.magnetic_field == (0.0, 0.0, magnetic)
    period = 2.0 * math.pi * M_E / (ELEMENTARY_CHARGE * magnetic)
    history = reactor.run(period / 100, 1000)
    mean = history.mean_velocity["e"][1:].mean(axis=0)
    assert mean[1] == pytest.approx(-field / magnetic, rel=0.01)
    assert abs(mean[0]) < 0.01 * field / magnetic


def test_rf_field_is_evaluated_at_mid_step() -> None:
    none = MonteCarloCollisions({"e": M_E}, [], {}, seed=1)
    electrons = ParticleArrays.maxwellian(10, M_E, 0.0, seed=6)
    times = []

    def field(t: float) -> tuple[float, float, float]:
        times.append(t)
        return (1.0, 0.0, 0.0)

    reactor = ZeroDReactor(
        none, {"e": electrons}, electric_field=field, charges={"e": -1}
    )
    assert reactor.electric_field is field
    reactor.run(1.0e-9, 3)
    np.testing.assert_allclose(times, [0.5e-9, 1.5e-9, 2.5e-9])
    np.testing.assert_allclose(
        _host(electrons.velocities[:, 0]), -ELEMENTARY_CHARGE * 3.0e-9 / M_E, rtol=1e-12
    )
    with pytest.raises(ValueError, match="three components"):
        ZeroDReactor(
            none, {"e": electrons}, electric_field=lambda t: (1.0,), charges={"e": -1}
        ).step(1e-9)


def test_rate_and_townsend_coefficients_from_the_counts() -> None:
    reactor = ZeroDReactor(
        _maxwell(),
        {"He+": ParticleArrays.maxwellian(5000, M_HE, 300.0, seed=7)},
        electric_field=(1.0e3, 0.0, 0.0),
        charges={"He+": 1},
    )
    history = reactor.run(0.01 / NU, 1500, every=50)
    safety = 1.02  # the bound safety: the null-collision step error is (1 - e^-x) / x
    x = safety * 0.01
    expected = RATE * (1.0 - math.exp(-x)) / x
    assert history.rate_coefficient("elastic:He", "He+", "He") == pytest.approx(
        expected, rel=0.02
    )
    drift = ELEMENTARY_CHARGE * 1.0e3 / (M_HE / 2.0 * NU)
    townsend = history.townsend_coefficient(
        "elastic:He", "He+", "He", start_time=10.0 / NU
    )
    assert townsend == pytest.approx(expected * GAS / drift, rel=0.05)
    with pytest.raises(ValueError, match="two records"):
        history.rate_coefficient("elastic:He", "He+", "He", start_time=1.0e3)


def test_rate_coefficient_edge_cases() -> None:
    reactor = ZeroDReactor(_maxwell(), {"He+": ParticleArrays.empty()})
    history = reactor.run(1.0e-9, 2)
    assert history.rate_coefficient("elastic:He", "He+", "He") == 0.0
    with pytest.raises(ValueError, match="does not drift"):
        history.townsend_coefficient("elastic:He", "He+", "He")


def test_energy_distribution_is_normalized() -> None:
    electrons = ParticleArrays.maxwellian(50_000, M_E, 11604.5, seed=8)  # 1 eV
    reactor = ZeroDReactor(operator([], {"e": M_E}), {"e": electrons})
    centres, f = reactor.energy_distribution("e", bins=200, max_energy=15.0)
    widths = np.diff(np.linspace(0.0, 15.0, 201))
    assert float(np.sum(f * widths)) == pytest.approx(1.0, rel=1e-6)
    mean = float(np.sum(centres * f * widths))
    assert mean == pytest.approx(1.5, rel=0.02)
    edges = [0.0, 1.0, 2.0, 100.0]
    _, f_edges = reactor.energy_distribution("e", bins=edges)
    assert f_edges.shape == (3,)
    centres, f = reactor.energy_distribution("e", bins=10)
    assert centres.shape == (10,)
    empty = ZeroDReactor(operator([], {"e": M_E, "x": M_E}), {"e": electrons})
    assert not np.any(empty.energy_distribution("x", bins=5)[1])


def test_reactor_with_charged_collisions() -> None:
    masses = {"H-": M_HE, "H+": M_HE}
    reaction = ChargedReaction(
        "mutual_neutralization", ("H-", "H+"), rate_coefficient=1.0e-13
    )
    charged = ChargedCollisions(
        masses, {"H-": -1, "H+": 1}, coulomb_pairs="all", reactions=[reaction], seed=9
    )
    none = MonteCarloCollisions(masses, [], {}, seed=1)
    volume = 1.0e-6
    species = {
        "H-": ParticleArrays.maxwellian(
            5000, M_HE, 1000.0, weight=1.0e16 * volume / 5000, seed=10
        ),
        "H+": ParticleArrays.maxwellian(
            5000, M_HE, 1000.0, weight=1.0e16 * volume / 5000, seed=11
        ),
    }
    reactor = ZeroDReactor(
        none, species, volume=volume, charged=charged, electric_field=(1.0, 0, 0)
    )
    history = reactor.run(0.01 / (1.0e-13 * 1.0e16), 50, every=10)
    expected = 1.0e16 / (1.0 + 1.0e-13 * 1.0e16 * history.time[-1])
    assert history.density["H-"][-1] == pytest.approx(expected, rel=0.03)
    assert history.collisions[reaction.name][-1] > 0
    assert history.rate_coefficient(reaction.name, "H-", "H+") == pytest.approx(
        1.0e-13, rel=0.05
    )


def test_reactor_validates_magnetic_field() -> None:
    with pytest.raises(ValueError, match="magnetic_field"):
        ZeroDReactor(_maxwell(), {}, magnetic_field=(0.0, math.inf, 0.0))


# --------------------------------------------------------------------- #
# Fits
# --------------------------------------------------------------------- #
def test_log_polynomial_fits() -> None:
    # ln y = 1 + 2 ln x  ->  y = e x^2
    np.testing.assert_allclose(
        _host(log_polynomial([1.0, 2.0], xp.asarray([1.0, 2.0, 10.0]))),
        math.e * np.array([1.0, 4.0, 100.0]),
    )
    assert float(_host(log_polynomial([0.0], 5.0, unit=CM3_PER_S))) == pytest.approx(
        1e-6
    )
    with pytest.raises(ValueError, match="x > 0"):
        log_polynomial([1.0], 0.0)
    # ln y = a00 + a10 ln E + a01 ln T + a11 ln E ln T
    table = [[0.5, 1.0], [2.0, 0.25]]
    energy, temperature = 3.0, 7.0
    expected = math.exp(
        0.5
        + 1.0 * math.log(temperature)
        + 2.0 * math.log(energy)
        + 0.25 * math.log(energy) * math.log(temperature)
    )
    assert float(
        _host(double_log_polynomial(table, energy, temperature))
    ) == pytest.approx(expected)
    flat: Any = [1.0, 2.0]
    with pytest.raises(ValueError, match="2D"):
        double_log_polynomial(flat, 1.0, 1.0)
    with pytest.raises(ValueError, match="positive"):
        double_log_polynomial(table, -1.0, 1.0)


def test_cross_section_from_a_fit() -> None:
    table = cross_section_from_log_polynomial(
        [math.log(2.0)], energy_range=(1.0, 100.0), num=10, threshold=0.5, label="fit"
    )
    np.testing.assert_allclose(
        _host(table(xp.asarray([0.1, 1.0, 50.0, 1e4]))),
        [0.0, 2.0 * CM2, 2.0 * CM2, 2.0 * CM2],
    )
    assert table.label == "fit"
    assert (
        cross_section_from_log_polynomial([0.0], (1.0, 2.0)).label
        == "log-polynomial fit"
    )
    with pytest.raises(ValueError, match="energy_range"):
        cross_section_from_log_polynomial([0.0], (2.0, 1.0))
