"""Physics and bookkeeping of the null-collision Monte Carlo collision operator."""

from __future__ import annotations

import math
import warnings

import cunumpy as xp
import numpy as np
import pytest

from pymcc import (
    CollisionProcess,
    CrossSection,
    DensityProfile,
    MonteCarloCollisions,
    NeutralBackground,
    ParticleArrays,
    make_rng,
)
from pymcc.constants import BOLTZMANN

from ._helpers import (
    HEAVY,
    M_E,
    M_HE,
    M_P,
    beam,
    constant,
    energy_ev,
    operator,
    speed_for,
)


def _ones(num: int):
    return xp.ones(num)


def _origin(num: int):
    return xp.zeros((num, 1))


# --------------------------------------------------------------------- #
# Collision rates
# --------------------------------------------------------------------- #
def test_collision_probability_matches_null_collision_rate() -> None:
    """The fraction of markers that collide per step is (1 - exp(-nu_max dt)) nu / nu_max."""
    density, sigma, energy = 1.0e20, 1.0e-19, 10.0
    mcc = operator(
        [CollisionProcess("elastic", "gas", constant(sigma))],
        {"e": M_E},
        density=density,
        bound_safety=1.02,
    )
    num = 200_000
    speed = speed_for(energy, M_E)
    dt = 0.1 / (density * sigma * speed)

    result = mcc.collide("e", _origin(num), beam(num, energy, M_E), _ones(num), dt)

    nu = density * sigma * speed
    nu_max = mcc.frequency_bound("e", speed)
    expected = (1.0 - math.exp(-nu_max * dt)) * nu / nu_max
    collided = result.diagnostics.counts["elastic:gas"] / num
    tolerance = 5.0 * math.sqrt(expected * (1.0 - expected) / num)
    assert collided == pytest.approx(expected, abs=tolerance)
    assert result.diagnostics.bound_violations == 0
    diagnostics = result.diagnostics
    assert (
        diagnostics.candidates
        == diagnostics.null_collisions + diagnostics.real_collisions
    )


def test_null_collisions_reject_in_proportion_to_frequency() -> None:
    """With a two-level cross section, slow markers collide less often."""
    table = CrossSection(
        energy=[0.0, 5.0, 5.0 + 1e-9, 100.0], sigma=[1.0e-20, 1.0e-20, 4.0e-20, 4.0e-20]
    )
    density = 1.0e24
    mcc = operator(
        [CollisionProcess("elastic", "gas", table)], {"e": M_E}, density=density
    )
    num = 100_000
    slow_speed, fast_speed = speed_for(1.0, M_E), speed_for(16.0, M_E)
    velocities = xp.concatenate([beam(num, 1.0, M_E), beam(num, 16.0, M_E)])
    dt = 5.0e-12

    result = mcc.collide("e", _origin(2 * num), velocities, _ones(2 * num), dt)

    # Elastic collisions on an infinitely heavy target only rotate the
    # velocity, so the collided markers are the ones that left the x axis.
    initial = xp.concatenate([xp.full(num, slow_speed), xp.full(num, fast_speed)])
    rotated = xp.abs(result.velocities[:, 0] - initial) > 1e-6 * initial
    slow_rate = float(xp.mean(rotated[:num]))
    fast_rate = float(xp.mean(rotated[num:]))
    nu_max = mcc.frequency_bound("e", fast_speed)
    probability = 1.0 - math.exp(-nu_max * dt)
    expected_slow = probability * density * 1.0e-20 * slow_speed / nu_max
    expected_fast = probability * density * 4.0e-20 * fast_speed / nu_max
    for rate, expected in ((slow_rate, expected_slow), (fast_rate, expected_fast)):
        tolerance = 5.0 * math.sqrt(expected * (1.0 - expected) / num)
        assert rate == pytest.approx(expected, abs=tolerance)


def test_frequency_bound_covers_all_speeds() -> None:
    """nu_max bounds the total frequency, including beyond the table end."""
    table = CrossSection(
        energy=[0.0, 2.0, 3.0, 50.0], sigma=[1.0e-20, 8.0e-20, 1.0e-20, 3.0e-20]
    )
    mcc = operator(
        [CollisionProcess("excitation", "gas", table.with_threshold(1.0))], {"e": M_E}
    )
    speeds = xp.asarray(np.geomspace(1.0, 2.0 * speed_for(50.0, M_E), 10_000))
    frequencies = mcc.total_frequency("e", speeds)
    bound = mcc.frequency_bound("e", float(speeds[-1]))
    assert float(xp.max(frequencies)) <= bound * (1.0 + 1e-12)
    assert mcc.frequency_bound("e", 0.0) == 0.0
    assert mcc.frequency_bound("no such species", 1.0e6) == 0.0


def test_collision_frequency_follows_the_density_profile() -> None:
    """Markers collide in proportion to the local density; none where it is 0."""
    profile = DensityProfile(axis=0, positions=[0.0, 1.0, 2.0], factors=[0.0, 0.5, 1.0])
    mcc = MonteCarloCollisions(
        species_masses={"H+": M_P},
        backgrounds=[NeutralBackground("H2", 1.0e20, 0.0, 2 * M_P, profile=profile)],
        processes={
            "H+": [CollisionProcess("charge_transfer", "H2", constant(1.0e-19))]
        },
        seed=3,
    )
    num = 300_000
    positions = xp.zeros((num, 1))
    positions[num // 3 : 2 * num // 3, 0] = 1.0  # factor 0.5
    positions[2 * num // 3 :, 0] = 2.0  # factor 1
    velocities = xp.zeros((num, 3))
    velocities[:, 0] = 1.0e4  # cold neutrals: relative speed 1e4 m/s
    # nu = 1e20 * 1e-19 * 1e4 = 1e5 /s at factor 1; dt gives P ~ 5 %
    result = mcc.collide("H+", positions, velocities, _ones(num), dt=5.0e-7)
    removed = xp.to_numpy(result.removed)
    thirds = np.array_split(removed, 3)
    expected = (1.0 - math.exp(-1.02e5 * 5.0e-7)) / 1.02
    assert thirds[0].sum() == 0
    assert thirds[2].mean() == pytest.approx(expected, rel=0.05)
    assert thirds[1].mean() == pytest.approx(0.5 * expected, rel=0.08)


# --------------------------------------------------------------------- #
# Kinematics
# --------------------------------------------------------------------- #
def test_elastic_scattering_on_heavy_target_is_isotropic_and_elastic() -> None:
    mcc = operator([CollisionProcess("elastic", "gas", constant(1.0e-18))], {"e": M_E})
    num = 50_000
    result = mcc.collide("e", _origin(num), beam(num, 10.0, M_E), _ones(num), dt=1.0)

    assert result.diagnostics.counts["elastic:gas"] == num
    speed = xp.sqrt(xp.sum(result.velocities**2, axis=1))
    assert bool(xp.allclose(speed, speed_for(10.0, M_E), rtol=1e-6))
    mean_direction = xp.mean(result.velocities / speed[:, None], axis=0)
    assert float(xp.max(xp.abs(mean_direction))) < 5.0 / math.sqrt(num)


def test_elastic_scattering_keeps_a_thermal_distribution_thermal() -> None:
    """Equal-mass ions at the gas temperature stay there (detailed balance)."""
    temperature = 300.0
    mcc = operator(
        [CollisionProcess("elastic", "gas", constant(1.0e-18))],
        {"ion": M_HE},
        neutral_mass=M_HE,
        temperature=temperature,
        incident="ion",
    )
    ions = ParticleArrays.maxwellian(200_000, M_HE, temperature, seed=0)
    before = float(xp.mean(energy_ev(ions.velocities, M_HE)))
    result = mcc.collide("ion", ions.positions, ions.velocities, ions.weights, dt=1.0)
    after = float(xp.mean(energy_ev(result.velocities, M_HE)))
    assert after == pytest.approx(before, rel=0.01)


def test_cold_ions_thermalize_to_the_background_temperature() -> None:
    """Repeated equal-mass elastic collisions relax cold ions to the gas temperature."""
    temperature = 300.0
    mcc = operator(
        [CollisionProcess("elastic", "gas", constant(1.0e-18))],
        {"ion": M_HE},
        neutral_mass=M_HE,
        temperature=temperature,
        incident="ion",
    )
    num = 100_000
    velocities = xp.zeros((num, 3))
    temperatures = []
    for _ in range(100):
        velocities = mcc.collide(
            "ion", _origin(num), velocities, _ones(num), dt=1.0
        ).velocities
        mean_square = float(xp.mean(xp.sum(velocities**2, axis=1)))
        temperatures.append(M_HE * mean_square / (3.0 * BOLTZMANN))
    assert temperatures[0] < temperature
    assert temperatures[-1] == pytest.approx(temperature, rel=0.02)
    assert max(temperatures[-10:]) == pytest.approx(temperature, rel=0.03)


def test_excitation_removes_exactly_the_threshold_energy() -> None:
    threshold = 19.82
    mcc = operator(
        [CollisionProcess("excitation", "gas", constant(1.0e-18, threshold=threshold))],
        {"e": M_E},
    )
    num = 1000
    result = mcc.collide("e", _origin(num), beam(num, 30.0, M_E), _ones(num), dt=1.0)
    assert bool(xp.allclose(energy_ev(result.velocities, M_E), 30.0 - threshold))


def test_excitation_is_impossible_below_threshold() -> None:
    mcc = operator(
        [CollisionProcess("excitation", "gas", constant(1.0e-18, threshold=20.0))],
        {"e": M_E},
    )
    num = 1000
    velocities = beam(num, 19.0, M_E)
    result = mcc.collide("e", _origin(num), velocities, _ones(num), dt=1.0)
    assert result.diagnostics.real_collisions == 0
    assert bool(xp.allclose(result.velocities, velocities))


def test_processes_out_of_reach_are_counted_as_zero() -> None:
    mcc = operator(
        [
            CollisionProcess("elastic", "gas", constant(1.0e-18)),
            CollisionProcess("excitation", "gas", constant(1.0e-18, threshold=50.0)),
        ],
        {"e": M_E},
    )
    result = mcc.collide("e", _origin(100), beam(100, 10.0, M_E), _ones(100), dt=1.0)
    assert result.diagnostics.counts == {"elastic:gas": 100, "excitation:gas": 0}


def test_ionization_shares_residual_energy_and_creates_pair() -> None:
    threshold = 24.59
    mcc = operator(
        [
            CollisionProcess(
                "ionization",
                "gas",
                constant(1.0e-18, threshold=threshold),
                products={"electron": "e", "ion": "ion"},
            )
        ],
        {"e": M_E, "ion": M_HE},
    )
    num = 500
    positions = xp.linspace(0.0, 1.0, num)[:, None]
    weights = xp.full(num, 7.0)
    result = mcc.collide("e", positions, beam(num, 100.0, M_E), weights, dt=1.0)

    residual = 100.0 - threshold
    assert bool(xp.allclose(energy_ev(result.velocities, M_E), residual / 2.0))
    (secondary,) = result.created["e"]
    (ion,) = result.created["ion"]
    assert bool(xp.allclose(energy_ev(secondary.velocities, M_E), residual / 2.0))
    assert bool(xp.allclose(ion.velocities, 0.0))
    assert bool(xp.allclose(secondary.positions, positions))
    assert bool(xp.allclose(ion.positions, positions))
    assert bool(xp.allclose(secondary.weights, 7.0))
    assert bool(xp.allclose(ion.weights, 7.0))
    assert not bool(xp.any(result.removed))


def test_uniform_energy_sharing_conserves_residual_energy() -> None:
    mcc = operator(
        [
            CollisionProcess(
                "ionization",
                "gas",
                constant(1.0e-18, threshold=10.0),
                products={"electron": "e", "ion": "ion"},
                energy_sharing="uniform",
            )
        ],
        {"e": M_E, "ion": M_HE},
    )
    num = 500
    result = mcc.collide("e", _origin(num), beam(num, 50.0, M_E), _ones(num), dt=1.0)
    secondary = result.created["e"][0].velocities
    total = energy_ev(result.velocities, M_E) + energy_ev(secondary, M_E)
    assert bool(xp.allclose(total, 40.0))
    assert float(xp.std(energy_ev(secondary, M_E))) > 1.0


def test_backscatter_gives_ion_the_neutral_velocity() -> None:
    mcc = operator(
        [CollisionProcess("backscatter", "gas", constant(1.0e-18))],
        {"ion": M_HE},
        neutral_mass=M_HE,
        incident="ion",
    )
    num = 100
    result = mcc.collide("ion", _origin(num), beam(num, 50.0, M_HE), _ones(num), dt=1.0)
    assert bool(xp.allclose(result.velocities, 0.0, atol=1e-9))


def test_attachment_turns_electrons_into_negative_ions() -> None:
    mcc = operator(
        [
            CollisionProcess(
                "attachment", "gas", constant(1.0e-18), products={"negative_ion": "H-"}
            )
        ],
        {"e": M_E, "H-": M_P},
    )
    num = 100
    result = mcc.collide("e", _origin(num), beam(num, 5.0, M_E), _ones(num), dt=1.0)
    assert bool(xp.all(result.removed))
    assert result.created["H-"][0].velocities.shape == (num, 3)


def test_detachment_removes_negative_ion_and_emits_electron() -> None:
    """Detachment emits an electron with the residual centre-of-mass energy."""
    threshold = 0.75
    mcc = operator(
        [
            CollisionProcess(
                "detachment",
                "gas",
                constant(1.0e-18, threshold=threshold),
                products={"electron": "e"},
            )
        ],
        {"H-": M_P, "e": M_E},
        neutral_mass=M_P,
        incident="H-",
    )
    num = 100
    velocities = beam(num, 10.0, M_P)
    result = mcc.collide("H-", _origin(num), velocities, _ones(num), dt=1.0)

    assert bool(xp.all(result.removed))
    electrons = result.created["e"][0].velocities
    center_of_mass = velocities / 2.0
    residual = 10.0 / 2.0 - threshold  # centre-of-mass energy of equal masses
    assert bool(xp.allclose(energy_ev(electrons - center_of_mass, M_E), residual))


def test_charge_transfer_removes_the_incident_and_creates_the_ion() -> None:
    """H+ + H2 -> H + H2+: the proton goes, H2+ appears with the molecule's velocity."""
    mcc = MonteCarloCollisions(
        species_masses={"H+": M_P, "H2+": 2 * M_P},
        backgrounds=[NeutralBackground("H2", 1.0e22, 0.0, 2 * M_P)],
        processes={
            "H+": [
                CollisionProcess(
                    "charge_transfer",
                    "H2",
                    constant(1.0e-17),
                    "cx_h2",
                    products={"ion": "H2+"},
                )
            ]
        },
        seed=1,
    )
    num = 1000
    velocities = xp.zeros((num, 3))
    velocities[:, 2] = 3.0e4
    result = mcc.collide("H+", _origin(num), velocities, xp.full(num, 7.0), dt=1.0e-3)
    num_removed = int(xp.count_nonzero(result.removed))
    # the bound safety factor leaves a percent or so of candidates null
    assert num_removed == result.diagnostics.counts["cx_h2"] > 0.95 * num
    (ion,) = result.created["H2+"]
    assert ion.velocities.shape == (num_removed, 3)
    assert bool(xp.all(ion.velocities == 0.0))  # a cold molecule is at rest
    assert bool(xp.all(ion.weights == 7.0))
    survivors = result.velocities[~result.removed, 2]
    assert float(xp.max(xp.abs(survivors - 3.0e4))) == 0.0


def test_charge_transfer_without_ion_species_just_removes() -> None:
    mcc = MonteCarloCollisions(
        species_masses={"H+": M_P},
        backgrounds=[NeutralBackground("H2", 1.0e22, 0.0, 2 * M_P)],
        processes={
            "H+": [CollisionProcess("charge_transfer", "H2", constant(1e-17), "cx")]
        },
        seed=1,
    )
    velocities = xp.zeros((10, 3))
    velocities[:, 0] = 1.0e4
    result = mcc.collide("H+", _origin(10), velocities, _ones(10), 1e-3)
    assert int(xp.count_nonzero(result.removed)) == result.diagnostics.counts["cx"] > 0
    assert result.created == {}


def test_batched_processes_attribute_the_correct_background(monkeypatch) -> None:
    """A marker colliding on background B must not use background A's neutral.

    Candidate selection batches every process's relative velocity into one
    array indexed by background; a wrong index would swap which neutral a
    marker collides against. ``backscatter`` is deterministic, so the outcome
    pins down exactly which neutral velocity was used.
    """
    calls: list[str] = []

    def fake_sample_velocities(self, rng, num):
        del rng
        calls.append(self.name)
        return xp.full((num, 3), 100.0 if self.name == "A" else -100.0)

    monkeypatch.setattr(NeutralBackground, "sample_velocities", fake_sample_velocities)
    mcc = MonteCarloCollisions(
        species_masses={"ion": M_HE},
        backgrounds=[
            NeutralBackground(name="A", density=1.0e20, temperature=0.0, mass=M_HE),
            NeutralBackground(name="B", density=1.0e20, temperature=0.0, mass=M_HE),
        ],
        processes={
            "ion": [
                CollisionProcess("backscatter", "A", constant(1.0e-18), name="on_A"),
                CollisionProcess("backscatter", "B", constant(1.0e-18), name="on_B"),
            ],
        },
        seed=7,
    )
    num = 4000
    initial = beam(num, 50.0, M_HE)
    result = mcc.collide("ion", _origin(num), initial, _ones(num), dt=1.0)

    assert result.diagnostics.bound_violations == 0
    assert set(calls) == {"A", "B"}
    counts = result.diagnostics.counts
    assert counts["on_A"] > 0
    assert counts["on_B"] > 0
    collided = ~xp.isclose(result.velocities[:, 0], initial[0, 0])
    on_a = xp.all(xp.isclose(result.velocities, 100.0), axis=1)
    on_b = xp.all(xp.isclose(result.velocities, -100.0), axis=1)
    assert int(xp.count_nonzero(on_a)) == counts["on_A"]
    assert int(xp.count_nonzero(on_b)) == counts["on_B"]
    assert bool(xp.all(on_a | on_b | ~collided))


# --------------------------------------------------------------------- #
# Reproducibility, arguments and bookkeeping
# --------------------------------------------------------------------- #
def _two_process_operator(seed: int = 0, rank: int = 0) -> MonteCarloCollisions:
    return MonteCarloCollisions(
        species_masses={"e": M_E},
        backgrounds=[NeutralBackground("gas", 1.0e20, 300.0, M_HE)],
        processes={
            "e": [
                CollisionProcess("elastic", "gas", constant(1.0e-19), name="el"),
                CollisionProcess(
                    "excitation", "gas", constant(0.5e-19, threshold=5.0), name="exc"
                ),
            ]
        },
        seed=seed,
        rank=rank,
    )


def test_same_seed_and_rank_give_the_same_collisions() -> None:
    num = 5000
    velocities = xp.asarray(np.random.default_rng(3).normal(0.0, 2.0e6, (num, 3)))
    first = _two_process_operator(7).collide(
        "e", _origin(num), velocities, _ones(num), 1e-9
    )
    second = _two_process_operator(7).collide(
        "e", _origin(num), velocities, _ones(num), 1e-9
    )
    other_rank = _two_process_operator(7, rank=1).collide(
        "e", _origin(num), velocities, _ones(num), 1e-9
    )
    assert first.diagnostics.counts == second.diagnostics.counts
    assert bool(xp.array_equal(first.velocities, second.velocities))
    assert not bool(xp.allclose(first.velocities, other_rank.velocities))


def test_make_rng_streams_differ_by_rank() -> None:
    assert float(make_rng(1, 0).random()) == float(make_rng(1, 0).random())
    assert float(make_rng(1, 0).random()) != float(make_rng(1, 1).random())


class _NonNumpyGenerator:
    """A generator that is not a ``numpy.random.Generator``, like CuPy's."""

    def __init__(self, seed: int) -> None:
        self._rng = make_rng(seed)

    def random(self, size):
        return self._rng.random(size)

    def standard_normal(self, size):
        return self._rng.standard_normal(size)


def test_generators_without_binomial_draw_one_number_per_marker() -> None:
    """The per-marker candidate draw (CuPy's path) gives the same rate."""
    density, sigma, energy = 1.0e20, 1.0e-19, 10.0
    speed = speed_for(energy, M_E)
    dt = 0.1 / (density * sigma * speed)
    mcc = MonteCarloCollisions(
        species_masses={"e": M_E},
        backgrounds=[NeutralBackground("gas", density, 0.0, HEAVY)],
        processes={"e": [CollisionProcess("elastic", "gas", constant(sigma))]},
        rng=_NonNumpyGenerator(5),
        bound_safety=1.0,
    )
    num = 200_000
    alive = xp.ones(num, dtype=bool)
    alive[: num // 2] = False
    result = mcc.collide(
        "e", _origin(num), beam(num, energy, M_E), _ones(num), dt, alive=alive
    )
    expected = (1.0 - math.exp(-0.1)) * num / 2
    assert result.diagnostics.counts["elastic:gas"] == pytest.approx(expected, rel=0.03)
    assert bool(xp.all(result.velocities[: num // 2, 0] == speed))


def test_only_alive_markers_collide() -> None:
    mcc = operator([CollisionProcess("elastic", "gas", constant(1.0e-18))], {"e": M_E})
    num = 1000
    velocities = beam(num, 10.0, M_E)
    alive = xp.zeros(num, dtype=bool)
    alive[::2] = True
    result = mcc.collide("e", _origin(num), velocities, _ones(num), dt=1.0, alive=alive)
    assert result.diagnostics.counts["elastic:gas"] == num // 2
    assert bool(xp.allclose(result.velocities[1::2], velocities[1::2]))


def test_in_place_updates_the_given_velocities() -> None:
    mcc = operator([CollisionProcess("elastic", "gas", constant(1.0e-18))], {"e": M_E})
    velocities = beam(100, 10.0, M_E)
    copy = mcc.collide("e", _origin(100), velocities, _ones(100), dt=1.0)
    assert copy.velocities is not velocities
    assert bool(xp.all(velocities[:, 1] == 0.0))
    in_place = mcc.collide(
        "e", _origin(100), velocities, _ones(100), dt=1.0, in_place=True
    )
    assert in_place.velocities is velocities
    assert not bool(xp.all(velocities[:, 1] == 0.0))


@pytest.mark.parametrize(
    ("species", "num", "dt", "density"),
    [
        ("e", 0, 1.0, 1e20),
        ("e", 10, 0.0, 1e20),
        ("other", 10, 1.0, 1e20),
        ("e", 10, 1.0, 0.0),
    ],
)
def test_nothing_happens_without_markers_time_processes_or_gas(
    species: str, num: int, dt: float, density: float
) -> None:
    mcc = operator(
        [CollisionProcess("elastic", "gas", constant(1.0e-18))],
        {"e": M_E, "other": M_E},
        density=density,
    )
    velocities = beam(num, 10.0, M_E)
    result = mcc.collide(species, _origin(num), velocities, _ones(num), dt)
    assert result.diagnostics.candidates == 0
    assert bool(xp.array_equal(result.velocities, velocities))
    assert not bool(xp.any(result.removed))


def test_tiny_probability_can_leave_no_candidates() -> None:
    mcc = operator([CollisionProcess("elastic", "gas", constant(1.0e-18))], {"e": M_E})
    result = mcc.collide("e", _origin(10), beam(10, 10.0, M_E), _ones(10), dt=1.0e-30)
    assert result.diagnostics.candidates == 0


def test_collisions_need_three_velocity_components() -> None:
    mcc = operator([CollisionProcess("elastic", "gas", constant(1.0e-18))], {"e": M_E})
    with pytest.raises(ValueError, match="three velocity components"):
        mcc.collide("e", _origin(4), xp.ones((4, 1)), _ones(4), dt=1.0)


def test_exceeding_the_frequency_bound_warns_once(monkeypatch) -> None:
    mcc = operator([CollisionProcess("elastic", "gas", constant(1.0e-18))], {"e": M_E})
    monkeypatch.setattr(mcc, "frequency_bound", lambda species, speed: 1.0e3)
    velocities = beam(1000, 10.0, M_E)
    with pytest.warns(RuntimeWarning, match="frequency bound"):
        result = mcc.collide("e", _origin(1000), velocities, _ones(1000), dt=1.0)
    assert result.diagnostics.bound_violations > 0
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        again = mcc.collide("e", _origin(1000), velocities, _ones(1000), dt=1.0)
    assert again.diagnostics.bound_violations > 0
    assert record == []


def test_operator_validates_its_definition() -> None:
    gas = [NeutralBackground("gas", 1.0e20, 300.0, M_HE)]
    elastic = CollisionProcess("elastic", "gas", constant(1.0e-19))
    with pytest.raises(KeyError, match="colliding species"):
        MonteCarloCollisions({}, gas, {"e": [elastic]})
    with pytest.raises(KeyError, match="unknown background"):
        MonteCarloCollisions(
            {"e": M_E}, gas, {"e": [CollisionProcess("elastic", "Ar", constant(1))]}
        )
    ionization = CollisionProcess(
        "ionization",
        "gas",
        constant(1e-20, 10.0),
        products={"electron": "e", "ion": "He+"},
    )
    with pytest.raises(KeyError, match="product"):
        MonteCarloCollisions({"e": M_E}, gas, {"e": [ionization]})
    with pytest.raises(ValueError, match="unique"):
        MonteCarloCollisions({"e": M_E}, gas, {"e": [elastic, elastic]})
    with pytest.raises(ValueError, match="> 0"):
        MonteCarloCollisions({"e": 0.0}, gas, {"e": [elastic]})
    with pytest.raises(ValueError, match="bound_safety"):
        MonteCarloCollisions({"e": M_E}, gas, {"e": [elastic]}, bound_safety=0.5)


def test_operator_properties() -> None:
    mcc = _two_process_operator()
    assert list(mcc.processes) == ["e"]
    assert [p.name for p in mcc.processes["e"]] == ["el", "exc"]
    assert list(mcc.backgrounds) == ["gas"]
    assert mcc.species_masses == {"e": M_E}
    assert mcc.rng is not None
    empty = MonteCarloCollisions({"e": M_E}, [], {"e": []})
    assert empty.frequency_bound("e", 1.0e6) == 0.0


# --------------------------------------------------------------------- #
# Marker sets
# --------------------------------------------------------------------- #
def test_collide_species_adds_products_and_removes_consumed_markers() -> None:
    electrons = ParticleArrays(
        xp.full((200, 1), 0.5), beam(200, 100.0, M_E), xp.full(200, 3.0)
    )
    ions = ParticleArrays.empty()
    mcc = MonteCarloCollisions(
        species_masses={"e": M_E, "He+": M_HE},
        backgrounds=[NeutralBackground("He", 1.0e20, 300.0, M_HE)],
        processes={
            "e": [
                CollisionProcess(
                    "ionization",
                    "He",
                    constant(1.0e-18, threshold=24.59),
                    products={"electron": "e", "ion": "He+"},
                    energy_frame="lab",
                )
            ]
        },
        seed=3,
        bound_safety=1.0,
    )
    diagnostics = mcc.collide_species({"e": electrons, "He+": ions}, dt=1.0)
    assert diagnostics["e"].counts["ionization:He"] == 200
    assert len(electrons) == 400
    assert len(ions) == 200
    assert bool(xp.allclose(ions.weights, 3.0))
    assert bool(xp.allclose(ions.positions, 0.5))


def test_collide_species_removes_attached_electrons() -> None:
    electrons = ParticleArrays(xp.zeros((50, 1)), beam(50, 2.0, M_E), xp.ones(50))
    negative_ions = ParticleArrays.empty()
    mcc = MonteCarloCollisions(
        species_masses={"e": M_E, "H-": M_P},
        backgrounds=[NeutralBackground("H2", 1.0e20, 0.0, 2 * M_P)],
        processes={
            "e": [
                CollisionProcess(
                    "attachment",
                    "H2",
                    constant(1.0e-18),
                    products={"negative_ion": "H-"},
                )
            ]
        },
        seed=0,
        bound_safety=1.0,
    )
    mcc.collide_species(
        {"e": electrons, "H-": negative_ions, "unused": electrons}, dt=1.0
    )
    assert len(electrons) == 0
    assert len(negative_ions) == 50
    assert mcc.collide_species({"H-": negative_ions}, dt=1.0) == {}


def test_collide_species_needs_the_product_species() -> None:
    electrons = ParticleArrays(xp.zeros((50, 1)), beam(50, 2.0, M_E), xp.ones(50))
    mcc = MonteCarloCollisions(
        species_masses={"e": M_E, "H-": M_P},
        backgrounds=[NeutralBackground("H2", 1.0e20, 0.0, 2 * M_P)],
        processes={
            "e": [
                CollisionProcess(
                    "attachment",
                    "H2",
                    constant(1.0e-18),
                    products={"negative_ion": "H-"},
                )
            ]
        },
        seed=0,
    )
    with pytest.raises(KeyError, match="H-"):
        mcc.collide_species({"e": electrons}, dt=1.0)


def test_collide_species_respects_an_alive_mask() -> None:
    class Markers(ParticleArrays):
        @property
        def alive(self):
            mask = xp.zeros(len(self), dtype=bool)
            mask[:10] = True
            return mask

    electrons = Markers(xp.zeros((100, 1)), beam(100, 10.0, M_E), xp.ones(100))
    mcc = operator([CollisionProcess("elastic", "gas", constant(1.0e-18))], {"e": M_E})
    diagnostics = mcc.collide_species({"e": electrons}, dt=1.0)
    assert diagnostics["e"].counts["elastic:gas"] == 10
