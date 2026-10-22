"""The two-term Boltzmann solver against closed forms, published model-gas results and BOLOS.

The cross sections are model gases or synthetic, not physical data.
"""

from __future__ import annotations

import importlib
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from plasmacoll import (
    CollisionProcess,
    CrossSection,
    NeutralBackground,
    read_lxcat,
    solve_eedf,
    swarm_table,
)
from plasmacoll.boltzmann import GAMMA, _build, _log_slope, _matrix, _moments
from plasmacoll.constants import (
    ATOMIC_MASS,
    ELECTRON_MASS,
    TOWNSEND,
    kelvin_to_ev,
)

M_HE = 4.0 * ATOMIC_MASS
IONIZATION_PRODUCTS = {"electron": "e", "ion": "ion"}


def gas(mass: float = M_HE, temperature: float = 0.0, name: str = "gas"):
    return NeutralBackground(name, 1.0e22, temperature, mass)


def elastic(sigma: CrossSection, background: str = "gas") -> CollisionProcess:
    return CollisionProcess("elastic", background, sigma, name=f"elastic {background}")


def ramp(slope: float, threshold: float) -> CrossSection:
    """``slope (eps - threshold)`` in m^2 above ``threshold``."""
    return CrossSection(
        energy=[threshold, threshold + 1.0e3],
        sigma=[0.0, slope * 1.0e3],
        threshold=threshold,
    )


def normalised(eepf: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return eepf / np.sum(eepf * 2.0 / 3.0 * np.diff(edges**1.5))


# Reid's ramp model gas, Aust. J. Phys. 32, 231 (1979): a constant elastic cross
# section of 6 A^2 and an excitation of 10 (eps - 0.2) A^2/eV, M = 4 u, T = 0.
REID = [
    elastic(CrossSection.constant(6.0e-20)),
    CollisionProcess("excitation", "gas", ramp(10.0e-20, 0.2), name="excitation"),
]

# The Lucas-Saelee ionization model gas with F = 1, as restated by Flynn et al.,
# J. Phys. D 57, 255204 (2024), eq. (28): sigma_m = 4 eps^-1/2 A^2, an ionization
# of 0.1 (eps - 15.6) A^2/eV, m/M = 1e-3, T = 0.
LUCAS_SAELEE_MASS = ELECTRON_MASS / 1.0e-3


def lucas_saelee(sharing: str = "equal") -> list[CollisionProcess]:
    momentum = CrossSection.from_function(
        lambda energy: 4.0e-20 / np.sqrt(energy), np.geomspace(1.0e-6, 1.0e4, 2000)
    )
    return [
        elastic(momentum),
        CollisionProcess(
            "ionization",
            "gas",
            ramp(0.1e-20, 15.6),
            name="ionization",
            products=IONIZATION_PRODUCTS,
            energy_sharing=sharing,
            sharing_energy=10.0 if sharing == "opal" else None,
        ),
    ]


# --- closed forms -----------------------------------------------------------------


def test_a_constant_collision_frequency_gives_a_maxwellian() -> None:
    """sigma = K/v: F0 ~ exp(-eps/Te) with Te = kT + M (E/N)^2 gamma^2 / (6 m K^2)."""
    rate = 1.0e-13
    cross_section = CrossSection.maxwell_molecule(rate, ELECTRON_MASS, min_energy=1e-8)
    for reduced_field, temperature in ((1.0, 0.0), (3.0, 0.025)):
        solution = solve_eedf(
            [elastic(cross_section)], [gas()], reduced_field, temperature
        )
        field = reduced_field * TOWNSEND
        electron_temperature = temperature + field**2 * GAMMA**2 * M_HE / (
            6.0 * ELECTRON_MASS * rate**2
        )
        expected = normalised(
            np.exp(-solution.energy / electron_temperature), solution.energy_edges
        )
        assert solution.mean_energy == pytest.approx(
            1.5 * electron_temperature, rel=2e-3
        )
        assert solution.eepf[:100] == pytest.approx(expected[:100], rel=1e-2)
        # sigma = K / v gives mu N = e / (m K), D N = kTe / (m K)
        assert solution.mobility == pytest.approx(GAMMA**2 / (2.0 * rate), rel=1e-3)
        assert solution.characteristic_energy == pytest.approx(
            electron_temperature, rel=2e-3
        )


def test_without_a_field_the_electrons_take_the_gas_temperature() -> None:
    solution = solve_eedf([elastic(CrossSection.constant(1e-19))], [gas()], 0.0, 0.1)
    assert solution.mean_energy == pytest.approx(0.15, rel=1e-3)
    assert solution.power["field"] == 0.0
    assert math.isnan(solution.power_balance)
    # the gas heats the electrons as much as they cool on it
    cooling = GAMMA * np.sum(
        solution.energy**2
        * 2.0
        * ELECTRON_MASS
        / M_HE
        * 1e-19
        * solution.eepf
        * np.diff(solution.energy_edges)
    )
    assert abs(solution.power["elastic"]) < 1e-2 * cooling


def test_a_constant_cross_section_gives_the_druyvesteyn_distribution() -> None:
    """kT = 0: F0 ~ exp(-3 (m/M) (sigma eps / (E/N))^2)."""
    sigma = 6.0e-20
    for reduced_field in (1.0, 10.0):
        solution = solve_eedf(
            [elastic(CrossSection.constant(sigma))], [gas()], reduced_field, 0.0
        )
        field = reduced_field * TOWNSEND
        expected = normalised(
            np.exp(
                -3.0 * ELECTRON_MASS / M_HE * (sigma * solution.energy / field) ** 2
            ),
            solution.energy_edges,
        )
        visible = expected > 1e-8 * expected.max()
        assert solution.eepf[visible] == pytest.approx(expected[visible], rel=1e-8)
        # <eps> = 0.7418 (E/N)/sigma (M / 3m)^1/2 for the Druyvesteyn distribution
        druyvesteyn = (
            math.gamma(1.25)
            / math.gamma(0.75)
            * field
            / sigma
            * math.sqrt(M_HE / (3.0 * ELECTRON_MASS))
        )
        assert solution.mean_energy == pytest.approx(druyvesteyn, rel=1e-4)


# --- published two-term results for model gases -------------------------------------


@pytest.mark.parametrize(
    ("reduced_field", "mean_energy", "drift_velocity", "diffusion"),
    [
        # Flynn et al. (2024), Table 1, two-term MultiBolt, isotropic scattering
        (1.0, 0.10150, 1.2750e4, 0.9905e24),
        (12.0, 0.27350, 7.0290e4, 1.3690e24),
        (24.0, 0.41620, 9.1447e4, 1.4382e24),
        # Reid (1979), Table 4, two-term Boltzmann column
        (5.0, 0.1742, 4.36e4, 1.226e24),
        (40.0, 0.579, 1.050e5, 1.448e24),
    ],
)
def test_the_reid_ramp_model_matches_published_two_term_results(
    reduced_field: float, mean_energy: float, drift_velocity: float, diffusion: float
) -> None:
    solution = solve_eedf(REID, [gas()], reduced_field, 0.0)
    assert solution.mean_energy == pytest.approx(mean_energy, rel=2e-3)
    assert solution.drift_velocity == pytest.approx(drift_velocity, rel=2e-3)
    assert solution.diffusion == pytest.approx(diffusion, rel=2e-3)
    assert solution.net_ionization == 0.0
    assert solution.townsend_coefficient == 0.0


def test_the_lucas_saelee_model_matches_published_two_term_results() -> None:
    """Flynn et al. (2024), Table 2, two-term MultiBolt at 30 Td."""
    solution = solve_eedf(lucas_saelee(), [gas(LUCAS_SAELEE_MASS)], 30.0, 0.0)
    assert solution.mean_energy == pytest.approx(6.2840, rel=2e-3)
    assert solution.drift_velocity == pytest.approx(2.0895e5, rel=2e-3)
    assert solution.diffusion == pytest.approx(2.9940e25, rel=2e-3)
    assert solution.rate_coefficients["ionization"] == pytest.approx(
        2.7230e-16, rel=5e-3
    )
    assert solution.net_ionization == solution.rate_coefficients["ionization"]


# --- conservation --------------------------------------------------------------------


def _problem(processes, growth="temporal", reduced_field=30.0, max_energy=60.0):
    background = gas(LUCAS_SAELEE_MASS)
    return _build(
        processes,
        {"gas": 1.0},
        {"gas": background.mass},
        reduced_field * TOWNSEND,
        0.0,
        growth,
        max_energy,
        120,
    )


@pytest.mark.parametrize("sharing", ["equal", "uniform", "opal"])
def test_ionization_turns_every_electron_into_two(sharing: str) -> None:
    problem = _problem(lucas_saelee(sharing))
    (channel,) = problem.channels
    slope = np.full(problem.centres.size, -0.3)
    flow = channel.flow(slope)
    if channel.kernel is None:
        assert channel.multiplicity == 2.0
    else:
        np.testing.assert_allclose(channel.kernel.sum(axis=0), 2.0, rtol=1e-12)
    # products land below the incident electron, by at least the ionization energy
    landed = (
        channel.kernel.T @ problem.centres / 2.0
        if channel.kernel is not None
        else problem.centres[channel.target]
    )
    assert np.all(landed < channel.energy - 15.6 + problem.widths[0])
    assert flow.sum() > 0.0


@pytest.mark.parametrize("growth", ["temporal", "spatial"])
@pytest.mark.parametrize("sharing", ["equal", "uniform", "opal"])
def test_the_discretisation_conserves_the_electrons(growth: str, sharing: str) -> None:
    """The columns of the converged operator sum to zero: no electron is lost."""
    problem = _problem(lucas_saelee(sharing), growth)
    solution = solve_eedf(
        lucas_saelee(sharing),
        [gas(LUCAS_SAELEE_MASS)],
        30.0,
        0.0,
        growth=growth,
        num_cells=120,
        max_energy=60.0,
    )
    slope = _log_slope(solution.eepf, problem)
    moments = _moments(solution.eepf, slope, problem, solution.townsend_coefficient)
    matrix = _matrix(problem, slope, moments)
    # every equation holds, also the one replaced by the normalisation
    residual = matrix @ solution.eepf
    scale = np.abs(matrix).sum(axis=1) * solution.eepf.max()
    assert np.max(np.abs(residual) / scale) < 1e-7
    assert abs(residual.sum()) / scale.max() < 1e-7


def test_the_energy_balance_closes_and_converges_with_the_grid() -> None:
    errors = []
    for cells in (50, 100, 200):
        solution = solve_eedf(
            lucas_saelee(), [gas(LUCAS_SAELEE_MASS)], 30.0, 0.0, num_cells=cells
        )
        errors.append(abs(solution.power_balance))
        inelastic = 15.6 * solution.rate_coefficients["ionization"]
        assert solution.power["inelastic"] == pytest.approx(inelastic)
        assert solution.power["growth"] == pytest.approx(
            solution.net_ionization * solution.mean_energy
        )
    assert errors[2] < 1e-4
    # second order in the cell size
    assert errors[1] / errors[2] > 3.0


# --- growth models -------------------------------------------------------------------


def test_spatial_and_temporal_growth_agree_without_ionization() -> None:
    temporal = solve_eedf(REID, [gas()], 12.0, 0.0)
    spatial = solve_eedf(REID, [gas()], 12.0, 0.0, growth="spatial")
    assert spatial.mean_energy == pytest.approx(temporal.mean_energy, rel=1e-10)
    assert spatial.mobility == pytest.approx(temporal.mobility, rel=1e-10)


def test_spatial_growth_is_self_consistent() -> None:
    """With strong ionization the diffusion against the field cools the electrons."""
    processes = lucas_saelee()
    temporal = solve_eedf(processes, [gas(LUCAS_SAELEE_MASS)], 30.0, 0.0)
    spatial = solve_eedf(
        processes, [gas(LUCAS_SAELEE_MASS)], 30.0, 0.0, growth="spatial"
    )
    assert spatial.mean_energy < temporal.mean_energy
    # alpha/N solves (alpha/N)^2 DN - (alpha/N) muN E/N + k = 0
    alpha = spatial.townsend_coefficient
    field = 30.0 * TOWNSEND
    assert alpha**2 * spatial.diffusion - alpha * spatial.mobility * field == (
        pytest.approx(-spatial.net_ionization, rel=1e-9)
    )
    assert abs(spatial.power_balance) < 1e-4
    assert spatial.growth == "spatial"


@pytest.mark.parametrize("growth", ["temporal", "spatial"])
def test_attachment_removes_electrons(growth: str) -> None:
    attachment = CrossSection(
        energy=[0.0, 0.5, 1.0, 1.5, 100.0],
        sigma=[0.0, 0.0, 2.0e-21, 0.0, 0.0],
        label="attachment",
    )
    processes = [
        *REID,
        CollisionProcess(
            "attachment",
            "gas",
            attachment,
            name="attachment",
            products={"negative_ion": "ion-"},
        ),
    ]
    solution = solve_eedf(processes, [gas()], 24.0, 0.0, growth=growth)
    assert solution.net_ionization == -solution.rate_coefficients["attachment"]
    assert solution.net_ionization < 0.0
    assert solution.townsend_coefficient < 0.0
    assert solution.power["growth"] > 0.0
    assert abs(solution.power_balance) < 1e-4


def test_strong_attachment_keeps_the_effective_cross_section_positive() -> None:
    """A decay faster than the momentum transfer at low energy is clipped, not inverted."""
    processes = [
        elastic(CrossSection.constant(1.0e-21)),
        CollisionProcess(
            "dissociative_attachment",
            "gas",
            CrossSection.maxwell_molecule(1.0e-14, ELECTRON_MASS),
            name="attachment",
            products={"negative_ion": "ion-"},
        ),
    ]
    solution = solve_eedf(processes, [gas()], 10.0, 0.0, max_energy=20.0)
    assert solution.rate_coefficients["attachment"] == pytest.approx(1.0e-14, rel=1e-2)
    assert solution.mobility > 0.0


# --- inputs and outputs --------------------------------------------------------------


def test_a_mixture_weighs_the_gases_by_their_densities() -> None:
    sigma = CrossSection.constant(6.0e-20)
    light = NeutralBackground("light", 3.0e22, 0.0, M_HE)
    heavy = NeutralBackground("heavy", 1.0e22, 0.0, 10.0 * M_HE)
    mixed = solve_eedf(
        [elastic(sigma, "light"), elastic(sigma, "heavy")], [light, heavy], 5.0, 0.0
    )
    # the same energy loss as one gas with 1/M = 0.75/M_He + 0.25/(10 M_He)
    mean_mass = 1.0 / (0.75 / M_HE + 0.25 / (10.0 * M_HE))
    single = solve_eedf([elastic(sigma)], [gas(mean_mass)], 5.0, 0.0)
    assert mixed.mean_energy == pytest.approx(single.mean_energy, rel=1e-6)
    assert set(mixed.rate_coefficients) == {"elastic light", "elastic heavy"}


def test_the_solution_reports_consistent_quantities() -> None:
    solution = solve_eedf(REID, [gas(temperature=0.02)], 12.0)
    assert solution.gas_temperature == 0.02
    assert solution.reduced_field == 12.0
    assert np.sum(solution.eedf * np.diff(solution.energy_edges)) == pytest.approx(
        1.0, rel=1e-3
    )
    assert solution.drift_velocity == pytest.approx(solution.mobility * 12.0e-21)
    # the elastic rate is the momentum-transfer frequency over N
    expected = (
        6.0e-20
        * GAMMA
        * np.sum(solution.energy * solution.eepf * np.diff(solution.energy_edges))
    )
    assert solution.rate_coefficients["elastic gas"] == pytest.approx(
        expected, rel=1e-3
    )
    # the grid spans 10 to 20 decades of the distribution
    assert 10.0 <= math.log10(solution.eepf[0] / solution.eepf[-1]) <= 20.0
    fixed = solve_eedf(REID, [gas()], 12.0, max_energy=3.0, num_cells=150)
    assert fixed.energy_edges[-1] == 3.0
    assert fixed.energy.size == 150


def test_a_distribution_that_drops_at_a_threshold_ends_the_grid_there() -> None:
    """No range spans 10 to 20 decades: the grid stops just above the threshold."""
    processes = [
        elastic(CrossSection.constant(1e-21)),
        CollisionProcess(
            "excitation", "gas", CrossSection.constant(1e-14, threshold=2.0)
        ),
    ]
    solution = solve_eedf(processes, [gas()], 3.0, 0.0, num_cells=50)
    assert 2.0 < solution.energy_edges[-1] < 2.1
    assert solution.eepf[-1] < 1e-20 * solution.eepf.max()


@pytest.mark.parametrize(
    ("processes", "kwargs", "message"),
    [
        (REID, {"growth": "both"}, "growth"),
        (REID, {"reduced_field": -1.0}, ">= 0"),
        (REID, {"reduced_field": 0.0}, "no solution"),
        (REID, {"num_cells": 5}, "num_cells"),
        (REID, {"max_energy": 0.0}, "max_energy"),
        (REID, {"mixture": []}, "positive density"),
        ([elastic(CrossSection.constant(1e-20), "other")], {}, "no gas 'other'"),
        ([REID[0], REID[0]], {}, "unique"),
        (
            [CollisionProcess("backscatter", "gas", CrossSection.constant(1e-20))],
            {},
            "electron-impact",
        ),
        (
            [
                CollisionProcess(
                    "elastic",
                    "gas",
                    CrossSection.constant(1e-20),
                    scattering="vahedi_surendra",
                )
            ],
            {},
            "momentum-transfer",
        ),
        (
            [elastic(CrossSection(energy=[0.0, 1.0, 2.0], sigma=[1e-20, 0.0, 0.0]))],
            {"max_energy": 5.0},
            "positive",
        ),
    ],
)
def test_invalid_input_is_rejected(processes, kwargs, message: str) -> None:
    arguments: dict[str, Any] = {
        "mixture": [gas()],
        "reduced_field": 10.0,
        "gas_temperature": 0.0,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        solve_eedf(processes, **arguments)


def test_an_iteration_limit_that_is_too_small_raises() -> None:
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_eedf(
            lucas_saelee(), [gas(LUCAS_SAELEE_MASS)], 30.0, 0.0, max_iterations=2
        )


# --- swarm tables --------------------------------------------------------------------


def test_a_swarm_table_tabulates_against_the_mean_energy() -> None:
    fields = [1.0, 5.0, 12.0, 24.0, 40.0]
    table = swarm_table(REID, [gas()], fields, 0.0, num_cells=100)
    assert table.reduced_field.tolist() == fields
    assert len(table.solutions) == 5
    assert np.all(np.diff(table.mean_energy) > 0.0)
    middle = table.solutions[2]
    assert table.interpolate("reduced_field", middle.mean_energy) == pytest.approx(12.0)
    assert table.interpolate("mobility", [middle.mean_energy]) == pytest.approx(
        [middle.mobility]
    )
    assert table.interpolate("excitation", middle.mean_energy) == pytest.approx(
        middle.rate_coefficients["excitation"]
    )
    # a zero stays zero; outside the table the column is clamped
    assert table.interpolate("townsend_coefficient", 10.0) == 0.0
    assert table.interpolate("diffusion", 1e-3) == pytest.approx(table.diffusion[0])
    rates = table.rate_table("excitation")
    assert rates(2.0 / 3.0 * middle.mean_energy) == pytest.approx(
        middle.rate_coefficients["excitation"]
    )
    assert rates.label == "excitation"
    with pytest.raises(KeyError, match="no_such"):
        table.interpolate("no_such", 0.2)
    with pytest.raises(KeyError, match="no_such"):
        table.rate_table("no_such")


def test_a_swarm_table_refuses_to_interpolate_a_non_monotonic_mean_energy() -> None:
    table = swarm_table(REID, [gas()], [5.0, 12.0], 0.0, num_cells=60)
    flipped = type(table)(
        reduced_field=table.reduced_field,
        mean_energy=table.mean_energy[::-1].copy(),
        mobility=table.mobility,
        diffusion=table.diffusion,
        townsend_coefficient=table.townsend_coefficient,
        rate_coefficients=table.rate_coefficients,
        solutions=table.solutions,
    )
    with pytest.raises(ValueError, match="does not increase"):
        flipped.interpolate("mobility", 0.2)
    # a column with negative values is interpolated linearly in log energy
    attaching = type(table)(
        reduced_field=table.reduced_field,
        mean_energy=np.array([1.0, 4.0]),
        mobility=table.mobility,
        diffusion=table.diffusion,
        townsend_coefficient=np.array([-3.0e-21, 1.0e-21]),
        rate_coefficients=table.rate_coefficients,
        solutions=table.solutions,
    )
    assert attaching.interpolate("townsend_coefficient", 2.0) == pytest.approx(-1e-21)
    with pytest.raises(ValueError, match="increasing"):
        swarm_table(REID, [gas()], [12.0, 5.0], 0.0)


# --- BOLOS ---------------------------------------------------------------------------

# A synthetic argon-like gas (not measured data) with every kind of process.
_TABLE_ENERGY = np.round(np.geomspace(1e-2, 1e3, 120), 5)


def _model_gas_file() -> str:
    energy = _TABLE_ENERGY

    def block(kind, target, parameter, sigma):
        lines = [kind, target] + (
            [f" {parameter:.6e}"] if parameter is not None else []
        )
        lines += ["-----"] + [
            f" {e:.6e}\t{s:.6e}" for e, s in zip(energy, sigma, strict=True)
        ]
        return "\n".join([*lines, "-----"])

    x = np.clip(energy / 15.8, 1.0, None)
    return "\n\n".join(
        [
            block(
                "ELASTIC",
                "Gas",
                ELECTRON_MASS / (40.0 * ATOMIC_MASS),
                8e-20 / (1.0 + (energy / 10.0) ** 0.8),
            ),
            block(
                "EXCITATION",
                "Gas -> Gas*",
                11.5,
                np.where(
                    energy > 11.5,
                    2e-21 * (energy - 11.5) / (1.0 + (energy / 30.0) ** 1.5),
                    0.0,
                ),
            ),
            block(
                "IONIZATION",
                "Gas -> Gas^+",
                15.8,
                np.where(energy > 15.8, 3e-20 * np.log(x) / x, 0.0),
            ),
            block(
                "ATTACHMENT",
                "Gas -> Gas^-",
                None,
                1e-22 * np.exp(-(((energy - 6.0) / 1.5) ** 2)),
            ),
        ]
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("reduced_field", [30.0, 100.0, 300.0])
def test_bolos_agrees_on_a_shared_cross_section_set(
    reduced_field: float, tmp_path: Path, monkeypatch
) -> None:
    """BOLOS (LGPL, used only here if installed) solves the same model the same way."""
    pytest.importorskip("bolos")
    integrate = pytest.importorskip("scipy.integrate")
    grid = importlib.import_module("bolos.grid")
    parser = importlib.import_module("bolos.parser")
    solver = importlib.import_module("bolos.solver")

    # BOLOS 0.2 calls scipy.integrate.simps, which SciPy 1.14 renamed simpson
    if not hasattr(integrate, "simps"):
        monkeypatch.setattr(integrate, "simps", integrate.simpson, raising=False)
    text = _model_gas_file()
    path = tmp_path / "model_gas.txt"
    path.write_text(text)
    table = read_lxcat(path)
    temperature = kelvin_to_ev(300.0)
    processes = [
        CollisionProcess("elastic", "Gas", table[0].cross_section, name="elastic"),
        CollisionProcess(
            "excitation", "Gas", table[1].cross_section, name="excitation"
        ),
        CollisionProcess(
            "ionization",
            "Gas",
            table[2].cross_section,
            name="ionization",
            products=IONIZATION_PRODUCTS,
        ),
        CollisionProcess(
            "attachment",
            "Gas",
            table[3].cross_section,
            name="attachment",
            products={"negative_ion": "ion-"},
        ),
    ]
    background = NeutralBackground("Gas", 1e22, temperature, 40.0 * ATOMIC_MASS)
    ours = solve_eedf(processes, [background], reduced_field)

    bolos: Any = solver.BoltzmannSolver(
        grid.LinearGrid(0.0, ours.energy_edges[-1], 400)
    )
    bolos.load_collisions(parser.parse(io.StringIO(text)))
    bolos.target["Gas"].density = 1.0
    bolos.kT = temperature
    bolos.EN = reduced_field * solver.TOWNSEND
    bolos.init()
    eepf = bolos.converge(bolos.maxwell(2.0), maxn=500, rtol=1e-8)

    assert ours.mean_energy == pytest.approx(bolos.mean_energy(eepf), rel=5e-3)
    assert ours.mobility == pytest.approx(bolos.mobility(eepf), rel=5e-3)
    assert ours.diffusion == pytest.approx(bolos.diffusion(eepf), rel=5e-3)
    for name, signature in (
        ("excitation", "Gas -> Gas*"),
        ("ionization", "Gas -> Gas^+"),
        ("attachment", "Gas -> Gas^-"),
    ):
        assert ours.rate_coefficients[name] == pytest.approx(
            bolos.rate(eepf, signature), rel=5e-3
        )
