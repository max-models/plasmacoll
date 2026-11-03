"""The two-term Boltzmann solver against LoKI-B on model gases.

The reference, ``data/lokib_reference.json``, is the output of LoKI-B++ (the
C++ LoKI-B of Tejero-del-Caz et al., Plasma Sources Sci. Technol. 28, 043001
(2019); https://github.com/LoKI-Suite/LoKI-B-cpp, commit add5518, built from
source) for the synthetic gases of `_lokib_models`, which also regenerates it.
The cases test what the BOLOS comparison does not: spatial growth, the
Opal-Peterson-Beaty energy sharing and the superelastic collisions of a
molecule's rotational and vibrational levels, populated at the gas temperature
and at a vibrational temperature.

LoKI-B is first order in its energy step: its results on 4096 and on 2048
cells differ by up to 0.1 %. On LoKI-B's grid with 1000 cells ``solve_eedf``
agrees with it to within that difference, and with LoKI-B's results
extrapolated to a zero step to about 1e-4.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import numpy as np
import pytest

from plasmacoll import BoltzmannSolution, solve_eedf
from plasmacoll.constants import TOWNSEND
from plasmacoll.tests.unit._lokib_models import REFERENCE, plasmacoll_model

CASES = json.loads(REFERENCE.read_text())["cases"]
#: Rates further down the tail than this (m^3/s) are resolved by neither code.
SMALLEST_RATE = 1e-25


def _solve(case: dict, **options) -> BoltzmannSolution:
    processes, mixture, states = plasmacoll_model(
        case["gas"], case["ionization_operator"]
    )
    return solve_eedf(
        processes,
        mixture,
        case["reduced_field"],
        states=states,
        growth=case["growth"],
        **options,
    )


def _compared(case: dict, solution: BoltzmannSolution) -> Iterator[tuple]:
    """Yield (quantity, plasmacoll, LoKI-B, LoKI-B on half the cells)."""
    coarse = case["coarse"]
    for key in ("mean_energy", "mobility", "diffusion"):
        yield key, getattr(solution, key), case[key], coarse[key]
    effective = case["townsend"] - case["attachment"]
    if effective != 0.0:
        coarse_effective = coarse["townsend"] - coarse["attachment"]
        if case["growth"] == "spatial":
            # LoKI-B: alpha/N = k_iz / w with w = muN E/N - DN alpha_eff/N
            yield (
                "townsend_coefficient",
                solution.townsend_coefficient,
                effective,
                coarse_effective,
            )
        else:
            # LoKI-B's alpha/N is k_iz / w in temporal growth
            drift = case["mobility"] * case["reduced_field"] * TOWNSEND
            yield (
                "net_ionization",
                solution.net_ionization,
                effective * drift,
                coarse_effective * drift,
            )
    for reaction, (forward, backward) in case["rates"].items():
        coarse_forward, coarse_backward = coarse["rates"][reaction]
        if forward > SMALLEST_RATE:
            yield (
                reaction,
                solution.rate_coefficients[reaction],
                forward,
                coarse_forward,
            )
        if backward > SMALLEST_RATE:
            yield (
                f"{reaction} (superelastic)",
                solution.rate_coefficients[f"{reaction} (superelastic)"],
                backward,
                coarse_backward,
            )


@pytest.mark.parametrize("name", sorted(CASES))
def test_lokib_agrees_within_its_discretisation_error(name: str) -> None:
    """On LoKI-B's grid: within twice LoKI-B's change from 2048 to 4096 cells."""
    case = CASES[name]
    solution = _solve(case, num_cells=1000, max_energy=case["max_energy"])
    assert solution.energy_edges[-1] == case["max_energy"]
    for quantity, ours, lokib, coarse in _compared(case, solution):
        allowed = 2.0 * abs(lokib / coarse - 1.0) + 5e-4
        assert ours == pytest.approx(lokib, rel=allowed), quantity
    # the distribution, down to 1e-6 of its maximum (between our centres)
    energy = np.asarray(case["eedf_energy"])
    eepf = np.asarray(case["eedf"])
    shown = (eepf > 1e-6 * eepf.max()) & (energy >= solution.energy[0])
    ours = np.exp(np.interp(energy[shown], solution.energy, np.log(solution.eepf)))
    np.testing.assert_allclose(ours, eepf[shown], rtol=5e-3)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_default_grid_agrees_with_lokib(name: str) -> None:
    """200 cells on the grid ``solve_eedf`` chooses: 0.6 % at 1000 Td, 0.2 % below."""
    case = CASES[name]
    solution = _solve(case)
    allowed = 6e-3 if case["reduced_field"] >= 1000.0 else 2e-3
    for quantity, ours, lokib, _ in _compared(case, solution):
        assert ours == pytest.approx(lokib, rel=allowed), quantity


def test_the_reference_closes_its_power_balance() -> None:
    """LoKI-B conserves energy to rounding; every case and process is there."""
    assert len(CASES) == 16
    for case in CASES.values():
        assert abs(case["relative_power_balance"]) < 1e-10
    molecule = CASES["mol_1"]
    superelastic = [rates[1] for rates in molecule["rates"].values() if rates[1]]
    assert len(superelastic) == 9  # J -> J + 2 for J = 0..7, and v = 0 -> 1
