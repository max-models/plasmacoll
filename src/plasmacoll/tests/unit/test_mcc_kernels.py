"""The compiled kernels of the Monte Carlo collisions against their NumPy versions.

The cross sections are synthetic, not physical data.
"""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest

from plasmacoll import CollisionProcess, CrossSection, MonteCarloCollisions
from plasmacoll.background import NeutralBackground, Profile
from plasmacoll.kernels.max_squared_speed import kernel as MAX_SQUARED_SPEED
from plasmacoll.kernels.mcc_collide import kernel as MCC_COLLIDE
from plasmacoll.mcc import _geometric_candidates
from plasmacoll.tests.unit._helpers import M_E, M_HE


def _operator(scattering: str, sharing: str) -> MonteCarloCollisions:
    """Every kind and model the kernel covers, on two drifting backgrounds."""
    elastic = CrossSection([0.0, 10.0, 100.0], [6.0e-20, 4.0e-20, 1.0e-20])
    return MonteCarloCollisions(
        species_masses={"e": M_E, "ion": M_HE},
        backgrounds=[
            NeutralBackground("A", 1.0e21, 0.025, M_HE, drift=(100.0, 0.0, 0.0)),
            NeutralBackground("B", 5.0e20, 0.05, M_HE),
        ],
        processes={
            "e": [
                CollisionProcess("elastic", "A", elastic, scattering=scattering),
                CollisionProcess(
                    "excitation", "B", CrossSection([20.0, 100.0], [1.0e-21, 2.0e-21], 20.0),
                    energy_loss=20.0, scattering=scattering,
                ),
                CollisionProcess(
                    "ionization", "A", CrossSection([24.6, 100.0], [0.0, 3.0e-21], 24.6),
                    energy_loss=24.6, products={"electron": "e", "ion": "ion"},
                    energy_sharing=sharing, sharing_energy=10.0, scattering=scattering,
                ),
                CollisionProcess("backscatter", "B", elastic),
            ],
        },
        seed=0,
    )  # fmt: skip


@pytest.mark.parametrize("scattering", ["isotropic", "vahedi_surendra", "okhrimovskyy"])
@pytest.mark.parametrize("sharing", ["equal", "uniform", "opal"])
def test_compiled_mcc_kernel_matches_numpy(scattering: str, sharing: str) -> None:
    """The Pyccel kernel and the same source run as Python give the same step."""
    mcc = _operator(scattering, sharing)
    plan = mcc.kernel_plan("e")
    assert plan is not None
    rng = np.random.default_rng(1)
    num, num_candidates = 500, 200
    start = rng.standard_normal((3, num)) * 3.0e6
    weights = rng.uniform(0.5, 1.5, num)
    candidates = np.sort(rng.choice(num, num_candidates, replace=False)).astype(
        np.int64
    )
    normals = rng.standard_normal((3 * plan.num_backgrounds, num_candidates))
    uniforms = rng.random((6, num_candidates))
    fastest = float(np.sqrt(np.max(np.sum(start**2, axis=0)))) + 1.0e5
    nu_max = mcc.frequency_bound("e", fastest)
    outputs = {}
    for implementation in ("pyccel", "numpy"):
        velocities = start.copy()
        counts = np.zeros(plan.num_processes + 2, dtype=np.int64)
        parents = np.empty(num_candidates, dtype=np.int64)
        electrons = np.empty((3, num_candidates))
        ions = np.empty((3, num_candidates))
        energy = np.zeros(1)
        with xp.kernels.use_host_kernel_implementation(implementation):
            made = MCC_COLLIDE(
                velocities[0], velocities[1], velocities[2], weights, candidates,
                normals, uniforms, nu_max, M_E, *plan.arguments,
                counts, parents, electrons, ions, energy,
            )  # fmt: skip
        outputs[implementation] = (
            velocities,
            counts,
            parents[:made],
            electrons[:, :made],
            ions[:, :made],
            energy,
        )
    for compiled, python in zip(outputs["pyccel"], outputs["numpy"], strict=True):
        np.testing.assert_allclose(compiled, python, rtol=1.0e-12, atol=0.0)
    counts = outputs["pyccel"][1]
    assert counts[:4].min() > 0  # every process happened
    assert counts[-1] == 0  # no bound violation under the operator's bound


@pytest.mark.parametrize("count", [1, 1001, 50001])
def test_max_squared_speed_kernel_matches_numpy(count) -> None:
    """Serial and (above 20 000 markers) split over the threads: the same maximum."""
    rng = np.random.default_rng(2)
    rows = rng.standard_normal((3, count))
    rows[:, count // 3] *= 10.0  # the fastest marker inside a chunk, not at its end
    values = []
    for implementation in ("pyccel", "numpy"):
        with xp.kernels.use_host_kernel_implementation(implementation):
            values.append(MAX_SQUARED_SPEED(rows[0], rows[1], rows[2]))
    expected = float(np.max(np.sum(rows**2, axis=0)))
    assert values == pytest.approx([expected, expected], rel=1.0e-14)


def test_geometric_candidates_are_bernoulli_draws() -> None:
    """Every marker is a candidate with the same probability, independently."""
    rng = np.random.default_rng(3)
    num, probability, draws = 1000, 0.05, 4000
    hits = np.zeros(num)
    sizes = []
    for _ in range(draws):
        candidates = _geometric_candidates(rng, num, probability)
        assert np.all(np.diff(candidates) > 0)
        assert candidates.size == 0 or (candidates[0] >= 0 and candidates[-1] < num)
        hits[candidates] += 1
        sizes.append(candidates.size)
    # per marker: binomial(draws, p), mean 200 and standard deviation 13.8
    assert abs(hits.mean() - draws * probability) < 1.0
    assert np.all(np.abs(hits - draws * probability) < 6.0 * 13.8)
    # the count per draw: binomial(num, p), variance num p (1 - p)
    assert np.var(sizes) == pytest.approx(
        num * probability * (1 - probability), rel=0.1
    )
    assert _geometric_candidates(rng, num, 1.0).size == num
    assert _geometric_candidates(rng, num, 0.0).size == 0


def test_the_kernel_plan_leaves_the_other_cases_to_the_array_operations() -> None:
    """Profiles, products the kernel does not make, and other kinds: no plan."""
    elastic = CrossSection([0.0, 100.0], [1.0e-20, 1.0e-20])

    class Step(Profile):
        max_factor = 1.0

        def __call__(self, positions):
            return xp.ones(positions.shape[0])

    def plan(process, background=None):
        background = background or NeutralBackground("gas", 1.0e20, 0.025, M_HE)
        return MonteCarloCollisions(
            species_masses={"e": M_E, "ion": M_HE, "neg": M_HE},
            backgrounds=[background],
            processes={"e": [process]},
        ).kernel_plan("e")

    assert plan(CollisionProcess("elastic", "gas", elastic)) is not None
    assert (
        plan(
            CollisionProcess("elastic", "gas", elastic),
            NeutralBackground("gas", 1.0e20, 0.025, M_HE, profile=Step()),
        )
        is None
    )
    assert (
        plan(
            CollisionProcess(
                "attachment", "gas", elastic, products={"negative_ion": "neg"}
            )
        )
        is None
    )
