"""Binary Coulomb collisions against the NRL rates, and their conservation laws.

The rates are compared with the NRL Plasma Formulary in zero dimensions: markers in a
few cells of unit volume, collided with `collide_intra` and `collide_inter`.
The time steps keep the thermal scattering parameter ``s`` of the pairs at a few
1e-3: binary collisions underestimate the energy exchange when ``s`` is not small
(slow pairs saturate), by about 4 % at ``s = 2.6e-3``, 7 % at ``s = 5e-3`` and
25 % at ``s = 0.1``.
"""

from __future__ import annotations

import math

import cunumpy as xp
import numpy as np
import pytest

from plasmacoll.constants import (
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    PROTON_MASS,
    VACUUM_PERMITTIVITY,
)
from plasmacoll.coulomb import (
    CellMarkers,
    collide_inter,
    collide_intra,
    coulomb_logarithm,
    pair_coefficient,
    relaxation_rate,
)

# NumPy markers: the CUDA versions of the kernels are checked against the Pyccel ones
# in test_coulomb_kernels.py
pytestmark = pytest.mark.skipif(
    xp.get_backend() == "cupy", reason="the Coulomb physics tests use NumPy arrays"
)

_erf = np.vectorize(math.erf)


def _kurtosis(values):
    """The excess kurtosis of ``values``."""
    centred = values - np.mean(values)
    return np.mean(centred**4) / np.mean(centred**2) ** 2 - 3.0


def _ks_statistic(values):
    """The Kolmogorov-Smirnov distance of ``values`` from the standard normal."""
    values = np.sort(values)
    cdf = 0.5 * (1.0 + _erf(values / np.sqrt(2.0)))
    n = values.size
    return max(np.max(np.arange(1, n + 1) / n - cdf), np.max(cdf - np.arange(n) / n))


E = ELEMENTARY_CHARGE
DENSITY = 1.0e20
LOG = 10.0


def _maxwellian(rng, mass, temperature, count, drift=0.0, exact=False):
    """``(3, count)`` velocities of a Maxwellian at ``temperature`` (eV).

    ``exact``: shifted and scaled to exactly no mean velocity and ``temperature``.
    """
    velocities = rng.normal(0.0, np.sqrt(temperature * E / mass), (3, count))
    if exact:
        velocities -= np.mean(velocities, axis=1, keepdims=True)
        velocities *= np.sqrt(
            3.0 * temperature * E / mass / np.mean(np.sum(velocities**2, axis=0))
        )
    velocities[0] += drift
    return velocities


def _cells(velocities, weights, num_cells, mass, charge):
    """The markers spread evenly over ``num_cells`` cells (already in cell order)."""
    keys = np.repeat(np.arange(num_cells), velocities.shape[1] // num_cells)
    return CellMarkers.from_sorted(velocities, weights, keys, num_cells, mass, charge)


def _temperature(velocities, mass, weights=None):
    """The temperature (eV) of markers, about their mean velocity."""
    weights = np.ones(velocities.shape[1]) if weights is None else weights
    mean = np.average(velocities, axis=1, weights=weights)
    spread = np.sum((velocities - mean[:, None]) ** 2, axis=0)
    return mass * np.average(spread, weights=weights) / (3.0 * E)


# Coulomb logarithm and rates ---------------------------------------------#


def test_coulomb_logarithm_follows_the_nrl_formulas():
    """Electron-electron, electron-ion (both ranges) and ion-ion values, and the floor."""
    me, mp = ELECTRON_MASS, PROTON_MASS
    n_cgs, t = 1.0e13, 10.0
    expected_ee = (
        23.5
        - np.log(np.sqrt(n_cgs) * t**-1.25)
        - np.sqrt(1e-5 + (np.log(t) - 2.0) ** 2 / 16.0)
    )
    assert float(coulomb_logarithm(me, -E, 1e19, t, me, -E, 1e19, t)) == pytest.approx(
        expected_ee, rel=1e-12
    )
    # T_e = 5 eV < 10 Z^2 eV, and T_e = 50 eV above it
    cold = coulomb_logarithm(me, -E, 1e19, 5.0, mp, E, 1e19, 1.0)
    hot = coulomb_logarithm(mp, E, 1e19, 1.0, me, -E, 1e19, 50.0)
    assert float(cold) == pytest.approx(
        23.0 - np.log(np.sqrt(n_cgs) * 5.0**-1.5), rel=1e-12
    )
    assert float(hot) == pytest.approx(24.0 - np.log(np.sqrt(n_cgs) / 50.0), rel=1e-12)
    # electrons colder than T_i m_e / m_i: the ion formula
    coldest = coulomb_logarithm(me, -E, 1e19, 1e-4, mp, E, 1e19, 10.0)
    assert float(coldest) == pytest.approx(
        16.0 - np.log(np.sqrt(n_cgs) * 10.0**-1.5 * mp / PROTON_MASS), rel=1e-12
    )
    # ion-ion, protons and alphas
    ii = coulomb_logarithm(mp, E, 1e19, 2.0, 4 * mp, 2 * E, 1e18, 3.0)
    expected_ii = 23.0 - np.log(
        1 * 2 * 5.0 / (1 * 3.0 + 4 * 2.0) * np.sqrt(n_cgs / 2.0 + 1e12 * 4 / 3.0)
    )
    assert float(ii) == pytest.approx(expected_ii, rel=1e-12)
    # a dense, cold plasma: the floor of 2
    assert float(coulomb_logarithm(mp, E, 1e32, 0.01, mp, E, 1e32, 0.01)) == 2.0
    # arrays broadcast
    values = coulomb_logarithm(me, -E, np.array([1e18, 1e20]), 10.0, mp, E, 1e19, 1.0)
    assert values.shape == (2,) and values[0] > values[1]


def test_relaxation_rate_is_the_nrl_rate_in_si():
    """``1.8e-19 (m_a m_b)^1/2 Z^2 Z^2 n lnL / (m_a T_b + m_b T_a)^3/2`` in cgs."""
    ma, mb = ELECTRON_MASS, PROTON_MASS
    rate = relaxation_rate(-E, ma, 10.0, E, mb, 1e19, 2.0, LOG)
    grams_a, grams_b = ma * 1e3, mb * 1e3
    nrl = (
        1.8e-19
        * np.sqrt(grams_a * grams_b)
        * 1e13
        * LOG
        / (grams_a * 2.0 + grams_b * 10.0) ** 1.5
    )
    assert rate / nrl == pytest.approx(1.0, abs=0.03)


# Rates against theory ------------------------------------------------------#


def test_temperature_relaxation_of_two_ion_species_follows_the_nrl_rate():
    """Protons at 2 eV and ions of 4 proton masses at 1 eV relax at the NRL rate.

    The NRL rate holds between Maxwellians: the collisions of each species with
    itself are made fast (a Coulomb logarithm 100 times larger) to keep both
    Maxwellian, so the comparison tests the collisions between the species.
    """
    rng = np.random.default_rng(1)
    count, num_cells, steps = 60_000, 30, 300
    ma, mb = PROTON_MASS, 4 * PROTON_MASS
    ta, tb = 2.0, 1.0
    va = _maxwellian(rng, ma, ta, count, exact=True)
    vb = _maxwellian(rng, mb, tb, count, exact=True)
    weights = np.full(count, DENSITY * num_cells / count)
    nu0 = relaxation_rate(E, ma, ta, E, mb, DENSITY, tb, LOG) + relaxation_rate(
        E, mb, tb, E, ma, DENSITY, ta, LOG
    )
    dt = 0.7 / nu0 / steps
    theory_a, theory_b = ta, tb
    for _ in range(steps):
        for _ in range(10):
            h = dt / 10
            ra = relaxation_rate(E, ma, theory_a, E, mb, DENSITY, theory_b, LOG)
            rb = relaxation_rate(E, mb, theory_b, E, ma, DENSITY, theory_a, LOG)
            theory_a, theory_b = (
                theory_a + h * ra * (theory_b - theory_a),
                theory_b + h * rb * (theory_a - theory_b),
            )
        a = _cells(va, weights, num_cells, ma, E)
        b = _cells(vb, weights, num_cells, mb, E)
        collide_intra(a, dt, 1.0, 100 * LOG, rng=rng)
        collide_intra(b, dt, 1.0, 100 * LOG, rng=rng)
        collide_inter(a, b, dt, 1.0, LOG, rng=rng)
    simulated = _temperature(va, ma) - _temperature(vb, mb)
    ratio = np.log(simulated / (ta - tb)) / np.log((theory_a - theory_b) / (ta - tb))
    # about 0.96 +- 0.015 at this step (s = 2.6e-3 at the thermal speed), 0.99 at
    # half of it
    assert ratio == pytest.approx(1.0, abs=0.07)


def _isotropisation_rate(t_par, t_perp, mass):
    """The NRL isotropisation rate of a bi-Maxwellian of one species (SI, eV)."""
    anisotropy = t_perp / t_par - 1.0
    factor = (
        -3.0 + (anisotropy + 3.0) * np.arctan(np.sqrt(anisotropy)) / np.sqrt(anisotropy)
    ) / anisotropy**2
    return (
        E**4
        * DENSITY
        * LOG
        / (
            8.0
            * np.pi**1.5
            * VACUUM_PERMITTIVITY**2
            * np.sqrt(mass)
            * (t_par * E) ** 1.5
        )
        * factor
    )


@pytest.mark.parametrize("method", ["nanbu", "takizuka_abe"])
def test_bi_maxwellian_isotropises_at_the_nrl_rate(method):
    """``T_perp = 2 T_par`` relaxes at the NRL rate over its first 5 %."""
    rng = np.random.default_rng(2)
    count, num_cells, steps = 600_000, 300, 80
    t_par, t_perp = 1.0, 2.0
    velocities = np.empty((3, count))
    velocities[:2] = rng.normal(0.0, np.sqrt(t_perp * E / PROTON_MASS), (2, count))
    velocities[2] = rng.normal(0.0, np.sqrt(t_par * E / PROTON_MASS), count)
    weights = np.full(count, DENSITY * num_cells / count)
    dt = 0.05 / _isotropisation_rate(t_par, t_perp, PROTON_MASS) / steps
    theory_par, theory_perp = t_par, t_perp
    for _ in range(steps):
        rate = _isotropisation_rate(theory_par, theory_perp, PROTON_MASS)
        change = dt * rate * (theory_perp - theory_par)
        theory_par, theory_perp = theory_par + 2 * change, theory_perp - change
        cells = _cells(velocities, weights, num_cells, PROTON_MASS, E)
        collide_intra(cells, dt, 1.0, LOG, method, rng)
    par = PROTON_MASS * np.mean(velocities[2] ** 2) / E
    perp = PROTON_MASS * np.mean(velocities[:2] ** 2) / E
    ratio = np.log(perp - par) / np.log(theory_perp - theory_par)
    assert ratio == pytest.approx(1.0, abs=0.05)


def test_test_beam_slows_down_at_the_nrl_rate():
    """Light-weight beam protons at about three thermal speeds slow down at ``nu_s``.

    ``dv/dt = -nu_s v`` with ``nu_s = (1 + m_a/m_b) psi(x) q_a^2 q_b^2 lnL n_b /
    (4 pi eps0^2 m_a^2 v^3)``, ``x = m_b v^2 / 2 T_b``, for each marker: the beam
    spreads, so the expected change of its mean velocity is the sum over the steps of
    ``-dt <nu_s(|v|) v_x>`` over its markers. The beam markers weigh 1e-6 of the
    background ones: the background barely changes.
    """
    rng = np.random.default_rng(3)
    num_cells, background_count, beam_count, steps = 20, 40_000, 20_000, 40
    temperature = 1.0
    mass = PROTON_MASS
    speed = 2.0 * np.sqrt(2.0 * temperature * E / mass)
    background = _maxwellian(rng, mass, temperature, background_count)
    beam = np.zeros((3, beam_count))
    beam[0] = speed
    w_background = np.full(background_count, DENSITY * num_cells / background_count)
    w_beam = np.full(beam_count, 1e-6 * DENSITY * num_cells / beam_count)

    def slowing_rate(v):
        x = mass * v**2 / (2.0 * temperature * E)
        psi = _erf(np.sqrt(x)) - 2.0 * np.sqrt(x / np.pi) * np.exp(-x)
        return (
            2.0
            * psi
            * E**4
            * LOG
            * DENSITY
            / (4.0 * np.pi * VACUUM_PERMITTIVITY**2 * mass**2 * v**3)
        )

    dt = 0.1 / slowing_rate(speed) / steps
    expected = 0.0
    initial_background = background.copy()
    for _ in range(steps):
        speeds = np.sqrt(np.sum(beam**2, axis=0))
        expected -= dt * np.mean(slowing_rate(speeds) * beam[0])
        collide_inter(
            _cells(beam, w_beam, num_cells, mass, E),
            _cells(background, w_background, num_cells, mass, E),
            dt,
            1.0,
            LOG,
            rng=rng,
        )
    assert (np.mean(beam[0]) - speed) / expected == pytest.approx(1.0, abs=0.05)
    # the beam slowed down by about 10 %
    assert np.mean(beam[0]) / speed == pytest.approx(0.9, abs=0.02)
    # about steps * 1e-6 of the background markers change
    changed = np.any(background != initial_background, axis=0)
    assert np.count_nonzero(changed) < 10


# Conservation ---------------------------------------------------------------#


def _random_cells(rng, counts, mass, charge, temperature, weights=None):
    """Markers in cells of the given ``counts`` (some odd, some empty)."""
    keys = np.repeat(np.arange(len(counts)), counts)
    total = int(np.sum(counts))
    velocities = _maxwellian(rng, mass, temperature, total, drift=1e4)
    weights = np.full(total, 1e15) if weights is None else weights(total)
    return CellMarkers.from_sorted(velocities, weights, keys, len(counts), mass, charge)


def _totals(*species):
    momentum = sum(s.mass * np.sum(s.weights * s.velocities, axis=1) for s in species)
    energy = sum(0.5 * s.mass * np.sum(s.weights * s.velocities**2) for s in species)
    return momentum, energy


@pytest.mark.parametrize("method", ["nanbu", "takizuka_abe"])
def test_equal_weights_keep_momentum_and_energy_to_round_off(method):
    """Every pair keeps its momentum and energy: the totals change by round-off."""
    rng = np.random.default_rng(4)
    electrons = _random_cells(rng, [0, 1, 2, 3, 7, 50, 51], ELECTRON_MASS, -E, 5.0)
    ions = _random_cells(rng, [3, 0, 1, 9, 2, 20, 1], PROTON_MASS, E, 1.0)
    momentum, energy = _totals(electrons, ions)
    scale = np.sqrt(2.0 * energy * PROTON_MASS * np.sum(ions.weights))
    before = electrons.velocities.copy()
    for _ in range(5):
        collide_intra(electrons, 1e-9, 1e-12, LOG, method, rng)
        collide_intra(ions, 1e-9, 1e-12, None, method, rng)
        collide_inter(electrons, ions, 1e-9, 1e-12, None, method, rng)
    after_momentum, after_energy = _totals(electrons, ions)
    assert not np.array_equal(before, electrons.velocities)
    assert np.max(np.abs(after_momentum - momentum)) / scale < 1e-13
    assert abs(after_energy / energy - 1.0) < 1e-13


def test_unequal_weights_keep_momentum_and_energy_on_average():
    """With random weights the changes are unbiased: their mean is within the noise."""
    rng = np.random.default_rng(5)
    counts = [200] * 20

    def weights(total):
        return rng.uniform(0.2, 1.8, total) * 1e15

    a = _random_cells(rng, counts, PROTON_MASS, E, 3.0, weights)
    b = _random_cells(rng, counts, 2 * PROTON_MASS, E, 1.0, weights)
    momentum, energy = _totals(a, b)
    momentum_steps, energy_steps = [], []
    for _ in range(60):
        start_momentum, start_energy = _totals(a, b)
        collide_intra(a, 2e-9, 1e-12, LOG, rng=rng)
        collide_inter(a, b, 2e-9, 1e-12, LOG, rng=rng)
        end_momentum, end_energy = _totals(a, b)
        momentum_steps.append(end_momentum[1] - start_momentum[1])
        energy_steps.append(end_energy - start_energy)
    energy_steps = np.array(energy_steps)
    momentum_steps = np.array(momentum_steps)
    # each step does change the totals (the rejection is per pair) ...
    assert np.std(energy_steps) > 1e-8 * energy
    # ... but not on average
    for steps in (energy_steps, momentum_steps):
        noise = np.std(steps) / np.sqrt(steps.size)
        assert abs(np.mean(steps)) < 4.0 * noise
    # and the temperatures did relax toward each other
    assert _temperature(a.velocities, PROTON_MASS, a.weights) < 2.9


def test_a_maxwellian_stays_maxwellian():
    """Many strong collision steps leave a Maxwellian's moments and shape unchanged."""
    rng = np.random.default_rng(6)
    count, num_cells = 100_000, 50
    velocities = _maxwellian(rng, PROTON_MASS, 1.0, count)
    weights = np.full(count, DENSITY * num_cells / count)
    energy = np.sum(velocities**2)
    momentum = np.sum(velocities, axis=1)
    sigma = np.sqrt(E / PROTON_MASS)
    # a step of about a tenth of the thermal collision time
    dt = 0.1 / (pair_coefficient(E, E, PROTON_MASS, PROTON_MASS, LOG, 1.0) * DENSITY)
    dt *= (np.sqrt(6.0) * sigma) ** 3
    for _ in range(30):
        collide_intra(
            _cells(velocities, weights, num_cells, PROTON_MASS, E),
            dt,
            1.0,
            LOG,
            rng=rng,
        )
    assert np.sum(velocities**2) / energy == pytest.approx(1.0, rel=1e-12)
    assert np.allclose(
        np.sum(velocities, axis=1), momentum, rtol=0.0, atol=1e-9 * sigma * count
    )
    for component in velocities:
        assert np.var(component) / sigma**2 == pytest.approx(1.0, abs=0.02)
        assert _kurtosis(component) == pytest.approx(0.0, abs=0.06)
        # a p-value above about 1e-3
        assert _ks_statistic(component / sigma) * np.sqrt(component.size) < 1.9


# Errors and cell moments ------------------------------------------------#


def test_inter_needs_the_same_cells_and_three_components():
    """Mismatched cells or 1V velocities are refused."""
    rng = np.random.default_rng(7)
    a = _random_cells(rng, [2, 2], PROTON_MASS, E, 1.0)
    b = _random_cells(rng, [2, 2, 2], PROTON_MASS, E, 1.0)
    with pytest.raises(ValueError, match="same cells"):
        collide_inter(a, b, 1e-9, 1.0, LOG)
    flat = CellMarkers.from_sorted(
        np.zeros((1, 2)), np.ones(2), np.zeros(2, dtype=np.int64), 1, PROTON_MASS, E
    )
    with pytest.raises(ValueError, match="three velocity"):
        collide_intra(flat, 1e-9, 1.0, LOG)
    with pytest.raises(ValueError, match="three velocity"):
        collide_inter(flat, flat, 1e-9, 1.0, LOG)


def test_cell_moments_give_density_and_temperature():
    """Per-cell density and temperature, and the floor for empty cells."""
    rng = np.random.default_rng(8)
    velocities = _maxwellian(rng, PROTON_MASS, 3.0, 20_000, drift=5e4)
    cells = _cells(velocities, np.full(20_000, 2.0), 2, PROTON_MASS, E)
    assert np.allclose(cells.density(4.0), 20_000 / 4.0)
    assert np.allclose(cells.temperature(), 3.0, rtol=0.05)
    empty = CellMarkers.from_sorted(
        velocities[:, :0], np.ones(0), np.zeros(0, dtype=np.int64), 3, PROTON_MASS, E
    )
    assert np.all(empty.temperature(0.5) == 0.5)
    assert empty.num_cells == 3


def test_reduce_sums_the_cells_over_ranks_and_scales_the_rates():
    """Two identical ranks: the densities double and so does each rank's rate.

    ``reduce`` doubling its argument is what an MPI sum over two ranks with the same
    markers gives; the pairs then collide as in a plasma of twice the density.
    """
    rng = np.random.default_rng(9)
    velocities = _maxwellian(rng, PROTON_MASS, 2.0, 4_000)
    weights = np.full(4_000, 1e15)
    keys = np.repeat(np.arange(4), 1_000)

    def double(array):
        array *= 2.0

    shared = CellMarkers.from_sorted(
        velocities.copy(), weights, keys, 4, PROTON_MASS, E, reduce=double
    )
    assert np.allclose(shared.weight, 2.0 * shared.local_weight)
    assert np.allclose(shared.density(1.0), 2e18)
    assert np.allclose(shared.temperature(), 2.0, rtol=0.1)
    alone = CellMarkers.from_sorted(
        velocities.copy(), 2.0 * weights, keys, 4, PROTON_MASS, E
    )
    collide_intra(shared, 1e-9, 1.0, None, rng=np.random.default_rng(10))
    collide_intra(alone, 1e-9, 1.0, None, rng=np.random.default_rng(10))
    np.testing.assert_allclose(shared.velocities, alone.velocities, rtol=1e-12)


def test_the_default_generator_is_seeded_from_the_system():
    """Without ``rng`` the collisions still run, and change the velocities."""
    rng = np.random.default_rng(11)
    cells = _random_cells(rng, [10, 10], PROTON_MASS, E, 1.0)
    other = _random_cells(rng, [10, 10], PROTON_MASS, E, 1.0)
    before = cells.velocities.copy()
    collide_intra(cells, 1e-6, 1e-12, LOG)
    collide_inter(cells, other, 1e-6, 1e-12, LOG)
    assert not np.array_equal(before, cells.velocities)
