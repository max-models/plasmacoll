"""Rudd's secondary-electron cross section against the review's table and closed forms."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from plasmacoll import RUDD_H2, RuddIonization
from plasmacoll.constants import PROTON_MASS
from plasmacoll.secondaries import BOHR_RADIUS, RYDBERG_ENERGY

ANGSTROM2 = 1.0e-20  # m^2

#: Rudd et al., Rev. Mod. Phys. 64, 441 (1992), Table VII: recommended H+ + H2
#: cross sections (A^2/eV) from Eq. (41), ``{proton energy (eV): {W (eV): value}}``.
TABLE_VII = {
    10.0e3: {0.0: 5.52e-2, 10.0: 2.15e-2, 50.0: 1.44e-4},
    100.0e3: {0.0: 1.24e-1, 10.0: 5.23e-2, 50.0: 8.58e-3, 300.0: 5.36e-6},
    1000.0e3: {0.0: 3.47e-2, 10.0: 9.46e-3, 100.0: 2.66e-4, 1000.0: 2.81e-6},
    3000.0e3: {0.0: 1.50e-2, 50.0: 3.51e-4, 7000.0: 2.36e-10},
}


@pytest.mark.parametrize("energy", sorted(TABLE_VII))
def test_h2_reproduces_the_recommended_cross_sections(energy: float) -> None:
    for w, expected in TABLE_VII[energy].items():
        value = float(RUDD_H2.differential(w, energy)[()]) / ANGSTROM2
        assert value == pytest.approx(expected, rel=0.01), w


def test_the_model_depends_on_the_projectile_only_through_its_speed() -> None:
    deuteron = RUDD_H2.with_projectile(2.0 * PROTON_MASS)
    w = np.array([0.0, 5.0, 40.0, 400.0])
    assert deuteron.differential(w, 2.0e6) == pytest.approx(
        RUDD_H2.differential(w, 1.0e6), rel=1e-12
    )
    assert deuteron.total(2.0e6) == pytest.approx(RUDD_H2.total(1.0e6), rel=1e-12)


def test_a_sharp_cutoff_integrates_to_the_closed_form() -> None:
    """With a step at ``w_c``, ``int (F1 + F2 w) / (1 + w)^3`` is elementary."""
    shell = dataclasses.replace(RUDD_H2.shells[0], alpha=1.0e4)
    target = RuddIonization([shell])
    energy = 1.0e6
    t = target.electron_energy(energy)
    v = shell.reduced_speed(t)
    f1, f2 = shell.factors(v)
    a = 1.0 + shell.cutoff(v)
    s = 4.0 * np.pi * BOHR_RADIUS**2 * shell.electrons
    s *= (RYDBERG_ENERGY / shell.binding_energy) ** 2
    zeroth = 0.5 * (1.0 - a**-2)  # int_0^wc (1 + w)^-3 dw
    first = (1.0 - 1.0 / a) - zeroth  # int_0^wc w (1 + w)^-3 dw
    assert target.total(energy) == pytest.approx(
        s * (f1 * zeroth + f2 * first), rel=2e-4
    )
    mean = shell.binding_energy * (f1 * first + f2 * (np.log(a) - 2.0 * first - zeroth))
    mean /= f1 * zeroth + f2 * first
    # int w^2 (1 + w)^-3 = ln(1 + w) - 2 int w (1 + w)^-3 - int (1 + w)^-3
    assert target.mean_energy(energy) == pytest.approx(mean, rel=2e-4)


@pytest.mark.parametrize("energy", [1.0e4, 1.0e6])
def test_the_moments_agree_with_a_fine_quadrature(energy: float) -> None:
    w = np.concatenate(
        [np.linspace(0.0, 100.0, 100001), np.geomspace(100.0, 2e4, 200001)[1:]]
    )
    f = RUDD_H2.differential(w, energy)
    total = np.trapezoid(f, w)
    assert RUDD_H2.total(energy) == pytest.approx(total, rel=1e-6)
    assert RUDD_H2.mean_energy(energy) == pytest.approx(
        np.trapezoid(f * w, w) / total, rel=1e-5
    )


def test_shells_add_and_the_total_tabulates() -> None:
    inner = dataclasses.replace(RUDD_H2.shells[0], binding_energy=40.0, electrons=1)
    both = RuddIonization([RUDD_H2.shells[0], inner])
    alone = RuddIonization([inner])
    energy = 3.0e5
    assert both.total(energy) == pytest.approx(
        RUDD_H2.total(energy) + alone.total(energy), rel=1e-12
    )
    mean = (
        RUDD_H2.total(energy) * RUDD_H2.mean_energy(energy)
        + alone.total(energy) * alone.mean_energy(energy)
    ) / both.total(energy)
    assert both.mean_energy(energy) == pytest.approx(mean, rel=1e-12)
    table = RUDD_H2.cross_section(np.geomspace(1e4, 1e7, 31))
    assert float(table(np.array([1.0e6]))[0]) == pytest.approx(
        RUDD_H2.total(1.0e6), rel=1e-12
    )
    # faster secondaries from faster projectiles, fewer of them above the peak
    assert RUDD_H2.mean_energy(1e6) > RUDD_H2.mean_energy(1e5)
    assert RUDD_H2.total(1e6) < RUDD_H2.total(1e5)


def test_bad_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="binding_energy"):
        dataclasses.replace(RUDD_H2.shells[0], binding_energy=0.0)
    with pytest.raises(ValueError, match="alpha"):
        dataclasses.replace(RUDD_H2.shells[0], alpha=0.0)
    with pytest.raises(ValueError, match="shell"):
        RuddIonization([])
    with pytest.raises(ValueError, match="projectile_mass"):
        RUDD_H2.with_projectile(0.0)
    with pytest.raises(ValueError, match="projectile_energy"):
        RUDD_H2.total(0.0)
    with pytest.raises(ValueError, match="secondary_energy"):
        RUDD_H2.differential(-1.0, 1e6)
