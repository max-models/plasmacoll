"""Binary Coulomb collisions between the markers of plasma species, per cell.

The markers of each species are sorted by cell (`CellMarkers.from_sorted`), shuffled in
each cell and collided in pairs (Takizuka and Abe 1977): the markers of a species with
each other (an odd count starts with a triplet of three half-step pairs), or every
marker of the species with more markers in the cell with one of the other species,
whose markers are reused in turn. Each pair turns its relative velocity by an angle
drawn either from the Takizuka-Abe distribution (``tan(chi/2)`` normal with the variance
``s / 2``) or from Nanbu's (1997) cumulative small-angle distribution, valid for any
``s``, with the fit of Pérez et al. (2012). The scattering parameter of a pair of
relative speed ``u`` is

    s = q_a^2 q_b^2 lnL n_a n_b dt / (4 pi eps0^2 mu^2 n_ab u^3),

with ``n_ab = sum_pairs min(w_i, w_j) / V`` (twice that for a species with itself): with
equal weights ``n_ab`` is the density of the species with fewer markers in the cell.

Unequal weights follow Pérez et al. (2012) (the rejection scheme of Nanbu and Yonemura
1998): both markers of a pair are turned by the same angle, the lighter one always and
the heavier one with the probability ``w_light / w_heavy``. Momentum and energy are
therefore kept exactly per pair when the weights are equal and on average otherwise.
The collisions are non-relativistic.

The Coulomb logarithm is fixed per pair of species or computed in every cell from the
two species' densities and temperatures there (`coulomb_logarithm`, NRL Plasma
Formulary). Collisions are local to the cells of the markers given. When several MPI
ranks hold markers of the same cells, pass ``reduce`` (an in-place sum over the ranks)
to `CellMarkers.from_sorted`: the cell densities and temperatures are then summed over
the ranks and the scattering parameter is scaled by the ratio of the global to the
local densities, so the rates are those of the whole plasma.

The pair kernels are ``plasmacoll.kernels.coulomb_collide_intra`` and
``coulomb_collide_inter`` (Pyccel, NumPy and CUDA versions).

Example:
    >>> import numpy as np
    >>> from plasmacoll.constants import ELECTRON_MASS, ELEMENTARY_CHARGE
    >>> rng = np.random.default_rng(1)
    >>> velocities = rng.normal(scale=1e6, size=(3, 100))
    >>> keys = np.repeat(np.arange(4), 25)  # four cells, sorted
    >>> electrons = CellMarkers.from_sorted(
    ...     velocities, np.full(100, 1e14), keys, 4, ELECTRON_MASS, -ELEMENTARY_CHARGE)
    >>> collide_intra(electrons, dt=1e-10, cell_volume=1e-6, coulomb_log=10.0, rng=rng)

References:
    T. Takizuka and H. Abe, J. Comput. Phys. 25, 205 (1977).
    K. Nanbu, Phys. Rev. E 55, 4642 (1997).
    K. Nanbu and S. Yonemura, J. Comput. Phys. 145, 639 (1998).
    F. Pérez et al., Phys. Plasmas 19, 083104 (2012).
    J. D. Huba, NRL Plasma Formulary (2019).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import cunumpy as xp
import numpy as np
import scope_profiler as sp
from methodcard import method

from plasmacoll._profiling import PROFILE_TAGS
from plasmacoll._types import Array
from plasmacoll.constants import (
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    PROTON_MASS,
    VACUUM_PERMITTIVITY,
)
from plasmacoll.kernels.coulomb_collide_inter import kernel as _inter_kernel
from plasmacoll.kernels.coulomb_collide_intra import kernel as _intra_kernel
from plasmacoll.mcc import make_rng

__all__ = [
    "METHODS",
    "MIN_COULOMB_LOG",
    "CellMarkers",
    "collide_inter",
    "collide_intra",
    "coulomb_logarithm",
    "pair_coefficient",
    "relaxation_rate",
]

#: The pair scattering, by name: the ``method`` argument of the kernels.
METHODS = {"takizuka_abe": 0, "nanbu": 1}

#: Lower bound of a computed Coulomb logarithm.
MIN_COULOMB_LOG = 2.0


def _is_electron(mass: float) -> bool:
    """Return whether a species of ``mass`` (kg) counts as electrons for the Coulomb logarithm."""
    return mass < 100.0 * ELECTRON_MASS


def coulomb_logarithm(
    mass_a: float,
    charge_a: float,
    density_a: Array | float,
    temperature_a: Array | float,
    mass_b: float,
    charge_b: float,
    density_b: Array | float,
    temperature_b: Array | float,
) -> Array:
    """Return the Coulomb logarithm of two species (NRL Plasma Formulary, 2019).

    Densities in m^-3, temperatures in eV, masses in kg and charges in C (arrays
    broadcast, e.g. one value per cell). Electron-electron, electron-ion (the three
    temperature ranges) and ion-ion (mixed, counter-streaming neglected) formulas; a
    species lighter than 100 electron masses counts as electrons. Densities below
    1 m^-3 (empty cells) count as 1 m^-3; the result is bounded below by 2.

    Args:
        mass_a: Mass of species ``a`` in kg.
        charge_a: Charge of species ``a`` in C.
        density_a: Density of species ``a`` in m^-3.
        temperature_a: Temperature of species ``a`` in eV.
        mass_b: Mass of species ``b`` in kg.
        charge_b: Charge of species ``b`` in C.
        density_b: Density of species ``b`` in m^-3.
        temperature_b: Temperature of species ``b`` in eV.

    Returns:
        The Coulomb logarithm, broadcast over the arguments.

    Example:
        >>> from plasmacoll.constants import ELECTRON_MASS, ELEMENTARY_CHARGE as e
        >>> m = ELECTRON_MASS
        >>> float(coulomb_logarithm(m, -e, 1e19, 10.0, m, -e, 1e19, 10.0))
        11.33...
    """
    # cm^-3; an empty cell (density 0) counts as 1 m^-3, a finite logarithm
    na = xp.maximum(xp.asarray(density_a, dtype=float), 1.0) * 1e-6
    nb = xp.maximum(xp.asarray(density_b, dtype=float), 1.0) * 1e-6
    ta = xp.asarray(temperature_a, dtype=float)
    tb = xp.asarray(temperature_b, dtype=float)
    za = abs(charge_a) / ELEMENTARY_CHARGE
    zb = abs(charge_b) / ELEMENTARY_CHARGE
    electron_a, electron_b = _is_electron(mass_a), _is_electron(mass_b)
    if electron_a and electron_b:
        ne = 0.5 * (na + nb)
        te = 0.5 * (ta + tb)
        value = (
            23.5
            - xp.log(xp.sqrt(ne) * te**-1.25)
            - xp.sqrt(1e-5 + (xp.log(te) - 2.0) ** 2 / 16.0)
        )
    elif electron_a or electron_b:
        if electron_b:
            na, nb, ta, tb, za, zb = nb, na, tb, ta, zb, za
            mass_b = mass_a
        # a: the electrons, b: the ions of charge zb and mass mass_b
        mu = mass_b / PROTON_MASS
        cold = ta < tb * ELECTRON_MASS / mass_b * zb
        hot = ta > 10.0 * zb**2
        value = xp.where(
            hot,
            24.0 - xp.log(xp.sqrt(na) / ta),
            23.0 - xp.log(xp.sqrt(na) * zb * ta**-1.5),
        )
        value = xp.where(
            cold, 16.0 - xp.log(xp.sqrt(nb) * tb**-1.5 * zb**2 * mu), value
        )
    else:
        mu_a, mu_b = mass_a / PROTON_MASS, mass_b / PROTON_MASS
        value = 23.0 - xp.log(
            za
            * zb
            * (mu_a + mu_b)
            / (mu_a * tb + mu_b * ta)
            * xp.sqrt(na * za**2 / ta + nb * zb**2 / tb)
        )
    return xp.maximum(value, MIN_COULOMB_LOG)


def pair_coefficient(
    charge_a: float,
    charge_b: float,
    mass_a: float,
    mass_b: float,
    coulomb_log: Array | float,
    dt: float,
) -> Array | float:
    """Return ``q_a^2 q_b^2 lnL dt / (4 pi eps0^2 mu^2)`` (SI): ``s u^3 / n``.

    Args:
        charge_a: Charge of species ``a`` in C.
        charge_b: Charge of species ``b`` in C.
        mass_a: Mass of species ``a`` in kg.
        mass_b: Mass of species ``b`` in kg.
        coulomb_log: The Coulomb logarithm (a number or one per cell).
        dt: The time step in s.

    Returns:
        The coefficient, in m^6 s^-3.
    """
    mu = mass_a * mass_b / (mass_a + mass_b)
    eps0 = VACUUM_PERMITTIVITY
    return (
        (charge_a * charge_b) ** 2 * coulomb_log * dt / (4.0 * np.pi * eps0**2 * mu**2)
    )


def relaxation_rate(
    charge_a: float,
    mass_a: float,
    temperature_a: float,
    charge_b: float,
    mass_b: float,
    density_b: float,
    temperature_b: float,
    coulomb_log: float,
) -> float:
    """Return the temperature relaxation rate ``nu`` of species ``a`` on ``b`` (1/s).

    ``dT_a/dt = nu (T_b - T_a)`` between two Maxwellians (NRL Plasma Formulary, in SI)::

        nu = 2^(1/2) n_b q_a^2 q_b^2 lnL (m_a m_b)^(1/2)
             / (6 pi^(3/2) eps0^2 (m_a T_b + m_b T_a)^(3/2)),

    with the temperatures in J. The rate of ``b`` on ``a`` swaps the roles.

    Args:
        charge_a: Charge of species ``a`` in C.
        mass_a: Mass of species ``a`` in kg.
        temperature_a: Temperature of species ``a`` in eV.
        charge_b: Charge of species ``b`` in C.
        mass_b: Mass of species ``b`` in kg.
        density_b: Density of species ``b`` in m^-3.
        temperature_b: Temperature of species ``b`` in eV.
        coulomb_log: The Coulomb logarithm.

    Returns:
        The rate in 1/s.
    """
    ta = temperature_a * ELEMENTARY_CHARGE
    tb = temperature_b * ELEMENTARY_CHARGE
    eps0 = VACUUM_PERMITTIVITY
    return float(
        np.sqrt(2.0)
        * density_b
        * (charge_a * charge_b) ** 2
        * coulomb_log
        * np.sqrt(mass_a * mass_b)
        / (6.0 * np.pi**1.5 * eps0**2 * (mass_a * tb + mass_b * ta) ** 1.5)
    )


@dataclass
class CellMarkers:
    """The markers of one species sorted by cell, and their weight per cell.

    ``velocities`` ``(3, N)`` and ``weights`` ``(N,)`` are in cell order (written in
    place by the collisions); cell ``c`` holds markers ``cell_start[c]`` to
    ``cell_start[c + 1] - 1`` (``keys`` are their cells). ``weight`` is the sum of
    ``w`` per cell over all ranks when the ranks share cells (``reduce``),
    ``local_weight`` on this rank. The velocity moments for the temperature are
    computed on first use (collective with ``reduce``).

    Attributes:
        velocities: Velocities ``(3, N)`` in m/s, sorted by cell.
        weights: Weights ``(N,)``, sorted by cell.
        cell_start: The first marker of every cell, ``(cells + 1,)``.
        mass: Mass of the species in kg.
        charge: Charge of the species in C.
        weight: Total weight per cell, over all ranks.
        local_weight: Total weight per cell on this rank.
        keys: The cell of every marker, ``(N,)``.
        reduce: Sums an array over the ranks in place, or None on one rank.
    """

    velocities: Array
    weights: Array
    cell_start: Array
    mass: float
    charge: float
    weight: Array
    local_weight: Array
    keys: Array
    reduce: Callable[[Array], None] | None = None
    _moments: Array | None = field(default=None, repr=False)

    @staticmethod
    def from_sorted(
        velocities: Array,
        weights: Array,
        keys: Array,
        num_cells: int,
        mass: float,
        charge: float,
        reduce: Callable[[Array], None] | None = None,
    ) -> CellMarkers:
        """Return the markers of ``velocities`` and ``weights``, sorted by their cell ``keys``.

        Args:
            velocities: Velocities ``(3, N)`` in m/s, sorted by cell (a view of the
                marker storage, to collide the markers in place).
            weights: Weights ``(N,)``, sorted by cell.
            keys: The cell of every marker, non-decreasing, ``(N,)``.
            num_cells: The number of cells.
            mass: Mass of the species in kg.
            charge: Charge of the species in C.
            reduce: Sums an array over the MPI ranks in place, when several ranks
                hold markers of the same cells; None otherwise.

        Returns:
            The markers by cell.
        """
        counts = xp.bincount(keys, minlength=num_cells)
        cell_start = xp.zeros(num_cells + 1, dtype=xp.int64)
        cell_start[1:] = xp.cumsum(counts)
        local_weight = xp.bincount(keys, weights=weights, minlength=num_cells)
        weight = local_weight
        if reduce is not None:
            weight = local_weight.copy()
            reduce(weight)
        return CellMarkers(
            velocities=velocities,
            weights=weights,
            cell_start=cell_start,
            mass=mass,
            charge=charge,
            weight=weight,
            local_weight=local_weight,
            keys=keys,
            reduce=reduce,
        )

    @property
    def num_cells(self) -> int:
        """Return the number of cells."""
        return int(self.cell_start.shape[0]) - 1

    @property
    def moments(self) -> Array:
        """Return the sums of ``w``, ``w v`` and ``w |v|^2`` per cell, ``(2 + nvel, cells)``.

        Over all ranks with ``reduce``: the first use is collective.
        """
        if self._moments is None:
            keys, weights, velocities = self.keys, self.weights, self.velocities
            num_cells = self.num_cells
            moments = xp.stack(
                [
                    self.local_weight,
                    *(
                        xp.bincount(
                            keys, weights=weights * velocities[d], minlength=num_cells
                        )
                        for d in range(velocities.shape[0])
                    ),
                    xp.bincount(
                        keys,
                        weights=weights * xp.sum(velocities * velocities, axis=0),
                        minlength=num_cells,
                    ),
                ]
            )
            if self.reduce is not None:
                self.reduce(moments)
            self._moments = moments
        return self._moments

    def density(self, cell_volume: float) -> Array:
        """Return the number density of every cell (m^-3, all ranks).

        Args:
            cell_volume: The volume of a cell in m^3.
        """
        return self.weight / cell_volume

    def temperature(self, floor: float = 0.0) -> Array:
        """Return the temperature of every cell (eV, at least ``floor``).

        Args:
            floor: The lowest temperature returned, in eV.
        """
        moments = self.moments
        weight = moments[0]
        safe = xp.where(weight > 0.0, weight, 1.0)
        mean_square = moments[-1] / safe
        drift_square = xp.sum((moments[1:-1] / safe) ** 2, axis=0)
        components = moments.shape[0] - 2
        temperature = (
            self.mass * (mean_square - drift_square) / (components * ELEMENTARY_CHARGE)
        )
        return xp.maximum(temperature, floor)


def _uniforms(count: int, rng: Any) -> Array:
    """Return ``(5, count)`` uniforms: row 0 in [0, 1), the Box-Muller/Nanbu row 1 in (0, 1]."""
    values = xp.asarray(rng.random((5, count)))
    values[1] = 1.0 - values[1]
    return values


def _cell_log(
    a: CellMarkers,
    b: CellMarkers,
    coulomb_log: float | None,
    cell_volume: float,
    temperature_floor: float,
) -> Array | float:
    """Return the Coulomb logarithm of every cell, or the fixed value."""
    if coulomb_log is not None:
        return coulomb_log
    return coulomb_logarithm(
        a.mass,
        a.charge,
        a.density(cell_volume),
        a.temperature(temperature_floor),
        b.mass,
        b.charge,
        b.density(cell_volume),
        b.temperature(temperature_floor),
    )


def _rank_scale(a: CellMarkers, b: CellMarkers) -> Array:
    """Return ``sqrt(W_a W_b / (w_a w_b))``: global over local weights (1 on one rank)."""
    local = a.local_weight * b.local_weight
    safe = xp.where(local > 0.0, local, 1.0)
    return xp.where(local > 0.0, xp.sqrt(a.weight * b.weight / safe), 0.0)


def _run(kernel: Any, outputs: list[Array], *args: Any) -> None:
    """Call a kernel on ``args``, copying back the ``outputs`` it got as copies."""
    convert = partial(xp.kernels.as_kernel_array, like=outputs[0], strided=True)
    converted = [convert(arg) if hasattr(arg, "shape") else arg for arg in args]
    kernel(*converted)
    for original, given in zip(args, converted, strict=True):
        if any(original is out for out in outputs) and given is not original:
            original[...] = given


def _rng(rng: Any) -> Any:
    """Return ``rng``, or a generator seeded from the operating system."""
    return make_rng() if rng is None else rng


@method(
    equations={
        "scattering": r"s = \frac{q_a^2 q_b^2 \ln\Lambda\, n_a n_b\, \Delta t}{4\pi\varepsilon_0^2 \mu^2\, n_{ab}\, u^3}, \qquad n_{ab} = \frac{1}{V}\sum_\mathrm{pairs}\min(w_i, w_j)",
        "pair": r"\Delta\mathbf{v}_a = \frac{m_b}{m_a + m_b}\Delta\mathbf{u}, \quad \Delta\mathbf{v}_b = -\frac{m_a}{m_a + m_b}\Delta\mathbf{u}, \quad |\mathbf{u} + \Delta\mathbf{u}| = |\mathbf{u}|",
    },
    assumptions=(
        r"binary collisions of the markers in the same cell, paired at random (Takizuka and Abe 1977), a triplet for an odd count",
        r"the angle from Nanbu's cumulative small-angle distribution (any $s$, fit of Pérez et al. 2012) or from Takizuka-Abe ($s \ll 1$)",
        r"unequal weights: the heavier marker of a pair changes with probability $w_\mathrm{light}/w_\mathrm{heavy}$; momentum and energy kept exactly per pair for equal weights, on average otherwise",
        r"the Coulomb logarithm fixed or from the NRL formulas with the cell densities and temperatures; non-relativistic",
    ),
    references=(
        "T. Takizuka and H. Abe, J. Comput. Phys. 25, 205 (1977)",
        "K. Nanbu, Phys. Rev. E 55, 4642 (1997)",
        "F. Pérez et al., Phys. Plasmas 19, 083104 (2012)",
    ),
)
@sp.profile("plasmacoll.coulomb_intra", tags=PROFILE_TAGS)
def collide_intra(
    cells: CellMarkers,
    dt: float,
    cell_volume: float,
    coulomb_log: float | None = None,
    method: str = "nanbu",
    rng: Any = None,
    temperature_floor: float = 0.01,
) -> None:
    """Collide the markers of one species with each other, in place.

    Args:
        cells: The markers of the species, sorted by cell.
        dt: The time step in s.
        cell_volume: The volume of a cell in m^3.
        coulomb_log: The Coulomb logarithm; None computes it per cell
            (`coulomb_logarithm`).
        method: The pair scattering, a key of `METHODS`.
        rng: The generator of the random numbers (`plasmacoll.make_rng`); one
            seeded from the operating system by default.
        temperature_floor: The lowest cell temperature (eV) of the Coulomb logarithm.

    Raises:
        ValueError: If the velocities do not have three components.
    """
    if cells.velocities.shape[0] != 3:
        raise ValueError("Coulomb collisions need three velocity components")
    num = int(cells.weights.shape[0])
    log = _cell_log(cells, cells, coulomb_log, cell_volume, temperature_floor)
    coef = (
        pair_coefficient(cells.charge, cells.charge, cells.mass, cells.mass, log, dt)
        * cells.local_weight**2
        / cell_volume
        * _rank_scale(cells, cells)
    )
    coef = xp.asarray(coef, dtype=float) * xp.ones(cells.num_cells)
    order = xp.empty(num, dtype=xp.int64)
    _run(
        _intra_kernel,
        [cells.velocities],
        coef,
        cells.cell_start,
        cells.velocities,
        cells.weights,
        order,
        _uniforms(num, _rng(rng)),
        float(cells.mass),
        METHODS[method],
    )


@sp.profile("plasmacoll.coulomb_inter", tags=PROFILE_TAGS)
def collide_inter(
    a: CellMarkers,
    b: CellMarkers,
    dt: float,
    cell_volume: float,
    coulomb_log: float | None = None,
    method: str = "nanbu",
    rng: Any = None,
    temperature_floor: float = 0.01,
) -> None:
    """Collide the markers of species ``a`` with those of ``b``, in place.

    Args:
        a: The markers of the first species, sorted by cell.
        b: The markers of the second species, sorted into the same cells.
        dt: The time step in s.
        cell_volume: The volume of a cell in m^3.
        coulomb_log: The Coulomb logarithm; None computes it per cell
            (`coulomb_logarithm`).
        method: The pair scattering, a key of `METHODS`.
        rng: The generator of the random numbers (`plasmacoll.make_rng`); one
            seeded from the operating system by default.
        temperature_floor: The lowest cell temperature (eV) of the Coulomb logarithm.

    Raises:
        ValueError: If the velocities do not have three components or the species
            are sorted into different cells.
    """
    if a.velocities.shape[0] != 3 or b.velocities.shape[0] != 3:
        raise ValueError("Coulomb collisions need three velocity components")
    if a.num_cells != b.num_cells:
        raise ValueError("both species must be sorted into the same cells")
    log = _cell_log(a, b, coulomb_log, cell_volume, temperature_floor)
    coef = (
        pair_coefficient(a.charge, b.charge, a.mass, b.mass, log, dt)
        * a.local_weight
        * b.local_weight
        / cell_volume
        * _rank_scale(a, b)
    )
    coef = xp.asarray(coef, dtype=float) * xp.ones(a.num_cells)
    num_a, num_b = int(a.weights.shape[0]), int(b.weights.shape[0])
    _run(
        _inter_kernel,
        [a.velocities, b.velocities],
        coef,
        a.cell_start,
        b.cell_start,
        a.velocities,
        b.velocities,
        a.weights,
        b.weights,
        xp.empty(num_a, dtype=xp.int64),
        xp.empty(num_b, dtype=xp.int64),
        _uniforms(num_a + num_b, _rng(rng)),
        float(a.mass),
        float(b.mass),
        METHODS[method],
    )
