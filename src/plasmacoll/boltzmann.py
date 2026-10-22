"""A two-term Boltzmann solver for electrons in a gas mixture in a uniform DC field.

:func:`solve_eedf` computes the steady electron energy distribution in a
reduced field ``E/N`` from the same :class:`~plasmacoll.process.CollisionProcess`
objects the Monte Carlo collisions use, and from it the mean energy, the
reduced mobility ``mu N`` and diffusion coefficient ``D N`` and the rate
coefficient of every process. :func:`swarm_table` repeats this over a range of
fields and tabulates the results against the mean energy, the form fluid and
global models look them up in. Every model then gets transport and rates
consistent with the cross sections of the particle simulation.

The method is the classical two-term expansion of Hagelaar and Pitchford,
Plasma Sources Sci. Technol. 14, 722 (2005), the method of BOLSIG+. With the
energy distribution ``F0`` (eV^-3/2) normalised by ``int eps^1/2 F0 deps = 1``,
the steady isotropic part of the Boltzmann equation is

    d/deps (W F0 - D dF0/deps) = S,

    W = -gamma eps^2 sigma_eps,
    D = gamma/3 (E/N)^2 eps / sigma_m + gamma kT eps^2 sigma_eps,

with ``gamma = (2 e / m_e)^1/2``, ``sigma_eps = sum_k x_k (2 m_e / M_k) sigma_k``
over the elastic (momentum-transfer) cross sections, ``sigma_m`` the sum of
all cross sections weighted by the mole fractions ``x_k``, and ``S`` the
inelastic collision terms: excitation moves electrons down by the energy
loss, ionization replaces one electron by two that share the residual energy,
and attachment removes electrons. The change of the electron number is
described either by an exponential growth in time (``growth="temporal"``,
which adds ``nu / (N gamma eps^1/2)`` to ``sigma_m`` and a sink to ``S``) or
in space along the field (``growth="spatial"``, the steady-state Townsend
experiment). The equation is discretised on a uniform energy grid with
Scharfetter-Gummel fluxes between the cells, and the collision terms are
integrated over the cells assuming ``F0`` exponential within each cell, so
that every electron removed by a collision reappears in another cell: the
discretisation conserves the particles exactly.

The solver runs on the host with NumPy, like the rate tables of
:mod:`plasmacoll.rates`; the cross sections may live on either backend.
Superelastic collisions and the gas temperature of the inelastic processes are
not included, and the electron energy of every table is taken in the gas
frame (the difference to the centre-of-mass frame is of order ``m_e / M``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import cunumpy as xp
import numpy as np
import scope_profiler as sp

from plasmacoll._profiling import PROFILE_TAGS
from plasmacoll._types import Array
from plasmacoll.background import NeutralBackground
from plasmacoll.constants import ELECTRON_MASS, ELEMENTARY_CHARGE, TOWNSEND
from plasmacoll.process import CollisionProcess
from plasmacoll.rates import RateTable

__all__ = [
    "GROWTH_MODELS",
    "BoltzmannSolution",
    "SwarmTable",
    "solve_eedf",
    "swarm_table",
]

#: How the change of the electron number by ionization and attachment is modelled.
GROWTH_MODELS = ("temporal", "spatial")

#: ``(2 e / m_e)^1/2``: the speed in m/s of an electron of energy ``eps`` is ``GAMMA eps^1/2``.
GAMMA = math.sqrt(2.0 * ELEMENTARY_CHARGE / ELECTRON_MASS)

_ELASTIC = ("elastic",)
_EXCITATION = ("excitation", "dissociation")
_IONIZATION = ("ionization", "dissociative_ionization")
_ATTACHMENT = ("attachment", "dissociative_attachment")
_KINDS = _ELASTIC + _EXCITATION + _IONIZATION + _ATTACHMENT

#: Gauss-Legendre nodes and weights on [-1, 1] for the collision integrals.
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(3)
#: Decades the distribution should fall from the first to the last cell.
_DECADES = (10.0, 20.0)
_TARGET_DECADES = 15.0
_TINY = 1.0e-300


@dataclass(frozen=True)
class BoltzmannSolution:
    """The steady electron energy distribution in a field and what follows from it.

    Attributes:
        reduced_field: ``E/N`` in Td.
        gas_temperature: ``kT`` of the gas in eV.
        growth: The growth model, one of :data:`GROWTH_MODELS`.
        energy: The cell centres in eV.
        energy_edges: The cell boundaries in eV, one more than ``energy``.
        eepf: The electron energy probability function ``F0`` in eV^-3/2 at
            ``energy``, normalised by ``int eps^1/2 F0 deps = 1``.
        mean_energy: ``<eps> = int eps^3/2 F0 deps`` in eV.
        mobility: The reduced mobility ``mu N`` in 1/(V m s).
        diffusion: The reduced (transverse) diffusion coefficient ``D N`` in 1/(m s).
        rate_coefficients: ``k = gamma int eps sigma F0 deps`` in m^3/s per
            process name, per molecule of the process's own gas.
        net_ionization: ``sum x k_ionization - sum x k_attachment`` in m^3/s, the
            electron growth frequency divided by ``N``.
        townsend_coefficient: The reduced Townsend coefficient ``alpha / N`` in
            m^2 (negative if attachment wins), from ``mu N``, ``D N`` and
            ``net_ionization`` by ``(alpha/N)^2 DN - (alpha/N) muN E/N + k = 0``.
        power: The power per electron divided by ``N`` in eV m^3/s gained from the
            field (``"field"``) and lost in elastic collisions (``"elastic"``),
            in excitation and ionization (``"inelastic"``) and to the change of
            the electron number (``"growth"``).
        iterations: The number of linear solves of the last grid.
    """

    reduced_field: float
    gas_temperature: float
    growth: str
    energy: np.ndarray
    energy_edges: np.ndarray
    eepf: np.ndarray
    mean_energy: float
    mobility: float
    diffusion: float
    rate_coefficients: dict[str, float]
    net_ionization: float
    townsend_coefficient: float
    power: dict[str, float]
    iterations: int

    @property
    def eedf(self) -> np.ndarray:
        """Return the energy distribution ``eps^1/2 F0`` in 1/eV, which integrates to 1."""
        return np.sqrt(self.energy) * self.eepf

    @property
    def drift_velocity(self) -> float:
        """Return the drift velocity ``mu N E/N`` in m/s."""
        return self.mobility * self.reduced_field * TOWNSEND

    @property
    def characteristic_energy(self) -> float:
        """Return ``D / mu`` in eV."""
        return self.diffusion / self.mobility

    @property
    def power_balance(self) -> float:
        """Return the power gained minus the power lost, relative to the power gained.

        It vanishes for an exact solution, so it measures the discretisation
        error in the energy, as the power balance of BOLSIG+ does. Without a
        field it is undefined (NaN).
        """
        if self.power["field"] == 0.0:
            return math.nan
        lost = self.power["elastic"] + self.power["inelastic"] + self.power["growth"]
        return (self.power["field"] - lost) / self.power["field"]


@dataclass(frozen=True)
class SwarmTable:
    """Solutions over a range of reduced fields, tabulated against the mean energy.

    Attributes:
        reduced_field: ``E/N`` in Td.
        mean_energy: The mean energy in eV at each field.
        mobility: ``mu N`` in 1/(V m s) at each field.
        diffusion: ``D N`` in 1/(m s) at each field.
        townsend_coefficient: ``alpha / N`` in m^2 at each field.
        rate_coefficients: ``k`` in m^3/s per process name, at each field.
        solutions: The solution at each field.
    """

    reduced_field: np.ndarray
    mean_energy: np.ndarray
    mobility: np.ndarray
    diffusion: np.ndarray
    townsend_coefficient: np.ndarray
    rate_coefficients: dict[str, np.ndarray]
    solutions: tuple[BoltzmannSolution, ...] = field(repr=False)

    def _column(self, quantity: str) -> np.ndarray:
        """Return the column of ``quantity`` or of the process of that name."""
        if quantity in self.rate_coefficients:
            return self.rate_coefficients[quantity]
        if quantity in (
            "reduced_field",
            "mobility",
            "diffusion",
            "townsend_coefficient",
        ):
            return getattr(self, quantity)
        raise KeyError(f"No column or process {quantity!r} in the swarm table")

    def interpolate(self, quantity: str, mean_energy: float | Array) -> np.ndarray:
        """Return ``quantity`` at ``mean_energy`` (eV), clamped to the table.

        Interpolates in log-log, as :class:`~plasmacoll.rates.RateTable` does
        (a zero stays zero), and linearly in the logarithm of the mean energy
        for a column with negative values such as an attachment-dominated
        Townsend coefficient.

        Args:
            quantity: ``"reduced_field"``, ``"mobility"``, ``"diffusion"``,
                ``"townsend_coefficient"`` or a process name.
            mean_energy: Mean electron energies in eV.

        Raises:
            KeyError: If ``quantity`` is neither a column nor a process.
            ValueError: If the mean energy does not increase along the table.
        """
        values = self._column(quantity)
        if np.any(np.diff(self.mean_energy) <= 0.0):
            raise ValueError(
                "the mean energy does not increase with the field; interpolate "
                "against the reduced field instead"
            )
        log_energy = np.log(self.mean_energy)
        wanted = np.log(
            np.clip(
                np.asarray(xp.to_numpy(xp.asarray(mean_energy)), dtype=float),
                self.mean_energy[0],
                self.mean_energy[-1],
            )
        )
        if np.any(values < 0.0):
            return np.interp(wanted, log_energy, values)
        # as RateTable: log-log where positive, and a zero stays zero
        tiny = np.finfo(float).tiny
        value = np.exp(np.interp(wanted, log_energy, np.log(np.maximum(values, tiny))))
        return np.where(value <= tiny * 10, 0.0, value)

    def rate_table(self, name: str) -> RateTable:
        """Return the rate coefficient of process ``name`` as a :class:`RateTable`.

        The table's temperature is ``2/3`` of the mean energy, the effective
        electron temperature of fluid and global models.
        """
        if name not in self.rate_coefficients:
            raise KeyError(f"No process {name!r} in the swarm table")
        return RateTable(
            2.0 / 3.0 * self.mean_energy, self.rate_coefficients[name], label=name
        )


@dataclass
class _Channel:
    """The collision integrals of one process on the grid.

    Each process is integrated over sub-intervals of the energy of the
    incident electron, ``nodes`` Gauss points each, chosen so that within a
    sub-interval the incident electron's cell and the cell its products land
    in do not change.
    """

    process: CollisionProcess
    fraction: float
    loss: float
    source: np.ndarray  # cell of the incident electron at each node
    offset: np.ndarray  # node energy minus the centre of its cell, eV
    weight: np.ndarray  # gamma eps sigma times the quadrature weight
    energy: np.ndarray  # node energy, eV
    target: np.ndarray | None = None  # cell the products land in at each node
    multiplicity: float = 0.0  # electrons per collision landing in target
    kernel: np.ndarray | None = None  # (cells, nodes): electrons landing per cell

    def flow(self, slope: np.ndarray) -> np.ndarray:
        """Return the collisions at each node per unit ``F0`` of its cell."""
        return self.weight * np.exp(slope[self.source] * self.offset)


def _cell_of(edges: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """Return the cell index of each energy, clamped to the grid."""
    return np.clip(np.searchsorted(edges, energy, side="right") - 1, 0, edges.size - 2)


def _quadrature(
    breaks: np.ndarray, low: float, high: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nodes, weights and the midpoint of their interval between ``breaks``."""
    inside = breaks[(breaks > low) & (breaks < high)]
    points = np.unique(np.concatenate([[low], inside, [high]]))
    half = 0.5 * np.diff(points)
    middle = 0.5 * (points[1:] + points[:-1])
    nodes = (middle[:, None] + half[:, None] * _NODES[None, :]).ravel()
    weights = (half[:, None] * _WEIGHTS[None, :]).ravel()
    return nodes, weights, np.repeat(middle, _NODES.size)


def _sigma(process: CollisionProcess, energy: np.ndarray) -> np.ndarray:
    """Return the cross section of ``process`` in m^2 at ``energy`` (eV) on the host."""
    return np.asarray(
        xp.to_numpy(process.cross_section(xp.asarray(energy))), dtype=float
    )


def _sharing_kernel(
    process: CollisionProcess, residual: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """Return the expected number of electrons per cell after an ionization.

    ``residual`` is the energy ``eps - eps_iz`` the two electrons share, at
    each node; the result has one row per cell and one column per node.
    """
    low = edges[:-1, None]
    high = edges[1:, None]
    share = residual[None, :]
    if process.energy_sharing == "uniform":
        # both electrons uniform on [0, residual]
        covered = np.clip(high, 0.0, share) - np.clip(low, 0.0, share)
        return 2.0 * covered / share
    # Opal-Peterson-Beaty: the secondary on [0, residual / 2] with the density
    # 1 / (1 + (E_s / w)^2), the primary takes the rest
    width = float(process.sharing_energy or 0.0)
    half = 0.5 * share
    norm = np.arctan(half / width)

    def secondary(energy: np.ndarray) -> np.ndarray:
        return np.arctan(np.clip(energy, 0.0, half) / width) / norm

    return (
        secondary(high)
        - secondary(low)
        + secondary(share - low)
        - secondary(share - high)
    )


def _channel(process: CollisionProcess, fraction: float, edges: np.ndarray) -> _Channel:
    """Return the collision integrals of a process on the grid ``edges``."""
    table = np.asarray(xp.to_numpy(process.cross_section.energy), dtype=float)
    threshold = process.cross_section.threshold
    loss = process.loss
    breaks = np.concatenate([edges, table, [threshold]])
    if process.kind in _EXCITATION:
        breaks = np.concatenate([breaks, edges + loss])
    elif process.kind in _IONIZATION and process.energy_sharing == "equal":
        breaks = np.concatenate([breaks, 2.0 * edges + loss])
    low = max(threshold, loss)
    nodes, weights, middle = (
        _quadrature(breaks, low, edges[-1])
        if low < edges[-1]
        else (np.empty(0), np.empty(0), np.empty(0))
    )
    source = _cell_of(edges, middle)
    centres = 0.5 * (edges[1:] + edges[:-1])
    channel = _Channel(
        process=process,
        fraction=fraction,
        loss=loss,
        source=source,
        offset=nodes - centres[source],
        weight=GAMMA * nodes * _sigma(process, nodes) * weights,
        energy=nodes,
    )
    if process.kind in _EXCITATION:
        channel.target = _cell_of(edges, middle - loss)
        channel.multiplicity = 1.0
    elif process.kind in _IONIZATION:
        if process.energy_sharing == "equal":
            channel.target = _cell_of(edges, 0.5 * (middle - loss))
            channel.multiplicity = 2.0
        else:
            channel.kernel = _sharing_kernel(process, nodes - loss, edges)
    return channel


@dataclass
class _Problem:
    """Everything about a field, a gas and a grid that does not depend on ``F0``."""

    reduced_field: float  # V m^2
    gas_temperature: float
    growth: str
    edges: np.ndarray
    channels: list[_Channel]
    elastic: list[_Channel]
    sigma_eps: np.ndarray  # at the inner edges
    sigma_m: np.ndarray  # at the inner edges
    sigma_m_centre: np.ndarray
    names: list[str]  # the process names in the order given

    @property
    def centres(self) -> np.ndarray:
        return 0.5 * (self.edges[1:] + self.edges[:-1])

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges)

    @property
    def inner(self) -> np.ndarray:
        return self.edges[1:-1]

    @property
    def weights(self) -> np.ndarray:
        """Return ``int eps^1/2 deps`` over each cell: the normalisation weights."""
        return 2.0 / 3.0 * np.diff(self.edges**1.5)


def _build(
    processes: Sequence[CollisionProcess],
    fractions: dict[str, float],
    masses: dict[str, float],
    reduced_field: float,
    gas_temperature: float,
    growth: str,
    max_energy: float,
    num_cells: int,
) -> _Problem:
    """Tabulate the cross sections and collision integrals on a grid to ``max_energy``."""
    edges = np.linspace(0.0, max_energy, num_cells + 1)
    inner = edges[1:-1]
    centres = 0.5 * (edges[1:] + edges[:-1])
    sigma_eps = np.zeros_like(inner)
    sigma_m = np.zeros_like(inner)
    sigma_m_centre = np.zeros_like(centres)
    channels = []
    elastic = []
    for process in processes:
        fraction = fractions[process.background]
        sigma_m += fraction * _sigma(process, inner)
        sigma_m_centre += fraction * _sigma(process, centres)
        if process.kind in _ELASTIC:
            ratio = 2.0 * ELECTRON_MASS / masses[process.background]
            sigma_eps += fraction * ratio * _sigma(process, inner)
            elastic.append(_channel(process, fraction, edges))
        else:
            channels.append(_channel(process, fraction, edges))
    if np.any(sigma_m <= 0.0) or np.any(sigma_m_centre <= 0.0):
        raise ValueError(
            "the total cross section must be positive at every energy of the grid; "
            "add an elastic process"
        )
    return _Problem(
        reduced_field=reduced_field,
        gas_temperature=gas_temperature,
        growth=growth,
        edges=edges,
        channels=channels,
        elastic=elastic,
        sigma_eps=sigma_eps,
        sigma_m=sigma_m,
        sigma_m_centre=sigma_m_centre,
        names=[process.name for process in processes],
    )


def _log_slope(eepf: np.ndarray, problem: _Problem) -> np.ndarray:
    """Return ``d ln F0 / deps`` per cell, the exponent of ``F0`` inside the cell."""
    log_f = np.log(np.maximum(eepf, _TINY))
    centres = problem.centres
    slope = np.empty_like(log_f)
    slope[1:-1] = (log_f[2:] - log_f[:-2]) / (centres[2:] - centres[:-2])
    slope[0] = (log_f[1] - log_f[0]) / (centres[1] - centres[0])
    slope[-1] = (log_f[-1] - log_f[-2]) / (centres[-1] - centres[-2])
    # keep exp(slope * offset) finite across a cell
    limit = 50.0 / problem.widths
    return np.clip(slope, -limit, limit)


def _bernoulli(z: np.ndarray) -> np.ndarray:
    """Return ``z / (exp(z) - 1)``, 1 at ``z = 0``."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        value = z / np.expm1(z)
    return np.where(np.abs(z) < 1.0e-10, 1.0 - 0.5 * z, value)


@dataclass
class _Moments:
    """The quantities of an iterate that the next matrix depends on."""

    rates: dict[str, float]
    net: float
    mobility: float
    diffusion: float
    townsend: float
    sigma_tilde: np.ndarray  # at the inner edges
    sigma_tilde_centre: np.ndarray


def _townsend(mobility: float, diffusion: float, net: float, field: float) -> float:
    """Return ``alpha/N`` from ``(alpha/N)^2 DN - (alpha/N) muN E/N + net = 0``."""
    drift = mobility * field
    discriminant = max(drift**2 - 4.0 * diffusion * net, 0.0)
    return (drift - math.sqrt(discriminant)) / (2.0 * diffusion)


def _moments(
    eepf: np.ndarray, slope: np.ndarray, problem: _Problem, townsend: float
) -> _Moments:
    """Return the rates, the transport coefficients and the effective cross section."""
    rates = {}
    net = 0.0
    for channel in problem.channels + problem.elastic:
        rate = float(np.sum(channel.flow(slope) * eepf[channel.source]))
        rates[channel.process.name] = rate
        if channel.process.kind in _IONIZATION:
            net += channel.fraction * rate
        elif channel.process.kind in _ATTACHMENT:
            net -= channel.fraction * rate
    sigma_tilde = problem.sigma_m
    sigma_tilde_centre = problem.sigma_m_centre
    if problem.growth == "temporal":
        sigma_tilde = np.maximum(
            problem.sigma_m + net / (GAMMA * np.sqrt(problem.inner)),
            1.0e-2 * problem.sigma_m,
        )
        sigma_tilde_centre = np.maximum(
            problem.sigma_m_centre + net / (GAMMA * np.sqrt(problem.centres)),
            1.0e-2 * problem.sigma_m_centre,
        )
    mobility = float(-GAMMA / 3.0 * np.sum(problem.inner / sigma_tilde * np.diff(eepf)))
    diffusion = float(
        GAMMA
        / 3.0
        * np.sum(problem.centres / sigma_tilde_centre * eepf * problem.widths)
    )
    if problem.growth == "spatial":
        townsend = _townsend(mobility, diffusion, net, problem.reduced_field)
    return _Moments(
        {name: rates[name] for name in problem.names},
        net,
        mobility,
        diffusion,
        townsend,
        sigma_tilde,
        sigma_tilde_centre,
    )


def _matrix(problem: _Problem, slope: np.ndarray, moments: _Moments) -> np.ndarray:
    """Return the matrix of the discretised equation for the next ``F0``."""
    cells = problem.centres.size
    field_ = problem.reduced_field
    inner = problem.inner
    drift = -GAMMA * inner**2 * problem.sigma_eps
    if problem.growth == "spatial":
        drift = (
            drift - GAMMA / 3.0 * field_ * moments.townsend * inner / problem.sigma_m
        )
    diffusion = (
        GAMMA / 3.0 * field_**2 * inner / moments.sigma_tilde
        + GAMMA * problem.gas_temperature * inner**2 * problem.sigma_eps
    )
    spacing = np.diff(problem.centres)
    diffusion = np.maximum(diffusion, _TINY)
    z = drift * spacing / diffusion
    lower = diffusion / spacing * _bernoulli(-z)  # coefficient of F_{k-1}
    upper = -diffusion / spacing * _bernoulli(z)  # coefficient of F_k
    left = np.arange(cells - 1)
    right = left + 1
    matrix = np.zeros((cells, cells))
    # flux out of the upper boundary of the left cell, into the right cell
    matrix[left, left] += lower
    matrix[left, right] += upper
    matrix[right, left] -= lower
    matrix[right, right] -= upper
    # collisions: the source S enters as -S
    for channel in problem.channels:
        flow = channel.fraction * channel.flow(slope)
        matrix[np.diag_indices(cells)] += np.bincount(
            channel.source, flow, minlength=cells
        )
        if channel.target is not None:
            gain = np.bincount(
                channel.target * cells + channel.source,
                channel.multiplicity * flow,
                minlength=cells * cells,
            )
            matrix -= gain.reshape(cells, cells)
        elif channel.kernel is not None and flow.size:
            starts = np.flatnonzero(np.r_[True, np.diff(channel.source) != 0])
            sources = channel.source[starts]
            matrix[:, sources] -= np.add.reduceat(
                channel.kernel * flow[None, :], starts, axis=1
            )
    if problem.growth == "temporal":
        matrix[np.diag_indices(cells)] += moments.net * problem.weights
    else:
        townsend = moments.townsend
        ratio = inner / problem.sigma_m
        gradient = np.zeros((cells, cells))
        # half of each neighbouring difference, so the sum is the mobility sum
        gradient[left, left] -= 0.5 * ratio
        gradient[left, right] += 0.5 * ratio
        gradient[right, left] -= 0.5 * ratio
        gradient[right, right] += 0.5 * ratio
        growth = GAMMA / 3.0 * townsend * field_ * gradient
        growth[np.diag_indices(cells)] += (
            GAMMA
            / 3.0
            * townsend**2
            * problem.centres
            / problem.sigma_m_centre
            * problem.widths
        )
        matrix -= growth
    return matrix


def _step(problem: _Problem, slope: np.ndarray, moments: _Moments) -> np.ndarray:
    """Return the normalised solution of the linear problem of one iteration."""
    weights = problem.weights
    matrix = _matrix(problem, slope, moments)
    # one equation is redundant (the operator conserves the particles once the
    # growth matches the ionization): replace it by the normalisation
    matrix[0, :] = weights
    rhs = np.zeros(weights.size)
    rhs[0] = 1.0
    new = np.maximum(np.linalg.solve(matrix, rhs), 0.0)
    return new / np.sum(new * weights)


def _solve(
    problem: _Problem,
    eepf: np.ndarray | None,
    rtol: float,
    max_iterations: int,
) -> tuple[np.ndarray, _Moments, np.ndarray, int]:
    """Iterate the linear problem until ``F0`` is self-consistent.

    Without a start, the first iterate has no inelastic shape and no growth.
    """
    if eepf is None:
        cells = problem.centres.size
        no_growth = _Moments(
            {}, 0.0, 0.0, 0.0, 0.0, problem.sigma_m, problem.sigma_m_centre
        )
        eepf = _step(problem, np.zeros(cells), no_growth)
    slope = _log_slope(eepf, problem)
    moments = _moments(eepf, slope, problem, 0.0)
    previous_error = math.inf
    for iteration in range(1, max_iterations + 1):
        new = _step(problem, slope, moments)
        error = float(np.sum(np.abs(new - eepf) * problem.weights))
        if error > previous_error:
            # damp an oscillation between two iterates
            new = 0.5 * (new + eepf)
        previous_error = error
        eepf = new
        slope = _log_slope(eepf, problem)
        moments = _moments(eepf, slope, problem, moments.townsend)
        if error < rtol:
            return eepf, moments, slope, iteration
    raise RuntimeError(
        f"the Boltzmann solver did not converge in {max_iterations} iterations; "
        "raise max_iterations or num_cells"
    )


def _power(
    problem: _Problem, eepf: np.ndarray, slope: np.ndarray, moments: _Moments
) -> tuple[float, dict[str, float]]:
    """Return the mean energy and the power terms of the energy balance."""
    edges = problem.edges
    inner = problem.inner
    mean_energy = float(np.sum(eepf * 0.4 * np.diff(edges**2.5)))
    field_ = problem.reduced_field
    spacing = np.diff(problem.centres)
    mean_f = 0.5 * (eepf[1:] + eepf[:-1])
    elastic = float(
        GAMMA
        * np.sum(
            inner**2
            * problem.sigma_eps
            * (mean_f * spacing + problem.gas_temperature * np.diff(eepf))
        )
    )
    inelastic = 0.0
    removed = 0.0
    for channel in problem.channels:
        rate = channel.fraction * moments.rates[channel.process.name]
        if channel.process.kind in _ATTACHMENT:
            flow = channel.flow(slope) * eepf[channel.source]
            removed += channel.fraction * float(np.sum(channel.energy * flow))
        else:
            inelastic += channel.loss * rate
    townsend = moments.townsend
    if problem.growth == "temporal":
        gained = field_**2 * moments.mobility
        growth = removed + moments.net * mean_energy
    else:
        gained = field_ * (field_ * moments.mobility - townsend * moments.diffusion)
        ratio = inner**2 / problem.sigma_m
        source = (
            GAMMA
            / 3.0
            * townsend
            * (
                field_ * np.sum(ratio * np.diff(eepf))
                + townsend * np.sum(ratio * mean_f * spacing)
            )
        )
        growth = removed - source
    return mean_energy, {
        "field": float(gained),
        "elastic": elastic,
        "inelastic": inelastic,
        "growth": float(growth),
    }


def _mixture(
    processes: Sequence[CollisionProcess],
    mixture: Sequence[NeutralBackground],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return the mole fraction and the mass of every gas, and check the processes."""
    total = sum(background.density for background in mixture)
    if not mixture or total <= 0.0:
        raise ValueError("the mixture needs a gas with a positive density")
    fractions = {background.name: background.density / total for background in mixture}
    masses = {background.name: background.mass for background in mixture}
    names = [process.name for process in processes]
    if len(set(names)) != len(names):
        raise ValueError(f"process names must be unique, got {names}")
    for process in processes:
        if process.kind not in _KINDS:
            raise ValueError(
                f"{process.name}: the Boltzmann solver takes electron-impact "
                f"processes of the kinds {_KINDS}, not {process.kind!r}"
            )
        if process.background not in fractions:
            raise ValueError(
                f"{process.name}: no gas {process.background!r} in the mixture"
            )
        if process.kind in _ELASTIC and process.scattering != "isotropic":
            raise ValueError(
                f"{process.name}: the two-term expansion needs the momentum-transfer "
                "cross section, i.e. an elastic process with isotropic scattering"
            )
    return fractions, masses


@sp.profile("plasmacoll.solve_eedf", tags=PROFILE_TAGS)
def solve_eedf(
    processes: Sequence[CollisionProcess],
    mixture: Sequence[NeutralBackground],
    reduced_field: float,
    gas_temperature: float | None = None,
    *,
    growth: str = "temporal",
    num_cells: int = 200,
    max_energy: float | None = None,
    rtol: float = 1.0e-8,
    max_iterations: int = 200,
) -> BoltzmannSolution:
    """Return the steady electron energy distribution in a uniform DC field.

    Args:
        processes: The electron-impact processes: ``elastic`` (with the
            momentum-transfer cross section), ``excitation``, ``dissociation``,
            ``ionization``, ``dissociative_ionization``, ``attachment`` and
            ``dissociative_attachment``. Ionization shares the residual energy
            as its ``energy_sharing`` says.
        mixture: The gases the processes refer to; the mole fractions are
            their densities divided by the total, and their masses set the
            elastic energy loss.
        reduced_field: ``E/N`` in Td (1 Td = 1e-21 V m^2).
        gas_temperature: ``kT`` of the gas in eV; the density-weighted mean of
            the mixture's temperatures if None.
        growth: ``"temporal"`` for an electron number growing exponentially in
            time, ``"spatial"`` for one growing along the field (steady-state
            Townsend); they agree unless ionization or attachment is strong.
        num_cells: Number of cells of the uniform energy grid.
        max_energy: The grid's upper end in eV. If None, it is chosen so that
            the distribution falls by 10 to 20 decades across the grid.
        rtol: Convergence tolerance on the change of ``F0`` between iterations,
            in the norm ``int eps^1/2 |dF0| deps``.
        max_iterations: Iterations of the nonlinear problem per grid.

    Raises:
        ValueError: For processes the solver does not take, gases missing
            from the mixture, or a zero field in a gas at zero temperature.
        RuntimeError: If the iteration does not converge.
    """
    if growth not in GROWTH_MODELS:
        raise ValueError(f"growth must be one of {GROWTH_MODELS}")
    fractions, masses = _mixture(processes, mixture)
    if gas_temperature is None:
        total = sum(background.density for background in mixture)
        gas_temperature = (
            sum(background.density * background.temperature for background in mixture)
            / total
        )
    if reduced_field < 0.0 or gas_temperature < 0.0:
        raise ValueError("reduced_field and gas_temperature must be >= 0")
    if reduced_field == 0.0 and gas_temperature == 0.0:
        raise ValueError("a zero field in a gas at zero temperature has no solution")
    if num_cells < 10:
        raise ValueError("num_cells must be >= 10")
    if max_energy is not None and max_energy <= 0.0:
        raise ValueError("max_energy must be > 0")

    field_ = reduced_field * TOWNSEND
    temperature = float(gas_temperature)

    def build(energy: float) -> _Problem:
        return _build(
            processes,
            fractions,
            masses,
            field_,
            temperature,
            growth,
            energy,
            num_cells,
        )

    energy = 10.0 if max_energy is None else max_energy
    problem = build(energy)
    eepf, moments, slope, iterations = _solve(problem, None, rtol, max_iterations)
    # ranges known to span too few and too many decades of F0
    too_short, too_long = 0.0, math.inf
    final = False
    for _ in range(40):
        decades = math.log10(eepf.max() / max(eepf[-1], _TINY))
        if final or max_energy is not None or _DECADES[0] <= decades <= _DECADES[1]:
            break
        if decades < _DECADES[0]:
            too_short = energy
        else:
            too_long = energy
        if too_long - too_short < 1.0e-3 * too_long:
            # F0 drops too steeply (at a threshold) to hit the range: keep the longer
            energy, final = too_long, True
        elif too_short > 0.0 and too_long < math.inf:
            energy = 0.5 * (too_short + too_long)
        else:
            # F0 falls like exp(-eps / T) or faster: change the range by at most 2x
            energy *= min(max(math.sqrt(_TARGET_DECADES / decades), 0.5), 2.0)
        previous, problem = problem, build(energy)
        # start from the previous solution, stretched onto the new grid
        start = np.exp(
            np.interp(
                problem.centres, previous.centres, np.log(np.maximum(eepf, _TINY))
            )
        )
        eepf, moments, slope, iterations = _solve(problem, start, rtol, max_iterations)
    mean_energy, power = _power(problem, eepf, slope, moments)
    return BoltzmannSolution(
        reduced_field=float(reduced_field),
        gas_temperature=temperature,
        growth=growth,
        energy=problem.centres,
        energy_edges=problem.edges,
        eepf=eepf,
        mean_energy=mean_energy,
        mobility=moments.mobility,
        diffusion=moments.diffusion,
        rate_coefficients=moments.rates,
        net_ionization=moments.net,
        townsend_coefficient=_townsend(
            moments.mobility, moments.diffusion, moments.net, field_
        ),
        power=power,
        iterations=iterations,
    )


@sp.profile("plasmacoll.swarm_table", tags=PROFILE_TAGS)
def swarm_table(
    processes: Sequence[CollisionProcess],
    mixture: Sequence[NeutralBackground],
    reduced_fields: Sequence[float] | Array,
    gas_temperature: float | None = None,
    **options: Any,
) -> SwarmTable:
    """Solve at each reduced field (Td) and tabulate the results.

    Args:
        processes: The electron-impact processes, as for :func:`solve_eedf`.
        mixture: The gases, as for :func:`solve_eedf`.
        reduced_fields: ``E/N`` in Td, increasing.
        gas_temperature: ``kT`` of the gas in eV, as for :func:`solve_eedf`.
        **options: Further keyword arguments of :func:`solve_eedf`.
    """
    fields = np.asarray(xp.to_numpy(xp.asarray(reduced_fields, dtype=float)))
    if fields.ndim != 1 or fields.size < 2 or np.any(np.diff(fields) <= 0.0):
        raise ValueError("reduced_fields must be increasing, at least two")
    solutions = tuple(
        solve_eedf(processes, mixture, float(value), gas_temperature, **options)
        for value in fields
    )
    names = list(solutions[0].rate_coefficients)
    return SwarmTable(
        reduced_field=fields,
        mean_energy=np.array([s.mean_energy for s in solutions]),
        mobility=np.array([s.mobility for s in solutions]),
        diffusion=np.array([s.diffusion for s in solutions]),
        townsend_coefficient=np.array([s.townsend_coefficient for s in solutions]),
        rate_coefficients={
            name: np.array([s.rate_coefficients[name] for s in solutions])
            for name in names
        },
        solutions=solutions,
    )
