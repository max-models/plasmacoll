"""A zero-dimensional (spatially homogeneous) reactor advanced by collisions and fields.

:class:`ZeroDReactor` evolves the velocity distributions and densities of a
set of species with :class:`~plasmacoll.mcc.MonteCarloCollisions`, optionally
with collisions between charged particles
(:class:`~plasmacoll.charged.ChargedCollisions`), in uniform electric and
magnetic fields (constant, or an electric field that varies in time, e.g. at
radio frequency), and optionally with a neutral gas that heats, flows and is
depleted by the collisions. There is no position push and no boundary. It is
the tool for testing a cross-section set or a reaction chain (rate
coefficients, thresholds, ionization against attachment, relaxation, drift
velocity, mean energy and Townsend coefficients in a field) without the cost
and the confounding factors of a spatial simulation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import cast

import cunumpy as xp
import numpy as np

from plasmacoll.charged import ChargedCollisions
from plasmacoll.constants import BOLTZMANN, ELEMENTARY_CHARGE
from plasmacoll.markers import ParticleArrays
from plasmacoll.mcc import BackgroundTransfer, MCCDiagnostics, MonteCarloCollisions
from plasmacoll.population import merge_markers

__all__ = ["ReactorHistory", "ReactorState", "ZeroDReactor"]

FieldLike = Sequence[float] | Callable[[float], Sequence[float]]


@dataclass
class ReactorState:
    """The state of the reactor after a step.

    Attributes:
        time: Elapsed time in s.
        density: Number density in m^-3 per species.
        mean_energy_ev: Mean kinetic energy in eV per species.
        temperature: Kinetic temperature in K about the mean velocity, per species.
        num_markers: Number of markers per species.
        collisions: Cumulative number of collisions (markers) per process or
            reaction name.
        mean_velocity: Weighted mean velocity in m/s per species, shape ``(3,)``:
            the drift velocity in a field.
        weighted_collisions: Cumulative number of physical collisions per
            process or reaction name.
        gas_density: Density in m^-3 of every neutral background.
        gas_temperature: Temperature in K of every neutral background.
    """

    time: float
    density: dict[str, float] = field(default_factory=dict)
    mean_energy_ev: dict[str, float] = field(default_factory=dict)
    temperature: dict[str, float] = field(default_factory=dict)
    num_markers: dict[str, int] = field(default_factory=dict)
    collisions: dict[str, int] = field(default_factory=dict)
    mean_velocity: dict[str, np.ndarray] = field(default_factory=dict)
    weighted_collisions: dict[str, float] = field(default_factory=dict)
    gas_density: dict[str, float] = field(default_factory=dict)
    gas_temperature: dict[str, float] = field(default_factory=dict)


@dataclass
class ReactorHistory:
    """Snapshots of the reactor as arrays over time, ready to plot.

    Attributes:
        time: Times in s, shape ``(T,)``.
        density: Number density in m^-3 per species, each of shape ``(T,)``.
        mean_energy_ev: Mean kinetic energy in eV per species.
        temperature: Kinetic temperature in K per species.
        num_markers: Number of markers per species.
        collisions: Cumulative number of collisions (markers) per process name.
        mean_velocity: Mean velocity in m/s per species, each of shape ``(T, 3)``.
        weighted_collisions: Cumulative physical collisions per process name.
        gas_density: Density in m^-3 per background.
        gas_temperature: Temperature in K per background.
        volume: The reactor volume in m^3.
    """

    time: np.ndarray
    density: dict[str, np.ndarray]
    mean_energy_ev: dict[str, np.ndarray]
    temperature: dict[str, np.ndarray]
    num_markers: dict[str, np.ndarray]
    collisions: dict[str, np.ndarray]
    mean_velocity: dict[str, np.ndarray] = field(default_factory=dict)
    weighted_collisions: dict[str, np.ndarray] = field(default_factory=dict)
    gas_density: dict[str, np.ndarray] = field(default_factory=dict)
    gas_temperature: dict[str, np.ndarray] = field(default_factory=dict)
    volume: float = 1.0

    @classmethod
    def from_states(
        cls, states: list[ReactorState], volume: float = 1.0
    ) -> ReactorHistory:
        """Stack a list of states into arrays."""

        def stack(attribute: str) -> dict[str, np.ndarray]:
            keys = dict.fromkeys(
                key for state in states for key in getattr(state, attribute)
            )
            return {
                key: np.asarray(
                    [getattr(state, attribute).get(key, 0) for state in states]
                )
                for key in keys
            }

        return cls(
            time=np.asarray([state.time for state in states]),
            density=stack("density"),
            mean_energy_ev=stack("mean_energy_ev"),
            temperature=stack("temperature"),
            num_markers=stack("num_markers"),
            collisions=stack("collisions"),
            mean_velocity=stack("mean_velocity"),
            weighted_collisions=stack("weighted_collisions"),
            gas_density=stack("gas_density"),
            gas_temperature=stack("gas_temperature"),
            volume=volume,
        )

    def _window(self, start_time: float) -> np.ndarray:
        window = self.time >= start_time
        if int(np.count_nonzero(window)) < 2:
            raise ValueError("need at least two records after start_time")
        return window

    def rate_coefficient(
        self, process: str, species: str, partner: str, start_time: float = 0.0
    ) -> float:
        """Return the rate coefficient ``k`` (m^3/s) of a process from the counts.

        ``k = Delta N / (V int n_species n_partner dt)``, with ``Delta N`` the
        physical collisions of ``process`` after ``start_time`` and the
        integral by the trapezoidal rule over the records.

        Args:
            process: The process or reaction name.
            species: The incident species.
            partner: A background name (its gas density) or a species name.
            start_time: Count from this time on, e.g. once a swarm is steady.
        """
        window = self._window(start_time)
        time = self.time[window]
        if partner in self.gas_density:
            partner_density = self.gas_density[partner][window]
        else:
            partner_density = self.density[partner][window]
        product = self.density[species][window] * partner_density
        exposure = self.volume * float(
            np.sum(0.5 * (product[1:] + product[:-1]) * np.diff(time))
        )
        if exposure <= 0.0:
            return 0.0
        count = self.weighted_collisions[process][window]
        return float(count[-1] - count[0]) / exposure

    def townsend_coefficient(
        self, process: str, species: str, gas: str, start_time: float = 0.0
    ) -> float:
        """Return the Townsend coefficient ``k n_gas / |v_d|`` (1/m) of a process.

        For ``process`` an ionization this is the first Townsend coefficient
        ``alpha``; for an attachment, ``eta``. Divide by the gas density for the
        reduced coefficient ``alpha / N``. ``v_d`` is the mean drift velocity
        of ``species`` after ``start_time``.

        Raises:
            ValueError: If ``species`` does not drift.
        """
        window = self._window(start_time)
        rate = self.rate_coefficient(process, species, gas, start_time)
        drift = float(np.linalg.norm(self.mean_velocity[species][window].mean(axis=0)))
        if drift <= 0.0:
            raise ValueError(
                f"{species} does not drift; the Townsend coefficient is undefined"
            )
        return rate * float(self.gas_density[gas][window].mean()) / drift


class ZeroDReactor:
    """A homogeneous reactor of volume ``volume`` advanced by collisions and fields.

    Each step accelerates every charged species in the electric and magnetic
    fields (Boris push, ``E`` evaluated at the middle of the step), collides
    them with the neutral gas, collides the charged species with each other if
    ``charged`` is given, updates the gas from what it received if
    ``evolve_backgrounds``, and merges species that exceed ``max_markers``.

    Args:
        collisions: The collision operator; it knows the mass of every species.
        species: The initial markers per species. Species of the operator that
            are missing here start empty, so products need not be listed.
        volume: Reactor volume in m^3; densities are total weights over volume.
        electric_field: The uniform electric field in V/m, three components, or
            a function of time returning them (e.g. an RF field).
        magnetic_field: The uniform magnetic field in T, three components.
        charges: The charge of each species in units of the elementary charge
            (e.g. ``{"e": -1, "Ar+": 1}``); species not listed are neutral.
            Taken from ``charged`` if None.
        charged: Collisions between charged particles, with the reactor
            volume as the single cell.
        max_markers: Merge a species' markers down to half this number with
            :func:`~plasmacoll.population.merge_markers` whenever it exceeds
            it, conserving weight, momentum and energy; a number for every
            species or one per species name. None never merges.
        evolve_backgrounds: Update every background's density, temperature and
            flow from the particles, momentum and energy the collisions gave it
            (gas depletion and heating of a closed volume).
        seed: Seed of the random directions of merged markers.

    Raises:
        KeyError: If a species has no mass in the operator.
    """

    def __init__(
        self,
        collisions: MonteCarloCollisions,
        species: Mapping[str, ParticleArrays],
        volume: float = 1.0,
        electric_field: FieldLike = (0.0, 0.0, 0.0),
        magnetic_field: Sequence[float] = (0.0, 0.0, 0.0),
        charges: Mapping[str, float] | None = None,
        charged: ChargedCollisions | None = None,
        max_markers: int | Mapping[str, int] | None = None,
        evolve_backgrounds: bool = False,
        seed: int | None = None,
    ) -> None:
        """Check the species and start every missing one without markers."""
        if volume <= 0.0:
            raise ValueError(f"volume must be > 0, got {volume}")
        masses = collisions.species_masses
        unknown = set(species) - set(masses)
        if unknown:
            raise KeyError(f"No mass given for species {sorted(unknown)}")
        if charges is None and charged is not None:
            charges = {
                name: charge
                for name, charge in charged.charges.items()
                if name in masses
            }
        charges = {name: float(charge) for name, charge in (charges or {}).items()}
        unknown = set(charges) - set(masses)
        if unknown:
            raise KeyError(f"Charge given for unknown species {sorted(unknown)}")
        field_function: Callable[[float], Sequence[float]] | None = None
        if callable(electric_field):
            field_function = cast(Callable[[float], Sequence[float]], electric_field)
            constant_field = (0.0, 0.0, 0.0)
        else:
            constant_field = _vector(electric_field, "electric_field")
        if isinstance(max_markers, Mapping):
            limits = {name: int(limit) for name, limit in max_markers.items()}
        elif max_markers is not None:
            limits = dict.fromkeys(masses, int(max_markers))
        else:
            limits = {}
        if any(limit < 4 for limit in limits.values()):
            raise ValueError("max_markers must be >= 4")
        ndim = {markers.positions.shape[1] for markers in species.values()} or {1}
        if len(ndim) != 1:
            raise ValueError("all species need the same number of position components")
        self._collisions = collisions
        self._charged = charged
        self._masses = masses
        self._volume = float(volume)
        self._species = {
            name: species[name] if name in species else ParticleArrays.empty(*ndim)
            for name in masses
        }
        self._counts: dict[str, int] = {}
        self._weighted: dict[str, float] = {}
        self._transfer: dict[str, BackgroundTransfer] = {}
        self._time = 0.0
        self._iteration = 0
        self._charges = charges
        self._electric_field: FieldLike = electric_field
        self._field_function = field_function
        self._constant_field = constant_field
        self._magnetic_field = _vector(magnetic_field, "magnetic_field")
        self._limits = limits
        self._evolve = bool(evolve_backgrounds)
        self._merge_rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    def state(self) -> ReactorState:
        """Return the current densities, energies, temperatures and counts."""
        backgrounds = self._collisions.backgrounds
        return ReactorState(
            time=self._time,
            density={
                name: markers.total_weight / self._volume
                for name, markers in self._species.items()
            },
            mean_energy_ev={
                name: markers.mean_energy_ev(self._masses[name])
                for name, markers in self._species.items()
            },
            temperature={
                name: markers.temperature(self._masses[name])
                for name, markers in self._species.items()
            },
            num_markers={name: len(markers) for name, markers in self._species.items()},
            collisions=dict(self._counts),
            mean_velocity={
                name: _mean_velocity(markers) for name, markers in self._species.items()
            },
            weighted_collisions=dict(self._weighted),
            gas_density={name: gas.density for name, gas in backgrounds.items()},
            gas_temperature={
                name: gas.temperature for name, gas in backgrounds.items()
            },
        )

    def energy_distribution(
        self,
        species: str,
        bins: int | Sequence[float] = 100,
        max_energy: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the energy distribution function of a species.

        Args:
            species: The species.
            bins: The number of equal bins from 0 to ``max_energy``, or the
                bin edges in eV.
            max_energy: The upper edge in eV for a number of bins; the largest
                marker energy if None.

        Returns:
            The bin centres in eV and ``f(E)`` in 1/eV, normalized so that
            ``sum(f * widths) = 1`` (all zeros without markers).
        """
        markers = self._species[species]
        squared = xp.sum(markers.velocities**2, axis=1)
        energy = (
            0.5 * self._masses[species] * np.asarray(xp.to_numpy(squared))
        ) / ELEMENTARY_CHARGE
        weights = np.asarray(xp.to_numpy(markers.weights))
        if isinstance(bins, int):
            top = (
                max_energy if max_energy is not None else float(energy.max(initial=1.0))
            )
            edges = np.linspace(0.0, top, bins + 1)
        else:
            edges = np.asarray(bins, dtype=float)
        counts, edges = np.histogram(energy, bins=edges, weights=weights)
        widths = np.diff(edges)
        total = float(np.sum(counts))
        density = counts / (total * widths) if total > 0.0 else np.zeros_like(widths)
        return 0.5 * (edges[1:] + edges[:-1]), density

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(self, dt: float) -> ReactorState:
        """Advance one step of ``dt`` (s).

        Returns:
            The state after the step.
        """
        self._advance(dt)
        return self.state()

    def _field_at(self, time: float) -> tuple[float, float, float]:
        if self._field_function is None:
            return self._constant_field
        return _vector(self._field_function(time), "electric_field")

    def _push(self, dt: float) -> None:
        """Accelerate the charged species with the Boris scheme."""
        electric = self._field_at(self._time + 0.5 * dt)
        magnetic = self._magnetic_field
        if not any(electric) and not any(magnetic):
            return
        electric_vector = xp.asarray(electric, dtype=float)
        magnetic_vector = xp.asarray(magnetic, dtype=float)
        for name, charge in self._charges.items():
            markers = self._species[name]
            if charge == 0.0 or len(markers) == 0:
                continue
            ratio = charge * ELEMENTARY_CHARGE / self._masses[name]
            half_kick = 0.5 * ratio * dt * electric_vector
            velocities = markers.velocities + half_kick
            if any(magnetic):
                rotation = 0.5 * ratio * dt * magnetic_vector
                scale = 2.0 / (1.0 + float(xp.sum(rotation**2)))
                primed = velocities + xp.cross(velocities, rotation)
                velocities = velocities + scale * xp.cross(primed, rotation)
            markers.velocities = velocities + half_kick

    def _advance(self, dt: float) -> None:
        """Advance one step without computing the diagnostics of the state."""
        self._push(dt)
        diagnostics = self._collisions.collide_species(self._species, dt)
        for species_diagnostics in diagnostics.values():
            for process, count in species_diagnostics.counts.items():
                self._counts[process] = self._counts.get(process, 0) + count
            for process, weight in species_diagnostics.weighted_counts.items():
                self._weighted[process] = self._weighted.get(process, 0.0) + weight
            for background, transfer in species_diagnostics.transfer.items():
                self._transfer.setdefault(background, BackgroundTransfer()).add(
                    transfer
                )
        if self._charged is not None:
            charged = self._charged.collide_species(
                self._species, dt, cell_volume=self._volume
            )
            for reaction, count in charged.reactions.items():
                self._counts[reaction] = self._counts.get(reaction, 0) + count
            for reaction, weight in charged.weighted_reactions.items():
                self._weighted[reaction] = self._weighted.get(reaction, 0.0) + weight
        if self._evolve:
            self._update_backgrounds(diagnostics)
        for name, limit in self._limits.items():
            if name in self._species and len(self._species[name]) > limit:
                merge_markers(self._species[name], limit // 2, rng=self._merge_rng)
        self._iteration += 1
        self._time += dt

    def _update_backgrounds(self, diagnostics: Mapping[str, MCCDiagnostics]) -> None:
        """Apply this step's transfers to the gas of the closed volume.

        The gas keeps its number of particles, momentum and energy (thermal
        plus flow) up to what the collisions gave it, from which its new
        density, flow velocity and temperature follow.
        """
        step_transfer: dict[str, BackgroundTransfer] = {}
        for species_diagnostics in diagnostics.values():
            for background, transfer in species_diagnostics.transfer.items():
                step_transfer.setdefault(background, BackgroundTransfer()).add(transfer)
        for name, transfer in step_transfer.items():
            gas = self._collisions.backgrounds[name]
            number = gas.density * self._volume
            drift = np.asarray(gas.drift, dtype=float)
            momentum = number * gas.mass * drift + transfer.momentum
            energy = (
                number
                * (
                    1.5 * BOLTZMANN * gas.temperature
                    + 0.5 * gas.mass * float(drift @ drift)
                )
                + transfer.energy
            )
            remaining = max(number - transfer.particles, 0.0)
            if remaining <= 0.0:
                self._collisions.set_background(replace(gas, density=0.0))
                continue
            new_drift = momentum / (remaining * gas.mass)
            thermal = energy / remaining - 0.5 * gas.mass * float(new_drift @ new_drift)
            self._collisions.set_background(
                replace(
                    gas,
                    density=remaining / self._volume,
                    temperature=max(thermal / (1.5 * BOLTZMANN), 0.0),
                    drift=(
                        float(new_drift[0]),
                        float(new_drift[1]),
                        float(new_drift[2]),
                    ),
                )
            )

    def run(self, dt: float, num_steps: int, every: int = 1) -> ReactorHistory:
        """Advance ``num_steps`` steps of ``dt`` (s).

        Args:
            dt: The time step in s.
            num_steps: The number of steps.
            every: Record the state every ``every`` steps.

        Returns:
            The initial state and the recorded states as arrays over time.
        """
        if every < 1:
            raise ValueError("every must be >= 1")
        states = [self.state()]
        for index in range(1, num_steps + 1):
            self._advance(dt)
            if index % every == 0:
                states.append(self.state())
        return ReactorHistory.from_states(states, volume=self._volume)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def species(self) -> dict[str, ParticleArrays]:
        """Return the markers per species."""
        return dict(self._species)

    @property
    def collisions(self) -> MonteCarloCollisions:
        """Return the collision operator."""
        return self._collisions

    @property
    def time(self) -> float:
        """Return the elapsed time in s."""
        return self._time

    @property
    def iteration(self) -> int:
        """Return the number of steps taken."""
        return self._iteration

    @property
    def volume(self) -> float:
        """Return the reactor volume in m^3."""
        return self._volume

    @property
    def electric_field(self) -> FieldLike:
        """Return the electric field in V/m, or the function of time giving it."""
        return self._electric_field

    @property
    def magnetic_field(self) -> tuple[float, float, float]:
        """Return the magnetic field in T."""
        return self._magnetic_field

    @property
    def transfer(self) -> dict[str, BackgroundTransfer]:
        """Return the cumulative momentum, energy and particles given to each background."""
        return dict(self._transfer)


def _vector(values: Iterable[float], name: str) -> tuple[float, float, float]:
    """Return three finite floats, or raise if ``values`` is not such a vector."""
    vector = tuple(float(component) for component in values)
    if len(vector) != 3 or not all(math.isfinite(component) for component in vector):
        raise ValueError(f"{name} must have three components, all finite")
    return (vector[0], vector[1], vector[2])


def _mean_velocity(markers: ParticleArrays) -> np.ndarray:
    """Return the weighted mean velocity of ``markers`` (zero if empty) on the host."""
    total = markers.total_weight
    if total <= 0.0:
        return np.zeros(3)
    mean = xp.sum(markers.weights[:, None] * markers.velocities, axis=0) / total
    return np.asarray(xp.to_numpy(mean), dtype=float)
