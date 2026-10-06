"""A zero-dimensional (spatially homogeneous) reactor advanced by collisions and a field.

:class:`ZeroDReactor` evolves the velocity distributions and densities of a
set of species with :class:`~plasmacoll.mcc.MonteCarloCollisions`, optionally in
a uniform, constant electric field that accelerates the charged species
(a swarm experiment); there is no position push and no boundary. It is the
tool for testing a cross-section set or a reaction chain (rate coefficients,
thresholds, ionization against attachment, relaxation to the gas temperature,
drift velocity and mean energy in a field) without the cost and the
confounding factors of a spatial simulation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import cunumpy as xp
import numpy as np

from plasmacoll.constants import ELEMENTARY_CHARGE
from plasmacoll.markers import ParticleArrays
from plasmacoll.mcc import MonteCarloCollisions
from plasmacoll.population import merge_markers

__all__ = ["ReactorHistory", "ReactorState", "ZeroDReactor"]


@dataclass
class ReactorState:
    """The state of the reactor after a step.

    Attributes:
        time: Elapsed time in s.
        density: Number density in m^-3 per species.
        mean_energy_ev: Mean kinetic energy in eV per species.
        temperature: Kinetic temperature in K about the mean velocity, per species.
        num_markers: Number of markers per species.
        collisions: Cumulative number of collisions per process name.
        mean_velocity: Weighted mean velocity in m/s per species, shape ``(3,)``:
            the drift velocity in a field.
    """

    time: float
    density: dict[str, float] = field(default_factory=dict)
    mean_energy_ev: dict[str, float] = field(default_factory=dict)
    temperature: dict[str, float] = field(default_factory=dict)
    num_markers: dict[str, int] = field(default_factory=dict)
    collisions: dict[str, int] = field(default_factory=dict)
    mean_velocity: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class ReactorHistory:
    """Snapshots of the reactor as arrays over time, ready to plot.

    Attributes:
        time: Times in s, shape ``(T,)``.
        density: Number density in m^-3 per species, each of shape ``(T,)``.
        mean_energy_ev: Mean kinetic energy in eV per species.
        temperature: Kinetic temperature in K per species.
        num_markers: Number of markers per species.
        collisions: Cumulative number of collisions per process name.
        mean_velocity: Mean velocity in m/s per species, each of shape ``(T, 3)``.
    """

    time: np.ndarray
    density: dict[str, np.ndarray]
    mean_energy_ev: dict[str, np.ndarray]
    temperature: dict[str, np.ndarray]
    num_markers: dict[str, np.ndarray]
    collisions: dict[str, np.ndarray]
    mean_velocity: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_states(cls, states: list[ReactorState]) -> ReactorHistory:
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
        )


class ZeroDReactor:
    """A homogeneous reactor of volume ``volume`` advanced by collisions and a field.

    Each step first accelerates every charged species in the electric field,
    ``v += q E dt / m``, then collides all species.

    Args:
        collisions: The collision operator; it knows the mass of every species.
        species: The initial markers per species. Species of the operator that
            are missing here start empty, so products need not be listed.
        volume: Reactor volume in m^3; densities are total weights over volume.
        electric_field: The uniform electric field in V/m, three components.
        charges: The charge of each species in units of the elementary charge
            (e.g. ``{"e": -1, "Ar+": 1}``); species not listed are neutral.
            Needed for the species the field should accelerate.
        max_markers: Merge a species' markers down to half this number with
            :func:`~plasmacoll.population.merge_markers` whenever it exceeds
            it, conserving weight, momentum and energy; a number for every
            species or one per species name. None never merges.
        seed: Seed of the random directions of merged markers.

    Raises:
        KeyError: If a species has no mass in the operator.
    """

    def __init__(
        self,
        collisions: MonteCarloCollisions,
        species: Mapping[str, ParticleArrays],
        volume: float = 1.0,
        electric_field: Sequence[float] = (0.0, 0.0, 0.0),
        charges: Mapping[str, float] | None = None,
        max_markers: int | Mapping[str, int] | None = None,
        seed: int | None = None,
    ) -> None:
        """Check the species and start every missing one without markers."""
        if volume <= 0.0:
            raise ValueError(f"volume must be > 0, got {volume}")
        masses = collisions.species_masses
        unknown = set(species) - set(masses)
        if unknown:
            raise KeyError(f"No mass given for species {sorted(unknown)}")
        charges = dict(charges or {})
        unknown = set(charges) - set(masses)
        if unknown:
            raise KeyError(f"Charge given for unknown species {sorted(unknown)}")
        field_vector = tuple(float(component) for component in electric_field)
        if len(field_vector) != 3:
            raise ValueError("electric_field must have three components")
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
        self._masses = masses
        self._volume = float(volume)
        self._species = {
            name: species[name] if name in species else ParticleArrays.empty(*ndim)
            for name in masses
        }
        self._counts: dict[str, int] = {}
        self._time = 0.0
        self._iteration = 0
        self._charges = {name: float(charge) for name, charge in charges.items()}
        self._electric_field = field_vector
        self._limits = limits
        self._merge_rng = np.random.default_rng(seed)

    def state(self) -> ReactorState:
        """Return the current densities, energies, temperatures and counts."""
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
        )

    def step(self, dt: float) -> ReactorState:
        """Accelerate, collide and (if needed) merge every species for ``dt`` (s).

        Returns:
            The state after the step.
        """
        self._advance(dt)
        return self.state()

    def _advance(self, dt: float) -> None:
        """Advance one step without computing the diagnostics of the state."""
        if any(self._electric_field):
            field_vector = xp.asarray(self._electric_field, dtype=float)
            for name, charge in self._charges.items():
                markers = self._species[name]
                if charge != 0.0 and len(markers) > 0:
                    kick = charge * ELEMENTARY_CHARGE * dt / self._masses[name]
                    markers.velocities = markers.velocities + kick * field_vector
        diagnostics = self._collisions.collide_species(self._species, dt)
        for species_diagnostics in diagnostics.values():
            for process, count in species_diagnostics.counts.items():
                self._counts[process] = self._counts.get(process, 0) + count
        for name, limit in self._limits.items():
            if name in self._species and len(self._species[name]) > limit:
                merge_markers(self._species[name], limit // 2, rng=self._merge_rng)
        self._iteration += 1
        self._time += dt

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
        return ReactorHistory.from_states(states)

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
    def electric_field(self) -> tuple[float, float, float]:
        """Return the electric field in V/m."""
        return self._electric_field


def _mean_velocity(markers: ParticleArrays) -> np.ndarray:
    """Return the weighted mean velocity of ``markers`` (zero if empty) on the host."""
    total = markers.total_weight
    if total <= 0.0:
        return np.zeros(3)
    mean = xp.sum(markers.weights[:, None] * markers.velocities, axis=0) / total
    return np.asarray(xp.to_numpy(mean), dtype=float)
