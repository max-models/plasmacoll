"""A zero-dimensional (spatially homogeneous) reactor advanced by collisions alone.

:class:`ZeroDReactor` evolves the velocity distributions and densities of a
set of species with :class:`~plasmacoll.mcc.MonteCarloCollisions` and nothing else:
there is no field, no push and no boundary. It is the tool for testing a
cross-section set or a reaction chain (rate coefficients, thresholds,
ionization against attachment, relaxation to the gas temperature) without the
cost and the confounding factors of a spatial simulation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from plasmacoll.markers import ParticleArrays
from plasmacoll.mcc import MonteCarloCollisions

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
    """

    time: float
    density: dict[str, float] = field(default_factory=dict)
    mean_energy_ev: dict[str, float] = field(default_factory=dict)
    temperature: dict[str, float] = field(default_factory=dict)
    num_markers: dict[str, int] = field(default_factory=dict)
    collisions: dict[str, int] = field(default_factory=dict)


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
    """

    time: np.ndarray
    density: dict[str, np.ndarray]
    mean_energy_ev: dict[str, np.ndarray]
    temperature: dict[str, np.ndarray]
    num_markers: dict[str, np.ndarray]
    collisions: dict[str, np.ndarray]

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
        )


class ZeroDReactor:
    """A homogeneous reactor of volume ``volume`` advanced only by collisions.

    Args:
        collisions: The collision operator; it knows the mass of every species.
        species: The initial markers per species. Species of the operator that
            are missing here start empty, so products need not be listed.
        volume: Reactor volume in m^3; densities are total weights over volume.

    Raises:
        KeyError: If a species has no mass in the operator.
    """

    def __init__(
        self,
        collisions: MonteCarloCollisions,
        species: Mapping[str, ParticleArrays],
        volume: float = 1.0,
    ) -> None:
        """Check the species and start every missing one without markers."""
        if volume <= 0.0:
            raise ValueError(f"volume must be > 0, got {volume}")
        masses = collisions.species_masses
        unknown = set(species) - set(masses)
        if unknown:
            raise KeyError(f"No mass given for species {sorted(unknown)}")
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
        )

    def step(self, dt: float) -> ReactorState:
        """Collide every species for ``dt`` (s) and return the new state."""
        diagnostics = self._collisions.collide_species(self._species, dt)
        for species_diagnostics in diagnostics.values():
            for process, count in species_diagnostics.counts.items():
                self._counts[process] = self._counts.get(process, 0) + count
        self._iteration += 1
        self._time += dt
        return self.state()

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
            state = self.step(dt)
            if index % every == 0:
                states.append(state)
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
