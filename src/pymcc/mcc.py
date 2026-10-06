"""Monte Carlo collisions of charged particles with prescribed neutral backgrounds.

The operator implements the null-collision method (Vahedi and Surendra,
Comput. Phys. Commun. 87, 179 (1995)). Once per time step each marker of a
species collides with probability ``1 - exp(-nu_max dt)``, where ``nu_max``
bounds the species' total collision frequency. A second random number then
selects a real process ``k`` with probability ``nu_k / nu_max`` or a null
collision. A marker therefore collides at most once per step, which requires
``nu_max dt << 1``.

All kinematics are evaluated in the centre-of-mass frame of the marker and a
neutral partner whose velocity is sampled from the background Maxwellian:

``elastic``
    Isotropic scattering of the relative velocity.
``backscatter``
    Relative velocity reversed (charge exchange for symmetric ion-atom pairs:
    the ion leaves with the neutral's velocity).
``excitation``
    Isotropic scattering with the relative kinetic energy reduced by the
    threshold.
``ionization``
    ``e + A -> 2e + A+``. The residual energy ``E - E_iz`` is shared between
    the primary and the secondary electron (equally by default, or with a
    uniformly distributed fraction), both scattered isotropically in the
    neutral's frame. The new ion starts with the neutral's velocity.
``attachment``
    ``e + A -> A-``. The electron is removed and a negative ion is created
    with the neutral's velocity.
``charge_transfer``
    ``A+ + B -> A + B+``. The incident ion becomes a fast neutral and is
    removed (neutrals are not tracked). If an ``ion`` product species is
    given, the new ion ``B+`` is created with the neutral's velocity. Use
    ``backscatter`` instead for resonant charge exchange, where ion and
    neutral are the same species.
``detachment``
    ``A- + B -> A + B + e``. The negative ion is removed and an electron is
    emitted isotropically in the centre-of-mass frame carrying the residual
    energy ``E - E_th``. The neutral fragments are not tracked.

Products take the position and weight of the incident marker, so species that
exchange particles must share the same macroparticle weight to conserve charge.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import cunumpy as xp
import numpy as np

from pymcc._types import Array
from pymcc.background import NeutralBackground
from pymcc.constants import ELEMENTARY_CHARGE
from pymcc.markers import MarkerSet
from pymcc.process import CollisionProcess

__all__ = [
    "MCCDiagnostics",
    "MCCResult",
    "MonteCarloCollisions",
    "NewMarkers",
    "make_rng",
]


class NewMarkers(NamedTuple):
    """Markers created by collisions, for one product species."""

    positions: Array
    velocities: Array
    weights: Array


@dataclass
class MCCDiagnostics:
    """Collision counters for one species and one time step.

    Attributes:
        candidates: Markers that drew a (real or null) collision.
        null_collisions: Candidates that drew the null collision.
        counts: Real collisions per process name.
        bound_violations: Candidates whose real frequency exceeded ``nu_max``,
            which should be zero; otherwise the collision rate is too low.
    """

    candidates: int = 0
    null_collisions: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    bound_violations: int = 0

    @property
    def real_collisions(self) -> int:
        """Return the number of real (non-null) collisions."""
        return sum(self.counts.values())


@dataclass
class MCCResult:
    """Outcome of colliding one species' marker arrays.

    Attributes:
        velocities: The velocities after the step (the input array if in place).
        removed: Markers consumed by a collision (attachment, detachment,
            charge transfer).
        created: New markers per product species, one batch per process.
        diagnostics: The collision counters.
    """

    velocities: Array
    removed: Array
    created: dict[str, list[NewMarkers]]
    diagnostics: MCCDiagnostics


def make_rng(seed: int | None = None, rank: int = 0) -> Any:
    """Return a ``Generator`` of the active backend for the stream ``(seed, rank)``.

    Different ``rank`` values give independent streams from the same seed, so
    that every MPI rank draws its own random numbers. Without a seed the
    stream is seeded from the operating system.
    """
    sequence = np.random.SeedSequence(entropy=seed, spawn_key=(int(rank),))
    if xp.get_backend() == "cupy":
        return xp.rng.get_rng(int(sequence.generate_state(1, dtype=np.uint64)[0]))
    return np.random.default_rng(sequence)


def _max_squared_speed(velocities: Array) -> float:
    """Return the largest squared speed, without a temporary array on NumPy."""
    if isinstance(velocities, np.ndarray):
        return float(np.einsum("ij,ij->i", velocities, velocities).max())
    return float(xp.max(xp.sum(velocities**2, axis=1)))


def _isotropic_directions(rng: Any, num: int) -> Array:
    """Return ``num`` unit vectors distributed uniformly on the sphere."""
    cos_theta = 1.0 - 2.0 * rng.random(num)
    sin_theta = xp.sqrt(xp.clip(1.0 - cos_theta**2, 0.0, None))
    phi = 2.0 * xp.pi * rng.random(num)
    return xp.stack(
        (sin_theta * xp.cos(phi), sin_theta * xp.sin(phi), cos_theta),
        axis=1,
    )


class MonteCarloCollisions:
    """Null-collision Monte Carlo collision operator for several species.

    Args:
        species_masses: The mass in kg of every species that collides or is
            produced by a collision.
        backgrounds: The neutral gases.
        processes: The processes of each incident species.
        seed: Seed of the random stream; from the operating system if None.
        rank: Index of this stream, e.g. the MPI rank, so that every rank draws
            independent random numbers from the same ``seed``.
        rng: A NumPy or CuPy ``Generator`` to use instead of ``seed`` and ``rank``.
        num_bound_samples: Speeds of the logarithmic grid the frequency bound
            is tabulated on, in addition to the speeds of the table points.
        bound_safety: Factor (>= 1) raising ``nu_max`` to cover frequency
            maxima that fall between the speeds of the bound grid; 1.0 is
            exact for piecewise constant cross sections.

    Raises:
        KeyError: If a species, background or product has no definition.
    """

    def __init__(
        self,
        species_masses: Mapping[str, float],
        backgrounds: Sequence[NeutralBackground],
        processes: Mapping[str, Sequence[CollisionProcess]],
        seed: int | None = None,
        rank: int = 0,
        rng: Any = None,
        num_bound_samples: int = 4096,
        bound_safety: float = 1.02,
    ) -> None:
        """Check the definitions and tabulate the collision-frequency bounds."""
        self._masses = {name: float(mass) for name, mass in species_masses.items()}
        self._backgrounds = {bg.name: bg for bg in backgrounds}
        self._processes = {name: tuple(procs) for name, procs in processes.items()}
        for name, mass in self._masses.items():
            if mass <= 0.0:
                raise ValueError(f"Mass of species {name!r} must be > 0")
        for species, procs in self._processes.items():
            if species not in self._masses:
                raise KeyError(f"No mass given for colliding species {species!r}")
            names = [process.name for process in procs]
            if len(set(names)) != len(names):
                raise ValueError(f"Process names of {species!r} are not unique")
            for process in procs:
                if process.background not in self._backgrounds:
                    raise KeyError(
                        f"Process {process.name!r} uses unknown background "
                        f"{process.background!r}"
                    )
                for product in process.products.values():
                    if product not in self._masses:
                        raise KeyError(f"No mass given for product {product!r}")
        if bound_safety < 1.0:
            raise ValueError("bound_safety must be >= 1")
        self._rng = make_rng(seed, rank) if rng is None else rng
        self._num_bound_samples = int(num_bound_samples)
        self._bound_safety = float(bound_safety)
        self._frequency_tables = {
            species: self._running_frequency_bound(species)
            for species in self._processes
        }
        self._warned_bound = False

    # ------------------------------------------------------------------ #
    # Collision frequencies
    # ------------------------------------------------------------------ #
    def _reduced_mass(self, species: str, process: CollisionProcess) -> float:
        """Return the reduced mass of the incident and background particles."""
        mass = self._masses[species]
        neutral_mass = self._backgrounds[process.background].mass
        return mass * neutral_mass / (mass + neutral_mass)

    def _energy_mass(self, species: str, process: CollisionProcess) -> float:
        """Return the mass entering the cross-section lookup energy."""
        if process.energy_frame == "lab":
            return self._masses[species]
        return self._reduced_mass(species, process)

    def _speed_of(self, species: str, process: CollisionProcess, energy: Any) -> Any:
        """Return the relative speed (m/s) at lookup energy ``energy`` (eV)."""
        return np.sqrt(
            2.0 * energy * ELEMENTARY_CHARGE / self._energy_mass(species, process)
        )

    def _running_frequency_bound(self, species: str) -> tuple[np.ndarray, np.ndarray]:
        """Tabulate the running maximum of the total frequency over speed.

        The frequency is evaluated on a dense logarithmic speed grid merged
        with the speeds of every table point and threshold, which resolves the
        local maxima of a piecewise-linear cross section times the speed.
        Returns host arrays ``(speeds, running_max)``.
        """
        processes = self._processes[species]
        if not processes:
            return np.zeros(1), np.zeros(1)
        cap = max(
            float(self._speed_of(species, p, p.cross_section.max_energy))
            for p in processes
        )
        grids = [np.geomspace(cap * 1e-8, cap, self._num_bound_samples)]
        for process in processes:
            energies = np.asarray(xp.to_numpy(process.cross_section.energy))
            grids.append(self._speed_of(species, process, energies))
            if process.cross_section.threshold > 0.0:
                threshold_speed = self._speed_of(
                    species, process, process.cross_section.threshold
                )
                grids.append(np.asarray([threshold_speed * (1.0 + 1e-12)]))
        speeds = np.unique(np.concatenate(grids))
        frequencies = xp.to_numpy(self.total_frequency(species, xp.asarray(speeds)))
        return speeds, np.maximum.accumulate(frequencies)

    def process_frequency(
        self,
        species: str,
        process: CollisionProcess,
        relative_speed: Array,
        density_factor: Array | float | None = None,
    ) -> Array:
        """Return ``n sigma(E) g`` (1/s) for relative speeds ``g`` (m/s).

        ``density_factor`` scales the background's reference density, per
        marker or for all. Without it the largest density the background
        reaches is used, which is the upper bound the null-collision method
        needs.
        """
        energy = (
            0.5
            * self._energy_mass(species, process)
            * relative_speed**2
            / ELEMENTARY_CHARGE
        )
        background = self._backgrounds[process.background]
        if density_factor is None:
            density_factor = background.max_density_factor
        density = background.density * density_factor
        return density * process.cross_section(energy) * relative_speed

    def total_frequency(self, species: str, relative_speed: Array) -> Array:
        """Return the total collision frequency (1/s) at relative speeds ``g``."""
        total = xp.zeros_like(xp.asarray(relative_speed, dtype=float))
        for process in self._processes.get(species, ()):
            total = total + self.process_frequency(species, process, relative_speed)
        return total

    def frequency_bound(self, species: str, max_relative_speed: float) -> float:
        """Return ``nu_max`` (1/s) valid for relative speeds up to ``max_relative_speed``.

        The bound is the larger of the running maximum of the tabulated
        frequency over grid speeds up to ``max_relative_speed`` and the
        frequency at ``max_relative_speed`` itself, raised by ``bound_safety``
        for maxima between grid points. Beyond the last table point every
        cross section is constant and the frequency grows linearly with speed,
        which the direct evaluation at ``max_relative_speed`` covers.
        """
        if species not in self._frequency_tables:
            return 0.0
        speeds, running_max = self._frequency_tables[species]
        index = int(np.searchsorted(speeds, max_relative_speed, side="right")) - 1
        bound = float(running_max[index]) if index >= 0 else 0.0
        speed = xp.asarray([float(max_relative_speed)])
        bound = max(bound, float(self.total_frequency(species, speed)[0]))
        return self._bound_safety * bound

    # ------------------------------------------------------------------ #
    # Kinematics
    # ------------------------------------------------------------------ #
    def _collide_process(
        self,
        species: str,
        process: CollisionProcess,
        marker_indices: Array,
        positions: Array,
        velocities: Array,
        weights: Array,
        neutral_velocities: Array,
        removed: Array,
        created: dict[str, list[NewMarkers]],
        relative: Array,
        speed: Array,
    ) -> None:
        """Apply the kinematics of one process to the selected markers.

        ``relative`` (incident - neutral) and ``speed`` (its norm) are already
        known from candidate selection, so they are passed in rather than
        recomputed.
        """
        num = int(marker_indices.shape[0])
        mass = self._masses[species]
        neutral_mass = self._backgrounds[process.background].mass
        total_mass = mass + neutral_mass
        reduced_mass = mass * neutral_mass / total_mass
        velocity = velocities[marker_indices]
        center_of_mass = (mass * velocity + neutral_mass * neutral_velocities) / (
            total_mass
        )
        loss = process.loss * ELEMENTARY_CHARGE

        def add_product(role: str, product_velocities: Array) -> None:
            created.setdefault(process.products[role], []).append(
                NewMarkers(
                    xp.array(positions[marker_indices], copy=True),
                    product_velocities,
                    xp.array(weights[marker_indices], copy=True),
                )
            )

        if process.kind in ("elastic", "excitation"):
            speed_squared = speed**2 - 2.0 * loss / reduced_mass
            new_speed = xp.sqrt(xp.clip(speed_squared, 0.0, None))
            velocities[marker_indices] = center_of_mass + (
                neutral_mass / total_mass
            ) * new_speed[:, None] * _isotropic_directions(self._rng, num)
        elif process.kind == "backscatter":
            velocities[marker_indices] = (
                center_of_mass - (neutral_mass / total_mass) * relative
            )
        elif process.kind == "ionization":
            energy = 0.5 * mass * speed**2
            residual = xp.clip(energy - loss, 0.0, None)
            if process.energy_sharing == "equal":
                fraction = xp.full(num, 0.5)
            else:
                fraction = self._rng.random(num)
            primary_speed = xp.sqrt(2.0 * (1.0 - fraction) * residual / mass)
            secondary_speed = xp.sqrt(2.0 * fraction * residual / mass)
            velocities[marker_indices] = neutral_velocities + primary_speed[
                :, None
            ] * _isotropic_directions(self._rng, num)
            add_product(
                "electron",
                neutral_velocities
                + secondary_speed[:, None] * _isotropic_directions(self._rng, num),
            )
            add_product("ion", xp.array(neutral_velocities, copy=True))
        elif process.kind == "charge_transfer":
            removed[marker_indices] = True
            if "ion" in process.products:
                add_product("ion", xp.array(neutral_velocities, copy=True))
        elif process.kind == "attachment":
            removed[marker_indices] = True
            add_product("negative_ion", xp.array(neutral_velocities, copy=True))
        else:  # detachment
            removed[marker_indices] = True
            electron_mass = self._masses[process.products["electron"]]
            residual = xp.clip(0.5 * reduced_mass * speed**2 - loss, 0.0, None)
            electron_speed = xp.sqrt(2.0 * residual / electron_mass)
            add_product(
                "electron",
                center_of_mass
                + electron_speed[:, None] * _isotropic_directions(self._rng, num),
            )

    # ------------------------------------------------------------------ #
    # Collision step
    # ------------------------------------------------------------------ #
    def _select_candidates(
        self,
        num_markers: int,
        probability: float,
        alive: Array | None,
    ) -> Array:
        """Return the sorted indices of the markers that may collide this step.

        Each marker is a candidate independently with ``probability``. That is
        the same as drawing how many are candidates (binomial) and then which
        ones (a uniform subset), which costs O(candidates) instead of one
        random number per marker on NumPy. Other generators (CuPy) draw one
        number per marker.
        """
        if isinstance(self._rng, np.random.Generator):
            count = int(self._rng.binomial(num_markers, probability))
            candidates = self._rng.choice(num_markers, size=count, replace=False)
            candidates.sort()
            if alive is not None:
                candidates = candidates[np.asarray(alive, dtype=bool)[candidates]]
            return candidates
        candidate_mask = self._rng.random(num_markers) < probability
        if alive is not None:
            candidate_mask &= xp.asarray(alive, dtype=bool)
        return xp.where(candidate_mask)[0]

    def collide(
        self,
        species: str,
        positions: Array,
        velocities: Array,
        weights: Array,
        dt: float,
        alive: Array | None = None,
        in_place: bool = False,
    ) -> MCCResult:
        """Collide one species' markers for one time step ``dt`` (s).

        Args:
            species: The incident species.
            positions: Marker positions, shape ``(N, ndim)``; read by density
                profiles and copied to products.
            velocities: Marker velocities in m/s, shape ``(N, 3)``.
            weights: Physical particles per marker, shape ``(N,)``; copied to
                products.
            dt: The time step in s.
            alive: Only rows flagged here can collide (all rows if None).
            in_place: Update the collided rows of ``velocities`` directly,
                which avoids copying the marker storage; otherwise the result
                holds a new array.

        Returns:
            The new velocities, the removed markers, the created markers per
            product species and the collision counters.

        Raises:
            ValueError: If ``velocities`` does not have three components.
        """
        if not in_place:
            velocities = xp.array(velocities, dtype=float, copy=True)
        num_markers = int(velocities.shape[0])
        removed = xp.zeros(num_markers, dtype=bool)
        created: dict[str, list[NewMarkers]] = {}
        diagnostics = MCCDiagnostics()
        processes = self._processes.get(species, ())
        if num_markers == 0 or dt <= 0.0 or not processes:
            return MCCResult(velocities, removed, created, diagnostics)
        if velocities.ndim != 2 or velocities.shape[1] != 3:
            raise ValueError(
                "Monte Carlo collisions need three velocity components per marker, "
                f"got shape {tuple(velocities.shape)}"
            )

        # Dead rows only raise the bound, which keeps it valid.
        max_speed = float(np.sqrt(_max_squared_speed(velocities)))
        max_thermal = max(
            self._backgrounds[p.background].thermal_speed for p in processes
        )
        nu_max = self.frequency_bound(species, max_speed + 8.0 * max_thermal)
        if nu_max <= 0.0:
            return MCCResult(velocities, removed, created, diagnostics)

        probability = 1.0 - float(np.exp(-nu_max * dt))
        candidates = self._select_candidates(num_markers, probability, alive)
        num_candidates = int(candidates.shape[0])
        diagnostics.candidates = num_candidates
        if num_candidates == 0:
            return MCCResult(velocities, removed, created, diagnostics)
        candidates = xp.asarray(candidates)

        incident = velocities[candidates]

        # One relative velocity and speed per process, batched into a single
        # (num_processes, num_candidates, ...) pair of array operations rather
        # than a Python loop of small ones. Only the cross-section lookup (a
        # different table per process) runs once per process.
        backgrounds_used = list(
            dict.fromkeys(process.background for process in processes)
        )
        background_index = {name: index for index, name in enumerate(backgrounds_used)}
        neutral_stack = xp.stack(
            [
                self._backgrounds[name].sample_velocities(self._rng, num_candidates)
                for name in backgrounds_used
            ],
        )
        process_background = [
            background_index[process.background] for process in processes
        ]
        # Thinning: the bound uses each background's largest density, and a
        # marker's real frequency the density where it is.
        density_factors = [
            self._backgrounds[name].density_factor(positions[candidates])
            for name in backgrounds_used
        ]
        relative_stack = incident[None, :, :] - neutral_stack[process_background]
        speed_stack = xp.sqrt(xp.sum(relative_stack**2, axis=-1))

        cumulative = xp.zeros(num_candidates)
        selection = self._rng.random(num_candidates) * nu_max
        chosen = xp.full(num_candidates, -1, dtype=int)
        for index, process in enumerate(processes):
            cumulative = cumulative + self.process_frequency(
                species,
                process,
                speed_stack[index],
                density_factors[process_background[index]],
            )
            chosen = xp.where((chosen < 0) & (selection < cumulative), index, chosen)
        violations = int(xp.count_nonzero(cumulative > nu_max * (1.0 + 1e-9)))
        diagnostics.bound_violations = violations
        if violations and not self._warned_bound:
            warnings.warn(
                f"{violations} {species} marker(s) exceeded the null-collision "
                "frequency bound; collision rates are underestimated.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_bound = True

        diagnostics.null_collisions = int(xp.count_nonzero(chosen < 0))
        for index, process in enumerate(processes):
            local = xp.where(chosen == index)[0]
            diagnostics.counts[process.name] = int(local.shape[0])
            if local.shape[0] == 0:
                continue
            self._collide_process(
                species=species,
                process=process,
                marker_indices=candidates[local],
                positions=positions,
                velocities=velocities,
                weights=weights,
                neutral_velocities=neutral_stack[process_background[index]][local],
                removed=removed,
                created=created,
                relative=relative_stack[index][local],
                speed=speed_stack[index][local],
            )
        return MCCResult(velocities, removed, created, diagnostics)

    def collide_species(
        self,
        species: Mapping[str, MarkerSet],
        dt: float,
    ) -> dict[str, MCCDiagnostics]:
        """Collide every species in ``species`` for one time step ``dt`` (s), in place.

        Velocities are updated in place, consumed markers removed and new ones
        added to their product species. New markers are added only after every
        species has collided, so products do not collide in the step that
        created them.

        Args:
            species: Marker sets by species name, e.g.
                :class:`~pymcc.markers.ParticleArrays`. Species without
                processes are only receivers of products.
            dt: The time step in s.

        Returns:
            The collision counters of every species that has processes.

        Raises:
            KeyError: If a product species is missing from ``species``.
        """
        diagnostics: dict[str, MCCDiagnostics] = {}
        pending: dict[str, list[NewMarkers]] = {}
        for name in self._processes:
            if name not in species:
                continue
            markers = species[name]
            result = self.collide(
                species=name,
                positions=markers.positions,
                velocities=markers.velocities,
                weights=markers.weights,
                dt=dt,
                alive=getattr(markers, "alive", None),
                in_place=True,
            )
            if bool(xp.any(result.removed)):
                markers.remove(result.removed)
            for product, batches in result.created.items():
                pending.setdefault(product, []).extend(batches)
            diagnostics[name] = result.diagnostics

        for product, batches in pending.items():
            if product not in species:
                raise KeyError(f"Product species {product!r} is not simulated")
            species[product].add(
                positions=xp.concatenate([batch.positions for batch in batches]),
                velocities=xp.concatenate([batch.velocities for batch in batches]),
                weights=xp.concatenate([batch.weights for batch in batches]),
            )
        return diagnostics

    @property
    def processes(self) -> dict[str, tuple[CollisionProcess, ...]]:
        """Return the processes per incident species."""
        return dict(self._processes)

    @property
    def backgrounds(self) -> dict[str, NeutralBackground]:
        """Return the neutral backgrounds by name."""
        return dict(self._backgrounds)

    @property
    def species_masses(self) -> dict[str, float]:
        """Return the mass in kg of every species, by name."""
        return dict(self._masses)

    @property
    def rng(self) -> Any:
        """Return the random generator of the operator."""
        return self._rng
