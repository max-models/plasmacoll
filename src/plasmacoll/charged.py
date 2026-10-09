"""Collisions between charged particles: Coulomb scattering and charged-charged reactions.

:class:`ChargedCollisions` pairs the markers of charged species within spatial
cells and, per pair,

* scatters them with the binary Coulomb collision model of Takizuka and Abe,
  J. Comput. Phys. 25, 205 (1977): the relative velocity is rotated by an
  angle ``chi`` with ``tan(chi/2)`` drawn from a normal distribution of
  variance ``(q_a q_b)^2 n lnL dt / (8 pi eps0^2 mu^2 u^3)``;
* lets them react (:class:`ChargedReaction`): mutual neutralization
  ``A- + B+ -> A + B``, radiative recombination ``e + A+ -> A`` and electron
  detachment ``e + A- -> A + 2e``, with a rate coefficient or a cross section.

Pairing follows Takizuka and Abe. Within one species the markers of a cell are
shuffled and paired; an odd count leaves a triple that collides as three pairs
of half the time step. Between two species every marker of the more numerous
one in a cell is paired with a marker of the other, which is reused
cyclically, and the scattering of each side is accepted with the probability
that gives it the right rate (Nanbu and Yonemura, J. Comput. Phys. 145, 639
(1998)). The kicks a reused marker receives are summed. Scattering conserves
momentum exactly and energy exactly for markers that are not reused, and on
average otherwise. Rates are exact in expectation for markers of equal weight
within a species and cell.

A pair reacts with probability ``K n_a n_b V dt / S``, where ``K = sigma g``
(or the rate coefficient), ``V`` the cell volume and ``S`` the sum over the
cell's pairs of the smaller weight of each pair, which gives the expected
number of physical reactions ``K n_a n_b V dt`` for any weights. A reaction
removes the smaller of the two weights from both markers (removing a marker
whose weight reaches zero) and creates every outgoing particle as a new marker
of that weight, so particle number and charge are conserved exactly. Each
marker reacts at most once per step, which requires small probabilities.

The pairing runs on the host (NumPy); the kinematics on the active backend.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import cunumpy as xp
import numpy as np

from plasmacoll._types import Array
from plasmacoll.constants import ELEMENTARY_CHARGE
from plasmacoll.cross_sections import CrossSection
from plasmacoll.markers import MarkerSet
from plasmacoll.mcc import NewMarkers, _isotropic_directions, _rotate, _unit, make_rng

__all__ = [
    "CHARGED_REACTION_KINDS",
    "ChargedCollisions",
    "ChargedDiagnostics",
    "ChargedReaction",
    "coulomb_logarithm",
]

VACUUM_PERMITTIVITY = 8.8541878128e-12  # F/m

#: The reactions :class:`ChargedCollisions` implements, with the product roles
#: each needs and may have.
CHARGED_REACTION_KINDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "mutual_neutralization": ((), ("neutral_1", "neutral_2")),
    "recombination": ((), ("neutral",)),
    "electron_detachment": (("electron",), ("neutral",)),
}


def coulomb_logarithm(
    electron_density: float, electron_temperature_ev: float, ion_charge: float = 1.0
) -> float:
    """Return the electron-ion Coulomb logarithm of the NRL Plasma Formulary.

    ``23 - ln(n^(1/2) Z T^(-3/2))`` for ``T < 10 Z^2`` eV and
    ``24 - ln(n^(1/2) / T)`` above, with ``n`` in cm^-3 and ``T`` in eV.

    Args:
        electron_density: Electron density in m^-3.
        electron_temperature_ev: Electron temperature in eV.
        ion_charge: Ion charge number ``Z``.
    """
    if electron_density <= 0.0 or electron_temperature_ev <= 0.0:
        raise ValueError("density and temperature must be > 0")
    density_cgs = electron_density * 1.0e-6
    if electron_temperature_ev < 10.0 * ion_charge**2:
        return 23.0 - math.log(
            math.sqrt(density_cgs) * ion_charge * electron_temperature_ev**-1.5
        )
    return 24.0 - math.log(math.sqrt(density_cgs) / electron_temperature_ev)


@dataclass(frozen=True)
class ChargedReaction:
    """A reaction between markers of two charged species.

    ``species`` names the two reactants in the order of the reaction:

    ``mutual_neutralization``
        ``(negative ion, positive ion)``. Optional products ``neutral_1`` and
        ``neutral_2`` (the neutralized first and second reactant) keep their
        parent's velocity.
    ``recombination``
        ``(electron, positive ion)``, radiative: the energy is radiated. An
        optional ``neutral`` product moves with the pair's centre of mass.
    ``electron_detachment``
        ``(electron, negative ion)``. The two outgoing electrons (product
        ``electron``) share the centre-of-mass energy above ``energy_loss``
        equally, emitted isotropically about the centre of mass; an optional
        ``neutral`` moves with it.

    Give either ``rate_coefficient`` (m^3/s, constant) or ``cross_section``
    against the centre-of-mass energy in eV.

    Attributes:
        kind: One of :data:`CHARGED_REACTION_KINDS`.
        species: The two reactant species.
        rate_coefficient: ``K`` in m^3/s.
        cross_section: ``sigma(E)``; then ``K = sigma g`` per pair.
        products: Product species by role.
        energy_loss: Energy in eV lost in electron detachment (the electron
            affinity); the cross section's threshold if None.
        name: A unique name for diagnostics; ``"<kind>:<a>:<b>"`` if empty.
    """

    kind: str
    species: tuple[str, str]
    rate_coefficient: float | None = None
    cross_section: CrossSection | None = None
    products: Mapping[str, str] = field(default_factory=dict)
    energy_loss: float | None = None
    name: str = ""

    def __post_init__(self) -> None:
        """Validate the reaction."""
        if self.kind not in CHARGED_REACTION_KINDS:
            raise ValueError(
                f"Unknown charged reaction {self.kind!r}; expected one of "
                f"{tuple(CHARGED_REACTION_KINDS)}"
            )
        species = tuple(self.species)
        if len(species) != 2 or species[0] == species[1]:
            raise ValueError("a charged reaction needs two different species")
        if (self.rate_coefficient is None) == (self.cross_section is None):
            raise ValueError("give exactly one of rate_coefficient and cross_section")
        if self.rate_coefficient is not None and self.rate_coefficient < 0.0:
            raise ValueError("rate_coefficient must be >= 0")
        required, optional = CHARGED_REACTION_KINDS[self.kind]
        missing = set(required) - set(self.products)
        if missing:
            raise ValueError(f"{self.kind} needs product species for {sorted(missing)}")
        unknown = set(self.products) - set(required) - set(optional)
        if unknown:
            raise ValueError(f"{self.kind} has no product role(s) {sorted(unknown)}")
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "products", dict(self.products))
        if not self.name:
            object.__setattr__(self, "name", f"{self.kind}:{species[0]}:{species[1]}")

    @property
    def loss(self) -> float:
        """Return the energy lost by electron detachment in eV (zero otherwise)."""
        if self.kind != "electron_detachment":
            return 0.0
        if self.energy_loss is not None:
            return self.energy_loss
        return self.cross_section.threshold if self.cross_section is not None else 0.0


@dataclass
class ChargedDiagnostics:
    """Counters of one step of :class:`ChargedCollisions`.

    Attributes:
        coulomb_pairs: Pairs scattered per species pair ``"a:b"``.
        reactions: Reactions per reaction name (pairs that reacted).
        weighted_reactions: Physical reactions per reaction name.
        max_reaction_probability: The largest reaction probability of a pair.
    """

    coulomb_pairs: dict[str, int] = field(default_factory=dict)
    reactions: dict[str, int] = field(default_factory=dict)
    weighted_reactions: dict[str, float] = field(default_factory=dict)
    max_reaction_probability: float = 0.0


@dataclass
class _Pairs:
    """Pairs of markers of species ``a`` and ``b`` (host index arrays)."""

    first: np.ndarray  # marker index in species a
    second: np.ndarray  # marker index in species b
    cell: np.ndarray  # cell of the pair
    accept_first: np.ndarray  # probability that a's scattering is applied
    accept_second: np.ndarray
    density: np.ndarray  # the density n entering the scattering variance
    time_fraction: np.ndarray  # fraction of dt (1/2 for the pairs of a triple)
    round: np.ndarray  # pairs of one round share no marker


class ChargedCollisions:
    """Coulomb collisions and reactions between the markers of charged species.

    Args:
        species_masses: The mass in kg of every species involved, products included.
        charges: The charge of every charged species in units of ``e``.
        coulomb_pairs: The species pairs that scatter, e.g.
            ``[("e", "e"), ("e", "Ar+")]``; ``"all"`` for every pair of charged
            species (each unordered pair once, like pairs included).
        reactions: The charged-charged reactions.
        coulomb_logarithm: ``ln Lambda``, the same for every pair; see
            :func:`coulomb_logarithm`.
        seed: Seed of the random streams.
        rank: Index of this stream (e.g. the MPI rank).
        max_reaction_probability: Warn (once) when a pair's reaction
            probability exceeds this.

    Raises:
        KeyError: If a species has no mass or no charge.
    """

    def __init__(
        self,
        species_masses: Mapping[str, float],
        charges: Mapping[str, float],
        coulomb_pairs: Iterable[tuple[str, str]] | str = (),
        reactions: Sequence[ChargedReaction] = (),
        coulomb_logarithm: float = 10.0,
        seed: int | None = None,
        rank: int = 0,
        max_reaction_probability: float = 0.1,
    ) -> None:
        """Check the definitions."""
        self._masses = {name: float(mass) for name, mass in species_masses.items()}
        self._charges = {name: float(charge) for name, charge in charges.items()}
        if coulomb_pairs == "all":
            charged = sorted(name for name, q in self._charges.items() if q != 0.0)
            pairs = [(a, b) for index, a in enumerate(charged) for b in charged[index:]]
        elif isinstance(coulomb_pairs, str):
            raise ValueError('coulomb_pairs must be pairs of species or "all"')
        else:
            pairs = [tuple(pair) for pair in coulomb_pairs]
        self._coulomb_pairs: list[tuple[str, str]] = []
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError(f"Coulomb pair {pair} must have two species")
            for name in pair:
                self._check_species(name)
                if self._charges.get(name, 0.0) == 0.0:
                    raise ValueError(f"Coulomb pair species {name!r} has no charge")
            self._coulomb_pairs.append((pair[0], pair[1]))
        self._reactions = tuple(reactions)
        names = [reaction.name for reaction in self._reactions]
        if len(set(names)) != len(names):
            raise ValueError("charged reaction names must be unique")
        for reaction in self._reactions:
            for name in (*reaction.species, *reaction.products.values()):
                if name not in self._masses:
                    raise KeyError(f"No mass given for species {name!r}")
        if coulomb_logarithm <= 0.0:
            raise ValueError("coulomb_logarithm must be > 0")
        self._coulomb_logarithm = float(coulomb_logarithm)
        self._rng = make_rng(seed, rank)
        self._host_rng = np.random.default_rng(
            np.random.SeedSequence(entropy=seed, spawn_key=(int(rank), 1))
        )
        self._max_probability = float(max_reaction_probability)
        self._warned_probability = False

    def _check_species(self, name: str) -> None:
        if name not in self._masses:
            raise KeyError(f"No mass given for species {name!r}")
        if name not in self._charges:
            raise KeyError(f"No charge given for species {name!r}")

    @property
    def charges(self) -> dict[str, float]:
        """Return the charge of every species in units of ``e``."""
        return dict(self._charges)

    @property
    def coulomb_pairs(self) -> list[tuple[str, str]]:
        """Return the species pairs that scatter."""
        return list(self._coulomb_pairs)

    @property
    def reactions(self) -> tuple[ChargedReaction, ...]:
        """Return the reactions."""
        return self._reactions

    # ------------------------------------------------------------------ #
    # Cells and pairs
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cells(
        species: Mapping[str, MarkerSet],
        names: Iterable[str],
        cell_size: float | Sequence[float] | None,
    ) -> tuple[dict[str, np.ndarray], int]:
        """Return the cell of every marker of ``names`` and the number of cells."""
        names = [name for name in dict.fromkeys(names) if name in species]
        if cell_size is None:
            return {
                name: np.zeros(int(species[name].weights.shape[0]), dtype=np.int64)
                for name in names
            }, 1
        keys = []
        for name in names:
            positions = np.asarray(xp.to_numpy(species[name].positions), dtype=float)
            size = np.broadcast_to(
                np.asarray(cell_size, dtype=float), positions.shape[1:]
            )
            if np.any(size <= 0.0):
                raise ValueError("cell_size must be > 0")
            keys.append(np.floor(positions / size).astype(np.int64))
        everything = np.concatenate(keys) if keys else np.zeros((0, 1), dtype=np.int64)
        _, inverse = np.unique(everything, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        cells: dict[str, np.ndarray] = {}
        start = 0
        for name, key in zip(names, keys, strict=True):
            cells[name] = inverse[start : start + key.shape[0]]
            start += key.shape[0]
        return cells, int(np.max(inverse)) + 1 if inverse.size else 1

    def _shuffled(self, cell: np.ndarray, num_cells: int) -> tuple[np.ndarray, ...]:
        """Shuffle markers within their cells.

        Returns:
            The order, the cell of each sorted marker, its rank within its
            cell, and the count and start of each cell.
        """
        order = np.lexsort((self._host_rng.random(cell.shape[0]), cell))
        sorted_cell = cell[order]
        count = np.bincount(cell, minlength=num_cells)
        start = np.cumsum(count) - count
        rank = np.arange(cell.shape[0]) - start[sorted_cell]
        return order, sorted_cell, rank, count, start

    def _like_pairs(
        self, cell: np.ndarray, weights: np.ndarray, num_cells: int, volume: float
    ) -> _Pairs:
        """Pair the markers of one species within each cell (Takizuka-Abe)."""
        order, sorted_cell, rank, count, start = self._shuffled(cell, num_cells)
        density = np.bincount(cell, weights=weights, minlength=num_cells) / volume
        n = count[sorted_cell]
        odd = n % 2 == 1
        # Ordinary pairs (r, r+1) for even r, below the triple of an odd cell.
        limit = np.where(odd, n - 3, n)
        first_rank = (rank % 2 == 0) & (rank + 1 < limit)
        firsts = [order[first_rank]]
        seconds = [order[np.flatnonzero(first_rank) + 1]]
        cells = [sorted_cell[first_rank]]
        fraction = [np.ones(int(first_rank.sum()))]
        rounds = [np.zeros(int(first_rank.sum()), dtype=np.int64)]
        # The triple (N-3, N-2, N-1) of an odd cell: three pairs of half the
        # time step, in three rounds so that no round reuses a marker.
        head = odd & (n >= 3) & (rank == n - 3)
        positions = np.flatnonzero(head)
        for index, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
            firsts.append(order[positions + i])
            seconds.append(order[positions + j])
            cells.append(sorted_cell[positions])
            fraction.append(np.full(positions.shape[0], 0.5))
            rounds.append(np.full(positions.shape[0], index, dtype=np.int64))
        cell_of_pair = np.concatenate(cells)
        num_pairs = cell_of_pair.shape[0]
        return _Pairs(
            first=np.concatenate(firsts),
            second=np.concatenate(seconds),
            cell=cell_of_pair,
            accept_first=np.ones(num_pairs),
            accept_second=np.ones(num_pairs),
            density=density[cell_of_pair],
            time_fraction=np.concatenate(fraction),
            round=np.concatenate(rounds),
        )

    def _unlike_pairs(
        self,
        cell_a: np.ndarray,
        cell_b: np.ndarray,
        weights_a: np.ndarray,
        weights_b: np.ndarray,
        num_cells: int,
        volume: float,
    ) -> _Pairs:
        """Pair the markers of two species within each cell.

        Every marker of the species with more markers in a cell gets a partner
        of the other, which is reused cyclically. A marker in ``k`` pairs must
        see the density ``n_partner / k`` per pair; the variance uses the larger
        of the two sides' values and each side is accepted with its share.
        """
        order_a, sorted_a, rank_a, count_a, start_a = self._shuffled(cell_a, num_cells)
        order_b, sorted_b, rank_b, count_b, start_b = self._shuffled(cell_b, num_cells)
        density_a = np.bincount(cell_a, weights=weights_a, minlength=num_cells) / volume
        density_b = np.bincount(cell_b, weights=weights_b, minlength=num_cells) / volume
        na, nb = count_a[sorted_a], count_b[sorted_a]
        a_leads = (nb > 0) & (na >= nb)
        first_lead = order_a[a_leads]
        cells_lead = sorted_a[a_leads]
        second_lead = order_b[
            start_b[cells_lead] + rank_a[a_leads] % count_b[cells_lead]
        ]
        na_b, nb_b = count_a[sorted_b], count_b[sorted_b]
        b_leads = (na_b > 0) & (nb_b > na_b)
        second_follow = order_b[b_leads]
        cells_follow = sorted_b[b_leads]
        first_follow = order_a[
            start_a[cells_follow] + rank_b[b_leads] % count_a[cells_follow]
        ]

        first = np.concatenate([first_lead, first_follow])
        second = np.concatenate([second_lead, second_follow])
        cell = np.concatenate([cells_lead, cells_follow])
        # Pairs per marker of each side in the pair's cell.
        pairs_in_cell = np.maximum(count_a, count_b)[cell]
        per_a = pairs_in_cell / count_a[cell]
        per_b = pairs_in_cell / count_b[cell]
        share_a = density_b[cell] / per_a
        share_b = density_a[cell] / per_b
        density = np.maximum(share_a, share_b)
        return _Pairs(
            first=first,
            second=second,
            cell=cell,
            accept_first=share_a / density,
            accept_second=share_b / density,
            density=density,
            time_fraction=np.ones(first.shape[0]),
            round=np.zeros(first.shape[0], dtype=np.int64),
        )

    def _pairs(
        self,
        species: Mapping[str, MarkerSet],
        a: str,
        b: str,
        cells: Mapping[str, np.ndarray],
        num_cells: int,
        volume: float,
    ) -> _Pairs | None:
        """Return the pairs of species ``a`` and ``b`` (``a == b`` for like pairs)."""
        if a not in species or b not in species:
            return None
        weights_a = np.asarray(xp.to_numpy(species[a].weights), dtype=float)
        if a == b:
            return self._like_pairs(cells[a], weights_a, num_cells, volume)
        weights_b = np.asarray(xp.to_numpy(species[b].weights), dtype=float)
        return self._unlike_pairs(
            cells[a], cells[b], weights_a, weights_b, num_cells, volume
        )

    # ------------------------------------------------------------------ #
    # Coulomb scattering
    # ------------------------------------------------------------------ #
    def _scatter(
        self, species: Mapping[str, MarkerSet], a: str, b: str, pairs: _Pairs, dt: float
    ) -> None:
        """Apply Takizuka-Abe scattering to every pair, round by round."""
        mass_a, mass_b = self._masses[a], self._masses[b]
        reduced = mass_a * mass_b / (mass_a + mass_b)
        charge_product = self._charges[a] * self._charges[b] * ELEMENTARY_CHARGE**2
        factor = (
            charge_product**2
            * self._coulomb_logarithm
            * dt
            / (8.0 * math.pi * VACUUM_PERMITTIVITY**2 * reduced**2)
        )
        velocities_a = species[a].velocities
        velocities_b = species[b].velocities
        for round_index in range(int(pairs.round.max()) + 1 if pairs.round.size else 0):
            # Round 0 always has pairs; rounds 1 and 2 hold the triples' others.
            selected = np.flatnonzero(pairs.round == round_index)
            first = xp.asarray(pairs.first[selected])
            second = xp.asarray(pairs.second[selected])
            num = int(selected.size)
            relative = velocities_a[first] - velocities_b[second]
            speed = xp.sqrt(xp.sum(relative**2, axis=1))
            safe = xp.where(speed > 0.0, speed, 1.0)
            variance = (
                factor
                * xp.asarray(pairs.density[selected] * pairs.time_fraction[selected])
                / safe**3
            )
            delta = xp.sqrt(variance) * self._rng.standard_normal(num)
            cos_chi = (1.0 - delta**2) / (1.0 + delta**2)
            phi = 2.0 * xp.pi * self._rng.random(num)
            rotated = speed[:, None] * _rotate(_unit(relative, speed), cos_chi, phi)
            change = xp.where((speed > 0.0)[:, None], rotated - relative, 0.0)
            accept_a = self._rng.random(num) < xp.asarray(pairs.accept_first[selected])
            accept_b = self._rng.random(num) < xp.asarray(pairs.accept_second[selected])
            kick_a = (reduced / mass_a) * change * accept_a[:, None]
            kick_b = -(reduced / mass_b) * change * accept_b[:, None]
            # Reused markers receive the sum of their kicks.
            if a == b:
                total = _scatter_sum(
                    first, kick_a, velocities_a.shape[0]
                ) + _scatter_sum(second, kick_b, velocities_a.shape[0])
                velocities_a += total
            else:
                velocities_a += _scatter_sum(first, kick_a, velocities_a.shape[0])
                velocities_b += _scatter_sum(second, kick_b, velocities_b.shape[0])

    # ------------------------------------------------------------------ #
    # Reactions
    # ------------------------------------------------------------------ #
    def _react(
        self,
        species: Mapping[str, MarkerSet],
        reaction: ChargedReaction,
        pairs: _Pairs,
        dt: float,
        volume: float,
        cells: Mapping[str, np.ndarray],
        num_cells: int,
        removed_weight: dict[str, Any],
        created: dict[str, list[NewMarkers]],
        diagnostics: ChargedDiagnostics,
    ) -> None:
        """Draw which pairs react and create their products."""
        a, b = reaction.species
        markers_a, markers_b = species[a], species[b]
        weights_a = np.asarray(xp.to_numpy(markers_a.weights), dtype=float)
        weights_b = np.asarray(xp.to_numpy(markers_b.weights), dtype=float)
        # Weight still available after earlier reactions of this step.
        weights_a = weights_a - np.asarray(xp.to_numpy(removed_weight[a]))
        weights_b = weights_b - np.asarray(xp.to_numpy(removed_weight[b]))
        min_weight = np.minimum(weights_a[pairs.first], weights_b[pairs.second])
        pair_sum = np.bincount(pairs.cell, weights=min_weight, minlength=num_cells)
        density_a = (
            np.bincount(cells[a], weights=weights_a, minlength=num_cells) / volume
        )
        density_b = (
            np.bincount(cells[b], weights=weights_b, minlength=num_cells) / volume
        )

        mass_a, mass_b = self._masses[a], self._masses[b]
        first = xp.asarray(pairs.first)
        second = xp.asarray(pairs.second)
        velocity_a = markers_a.velocities[first]
        velocity_b = markers_b.velocities[second]
        relative = velocity_a - velocity_b
        speed = xp.sqrt(xp.sum(relative**2, axis=1))
        if reaction.cross_section is not None:
            reduced = mass_a * mass_b / (mass_a + mass_b)
            energy = 0.5 * reduced * speed**2 / ELEMENTARY_CHARGE
            rate = np.asarray(xp.to_numpy(reaction.cross_section(energy) * speed))
        else:
            rate = np.full(
                pairs.first.shape[0], float(reaction.rate_coefficient or 0.0)
            )
        safe_sum = np.where(pair_sum > 0.0, pair_sum, 1.0)
        probability = (
            rate
            * (density_a * density_b * volume * dt / safe_sum)[pairs.cell]
            * (min_weight > 0.0)
        )
        if probability.size:
            diagnostics.max_reaction_probability = max(
                diagnostics.max_reaction_probability, float(probability.max())
            )
        reacts = self._host_rng.random(probability.shape[0]) < probability
        # Each marker reacts at most once: keep a reused marker's first pair
        # (pairs are in random order within a cell).
        candidates = np.flatnonzero(reacts)
        _, unique_first = np.unique(pairs.first[candidates], return_index=True)
        candidates = candidates[np.sort(unique_first)]
        _, unique_second = np.unique(pairs.second[candidates], return_index=True)
        candidates = candidates[np.sort(unique_second)]
        diagnostics.reactions[reaction.name] = int(candidates.shape[0])
        diagnostics.weighted_reactions[reaction.name] = float(
            min_weight[candidates].sum()
        )
        if candidates.size == 0:
            return

        index_a = pairs.first[candidates]
        index_b = pairs.second[candidates]
        weight = min_weight[candidates]
        removed_a = np.asarray(xp.to_numpy(removed_weight[a]), dtype=float).copy()
        removed_b = np.asarray(xp.to_numpy(removed_weight[b]), dtype=float).copy()
        removed_a[index_a] += weight
        removed_b[index_b] += weight
        removed_weight[a] = removed_a
        removed_weight[b] = removed_b

        selected = xp.asarray(candidates)
        va, vb = velocity_a[selected], velocity_b[selected]
        center = (mass_a * va + mass_b * vb) / (mass_a + mass_b)
        weight_device = xp.asarray(weight)
        position_a = xp.array(markers_a.positions[xp.asarray(index_a)], copy=True)
        position_b = xp.array(markers_b.positions[xp.asarray(index_b)], copy=True)

        def add(role: str, positions: Array, velocities: Array) -> None:
            if role in reaction.products:
                created.setdefault(reaction.products[role], []).append(
                    NewMarkers(positions, velocities, weight_device)
                )

        if reaction.kind == "mutual_neutralization":
            add("neutral_1", position_a, xp.array(va, copy=True))
            add("neutral_2", position_b, xp.array(vb, copy=True))
        elif reaction.kind == "recombination":
            add("neutral", position_b, center)
        else:  # electron detachment
            reduced = mass_a * mass_b / (mass_a + mass_b)
            relative_speed = speed[selected]
            residual = xp.clip(
                0.5 * reduced * relative_speed**2 - reaction.loss * ELEMENTARY_CHARGE,
                0.0,
                None,
            )
            electron_mass = self._masses[reaction.products["electron"]]
            electron_speed = xp.sqrt(residual / electron_mass)  # half the energy each
            num = int(candidates.shape[0])
            for _ in range(2):
                add(
                    "electron",
                    position_a,
                    center
                    + electron_speed[:, None] * _isotropic_directions(self._rng, num),
                )
            add("neutral", position_b, center)

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def collide_species(
        self,
        species: Mapping[str, MarkerSet],
        dt: float,
        cell_volume: float,
        cell_size: float | Sequence[float] | None = None,
    ) -> ChargedDiagnostics:
        """Scatter and react the markers of ``species`` for ``dt`` (s), in place.

        Velocities are updated in place; markers whose weight reaction uses up
        are removed, partly used ones have their weight reduced in place (so
        the container's ``weights`` must be a writable view), and products are
        added to their species after all reactions.

        Args:
            species: Marker sets by species name.
            dt: The time step in s.
            cell_volume: The volume of one cell in m^3 (the whole volume
                without ``cell_size``).
            cell_size: The size of the cells along each position component,
                in the units of the positions; one cell for all if None.

        Returns:
            The counters of the step.

        Raises:
            KeyError: If a product species is missing from ``species``.
        """
        if cell_volume <= 0.0:
            raise ValueError("cell_volume must be > 0")
        diagnostics = ChargedDiagnostics()
        names = [name for pair in self._coulomb_pairs for name in pair] + [
            name for reaction in self._reactions for name in reaction.species
        ]
        cells, num_cells = self._cells(species, names, cell_size)

        for a, b in self._coulomb_pairs:
            pairs = self._pairs(species, a, b, cells, num_cells, cell_volume)
            key = f"{a}:{b}"
            diagnostics.coulomb_pairs[key] = (
                0 if pairs is None else int(pairs.first.size)
            )
            if pairs is not None and pairs.first.size:
                self._scatter(species, a, b, pairs, dt)

        removed_weight: dict[str, Any] = {
            name: np.zeros(int(species[name].weights.shape[0]))
            for reaction in self._reactions
            for name in reaction.species
            if name in species
        }
        created: dict[str, list[NewMarkers]] = {}
        for reaction in self._reactions:
            a, b = reaction.species
            pairs = self._pairs(species, a, b, cells, num_cells, cell_volume)
            if pairs is None or pairs.first.size == 0:
                diagnostics.reactions[reaction.name] = 0
                diagnostics.weighted_reactions[reaction.name] = 0.0
                continue
            self._react(
                species,
                reaction,
                pairs,
                dt,
                cell_volume,
                cells,
                num_cells,
                removed_weight,
                created,
                diagnostics,
            )
        if (
            diagnostics.max_reaction_probability > self._max_probability
            and not self._warned_probability
        ):
            warnings.warn(
                f"Charged reaction probability per pair "
                f"{diagnostics.max_reaction_probability:.3g} exceeds "
                f"{self._max_probability:.3g}; markers react at most once per step, "
                "so reactions are lost. Reduce dt.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_probability = True

        for name, used in removed_weight.items():
            used = np.asarray(used, dtype=float)
            if not np.any(used > 0.0):
                continue
            markers = species[name]
            weights = np.asarray(xp.to_numpy(markers.weights), dtype=float)
            remaining = weights - used
            gone = remaining <= 1.0e-12 * np.maximum(weights, 1e-300)
            partial = (used > 0.0) & ~gone
            if np.any(partial):
                markers.weights[xp.asarray(np.flatnonzero(partial))] = xp.asarray(
                    remaining[partial]
                )
            if np.any(gone):
                markers.remove(xp.asarray(gone))
        for product, batches in created.items():
            if product not in species:
                raise KeyError(f"Product species {product!r} is not simulated")
            species[product].add(
                positions=xp.concatenate([batch.positions for batch in batches]),
                velocities=xp.concatenate([batch.velocities for batch in batches]),
                weights=xp.concatenate([batch.weights for batch in batches]),
            )
        return diagnostics


def _scatter_sum(indices: Array, values: Array, size: int) -> Array:
    """Return the sum of the rows of ``values`` per index, shape ``(size, 3)``."""
    return xp.stack(
        [
            xp.bincount(indices, weights=values[:, axis], minlength=size)
            for axis in range(values.shape[1])
        ],
        axis=1,
    )
