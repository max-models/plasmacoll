"""Charge states of a fast beam passing through gas and plasma targets.

A fast beam of negative ions, neutrals and positive ions (``"-"``, ``"0"``,
``"+"``) changes charge state on its targets: single stripping ``- -> 0``,
double stripping ``- -> +``, ionization of the neutrals ``0 -> +``, electron
capture ``+ -> 0`` and ``0 -> -``. At a fixed beam energy each cross section
is a constant, and the fractions ``F`` obey

    dF / dz = sum_t n_t(z) M_t F,

with ``n_t`` the density of target ``t`` (gas molecules, atoms, plasma
electrons, plasma ions) and ``M_t`` its transition matrix: ``sigma_ij`` from
state ``i`` to ``j`` adds to ``M[j, i]`` and subtracts from ``M[i, i]``. In
terms of the line densities ``pi_t = int n_t dz`` the solution over a step
where the targets' density ratios are constant is the matrix exponential of
``sum_t delta pi_t M_t``, which :func:`charge_state_fractions` takes step by
step along the path.

The cross sections are inputs, as everywhere in plasmacoll: the beam stripping
data of hydrogen and deuterium (e.g. the ORNL "Redbook", Barnett 1990, and
IAEA/ALADDIN) are not reproduced here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import scope_profiler as sp

from plasmacoll._profiling import PROFILE_TAGS

__all__ = ["ChargeStateTransitions", "charge_state_fractions"]

#: The charge states of a hydrogen (or deuterium) beam.
HYDROGEN_STATES = ("-", "0", "+")


def _expm(matrix: np.ndarray) -> np.ndarray:
    """Return ``exp(matrix)`` by scaling and squaring of a Taylor series.

    For the small matrices here (3 x 3) a degree-18 series after scaling the
    norm below 1/2 is accurate to round-off, also for defective matrices
    (equal cross sections), where an eigen-decomposition is not.
    """
    norm = float(np.max(np.sum(np.abs(matrix), axis=0)))
    squarings = max(0, int(np.ceil(np.log2(norm / 0.5)))) if norm > 0.5 else 0
    scaled = matrix / 2.0**squarings
    term = np.eye(matrix.shape[0])
    result = term.copy()
    for k in range(1, 19):
        term = term @ scaled / k
        result = result + term
    for _ in range(squarings):
        result = result @ result
    return result


@dataclass(frozen=True)
class ChargeStateTransitions:
    """The charge-changing cross sections of a beam, per target.

    Attributes:
        cross_sections: ``{target: {(from_state, to_state): sigma}}`` in m^2.
        states: The charge states, in the order of the fractions.
    """

    cross_sections: Mapping[str, Mapping[tuple[str, str], float]]
    states: Sequence[str] = field(default=HYDROGEN_STATES)

    def __post_init__(self) -> None:
        """Validate the states and the cross sections."""
        states = tuple(self.states)
        if len(set(states)) != len(states) or len(states) < 2:
            raise ValueError("states must be at least two distinct names")
        for target, table in self.cross_sections.items():
            for (start, end), sigma in table.items():
                if start not in states or end not in states or start == end:
                    raise ValueError(
                        f"target {target!r}: no transition {start!r} -> {end!r} "
                        f"between the states {states}"
                    )
                if sigma < 0.0:
                    raise ValueError(f"target {target!r}: cross sections must be >= 0")
        object.__setattr__(self, "states", states)
        object.__setattr__(
            self,
            "cross_sections",
            {t: dict(table) for t, table in self.cross_sections.items()},
        )

    @property
    def targets(self) -> tuple[str, ...]:
        """The target names."""
        return tuple(self.cross_sections)

    def matrix(self, target: str) -> np.ndarray:
        """Return ``M_t`` in m^2: ``dF/dpi_t = M_t F`` on the target alone."""
        index = {state: k for k, state in enumerate(self.states)}
        matrix = np.zeros((len(self.states), len(self.states)))
        for (start, end), sigma in self.cross_sections[target].items():
            matrix[index[end], index[start]] += sigma
            matrix[index[start], index[start]] -= sigma
        return matrix

    def equilibrium(self, weights: Mapping[str, float] | None = None) -> np.ndarray:
        """Return the fractions after a thick target (the null vector of ``M``).

        ``weights`` are the targets' shares of the line density (all equal if
        ``None``); only their ratios matter.
        """
        weights = weights or {t: 1.0 for t in self.targets}
        total = sum(w * self.matrix(t) for t, w in weights.items())
        values, vectors = np.linalg.eig(total)
        vector = np.real(vectors[:, np.argmin(np.abs(values))])
        return vector / vector.sum()


@sp.profile("plasmacoll.charge_state_fractions", tags=PROFILE_TAGS)
def charge_state_fractions(
    transitions: ChargeStateTransitions,
    line_density: Mapping[str, Sequence[float] | np.ndarray],
    initial: Sequence[float] | Mapping[str, float] | None = None,
) -> np.ndarray:
    """Return the fractions of every state at every point of a path.

    Args:
        transitions: The cross sections.
        line_density: ``{target: pi_t}``, the line density in m^-2 of each
            target accumulated from the start of the path, at the same points
            for every target (non-decreasing, starting at the first point).
            Targets missing from ``transitions`` are refused; targets of
            ``transitions`` missing here contribute nothing.
        initial: The fractions at the start, by state name or in the order of
            ``transitions.states``; all in the first state by default (a pure
            negative-ion beam).

    Returns:
        ``(points, states)`` fractions; each row sums to the initial sum.
    """
    if not line_density:
        raise ValueError("line_density needs at least one target")
    unknown = set(line_density) - set(transitions.targets)
    if unknown:
        raise ValueError(f"no cross sections for the target(s) {sorted(unknown)}")
    columns = {t: np.asarray(v, dtype=float) for t, v in line_density.items()}
    lengths = {c.shape for c in columns.values()}
    if len(lengths) != 1 or len(next(iter(lengths))) != 1:
        raise ValueError("every target needs a 1D line density at the same points")
    for target, column in columns.items():
        if np.any(np.diff(column) < 0.0):
            raise ValueError(f"the line density of {target!r} must be non-decreasing")
    states = transitions.states
    if initial is None:
        start = np.zeros(len(states))
        start[0] = 1.0
    elif isinstance(initial, Mapping):
        start = np.array([float(initial.get(s, 0.0)) for s in states])
    else:
        start = np.asarray(initial, dtype=float)
    matrices = {t: transitions.matrix(t) for t in columns}
    num = next(iter(columns.values())).size
    fractions = np.empty((num, len(states)))
    fractions[0] = start
    for k in range(1, num):
        step = sum(
            ((columns[t][k] - columns[t][k - 1]) * matrices[t] for t in columns),
            np.zeros((len(states), len(states))),
        )
        fractions[k] = _expm(step) @ fractions[k - 1]
    return fractions
