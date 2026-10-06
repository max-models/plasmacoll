"""Tabulated collision cross sections and readers for LXCat files and plain tables.

A :class:`CrossSection` is a table of cross section (m^2) against collision
energy (eV), interpolated linearly between the table points, held at the last
value beyond the table and zero below the threshold. Tables come from a
two-column text file (:meth:`CrossSection.from_table`), from an LXCat file
(:func:`read_lxcat`), or from a function sampled on an energy grid
(:meth:`CrossSection.from_function`).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cunumpy as xp
import numpy as np

from plasmacoll._types import Array, PathLikeStr
from plasmacoll.constants import ELEMENTARY_CHARGE

__all__ = [
    "LXCAT_PROCESS_KINDS",
    "CrossSection",
    "LXCatProcess",
    "elastic_from_lxcat",
    "find_lxcat_process",
    "read_lxcat",
]

#: The process keywords an LXCat cross-section file can contain.
LXCAT_PROCESS_KINDS = ("ELASTIC", "EFFECTIVE", "EXCITATION", "IONIZATION", "ATTACHMENT")


@dataclass(frozen=True)
class CrossSection:
    """A cross section tabulated against collision energy.

    ``energy`` is in eV and ``sigma`` in m^2. Values are interpolated linearly
    between table points, as prescribed by the Turner et al. (2013) benchmark.
    Beyond the last point the last value is used. Below ``threshold`` (eV)
    the cross section is zero.

    Attributes:
        energy: Table energies in eV, non-decreasing.
        sigma: Cross sections in m^2 at ``energy``, non-negative.
        threshold: Energy in eV below which the cross section is zero.
        label: A name for plots and messages.
    """

    energy: Array
    sigma: Array
    threshold: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        """Validate and normalize the table."""
        energy = xp.asarray(self.energy, dtype=float)
        sigma = xp.asarray(self.sigma, dtype=float)
        if energy.ndim != 1 or energy.shape != sigma.shape:
            raise ValueError("energy and sigma must be 1D arrays of equal length")
        if energy.size < 2:
            raise ValueError("a cross-section table needs at least two points")
        if bool(xp.any(xp.diff(energy) < 0.0)):
            raise ValueError("cross-section energies must be non-decreasing")
        if bool(xp.any(sigma < 0.0)):
            raise ValueError("cross sections must be non-negative")
        if self.threshold < 0.0:
            raise ValueError("threshold must be >= 0")
        object.__setattr__(self, "energy", energy)
        object.__setattr__(self, "sigma", sigma)

    @property
    def max_energy(self) -> float:
        """Return the largest tabulated energy in eV."""
        return float(self.energy[-1])

    @classmethod
    def constant(
        cls,
        sigma: float,
        threshold: float = 0.0,
        max_energy: float = 1.0e6,
        label: str = "",
    ) -> CrossSection:
        """Return a cross section that is ``sigma`` (m^2) at every energy above threshold.

        Args:
            sigma: The cross section in m^2.
            threshold: Energy in eV below which the cross section is zero.
            max_energy: Last table energy in eV; the value continues beyond it anyway.
            label: A name for plots and messages.
        """
        return cls(
            energy=xp.asarray([0.0, float(max_energy)]),
            sigma=xp.asarray([float(sigma), float(sigma)]),
            threshold=threshold,
            label=label or f"constant {sigma:.3g} m^2",
        )

    @classmethod
    def maxwell_molecule(
        cls,
        rate_coefficient: float,
        mass: float,
        threshold: float = 0.0,
        min_energy: float = 1.0e-6,
        max_energy: float = 1.0e6,
        num: int = 2001,
        label: str = "",
    ) -> CrossSection:
        """Return ``sigma = K / g``, whose collision frequency ``n K`` is speed independent.

        ``g = sqrt(2 E / mass)`` is the relative speed at energy ``E``, so pass
        the reduced mass for a process with ``energy_frame="center_of_mass"``
        and the incident mass for ``energy_frame="lab"``. With a constant
        frequency the number of collisions, and the relaxation of moments such
        as the temperature, have closed-form solutions, which makes this the
        cross section of choice for benchmarks. The table has ``num``
        logarithmically spaced points from ``min_energy`` to ``max_energy`` (eV);
        below and above them the end values are used.

        Args:
            rate_coefficient: ``K = sigma g`` in m^3/s.
            mass: The mass in kg that converts energy to relative speed.
            threshold: Energy in eV below which the cross section is zero.
            min_energy: First table energy in eV.
            max_energy: Last table energy in eV.
            num: Number of table points.
            label: A name for plots and messages.
        """
        if rate_coefficient < 0.0:
            raise ValueError("rate_coefficient must be >= 0")
        if mass <= 0.0:
            raise ValueError("mass must be > 0")
        if not 0.0 < min_energy < max_energy:
            raise ValueError("need 0 < min_energy < max_energy")
        energy = np.geomspace(min_energy, max_energy, num)
        speed = np.sqrt(2.0 * energy * ELEMENTARY_CHARGE / mass)
        return cls(
            energy=xp.asarray(energy),
            sigma=xp.asarray(rate_coefficient / speed),
            threshold=threshold,
            label=label or f"Maxwell molecule K = {rate_coefficient:.3g} m^3/s",
        )

    @classmethod
    def from_function(
        cls,
        function: Callable[[np.ndarray], np.ndarray],
        energy: Iterable[float] | Array,
        threshold: float = 0.0,
        label: str = "",
    ) -> CrossSection:
        """Tabulate ``function(energy)`` (m^2) on the energy grid ``energy`` (eV).

        The function is called once with a NumPy array of the grid energies.
        Choose the grid fine enough that linear interpolation between its
        points follows the function, for example ``numpy.geomspace`` for a
        function that varies on a logarithmic scale.

        Args:
            function: Cross section in m^2 as a function of energy in eV.
            energy: The table energies in eV, non-decreasing.
            threshold: Energy in eV below which the cross section is zero.
            label: A name for plots and messages.
        """
        grid = np.asarray(xp.to_numpy(xp.asarray(energy, dtype=float)))
        values = np.asarray(function(grid), dtype=float)
        return cls(
            energy=xp.asarray(grid),
            sigma=xp.asarray(values),
            threshold=threshold,
            label=label,
        )

    @classmethod
    def vibrational_mixture(
        cls,
        level_cross_sections: Sequence[CrossSection],
        populations: Sequence[float] | Array,
        label: str = "",
    ) -> CrossSection:
        """Combine level-resolved cross sections into a population-weighted mean.

        ``sigma_mix(E) = sum_v w_v * sigma_v(E)``, with ``w_v`` normalized from
        ``populations`` (they need not already sum to 1). A level's own
        threshold and tail behaviour are respected because each ``sigma_v(E)``
        is evaluated through its own :meth:`__call__` before summing, and a
        table point is placed on either side of every threshold so that the
        step is not blurred into a ramp.

        This performs the weighted average only; it does not supply the
        per-level cross sections. Use it with level-resolved data (for example
        H2(v) dissociative attachment from Wadehra, Phys. Rev. A 29, 106 (1984),
        or an LXCat entry) and the populations of a
        :class:`~plasmacoll.vibrational.VibrationalDistribution`.

        If ``populations`` covers only some of a distribution's levels, the
        omitted levels' population is redistributed among the included ones by
        the normalization; pass the full set of weights if that is not the
        intended mixture.

        Args:
            level_cross_sections: One cross section per level.
            populations: The weight of each level, non-negative.
            label: A name for plots and messages.
        """
        if len(level_cross_sections) == 0:
            raise ValueError("level_cross_sections must not be empty")
        if len(level_cross_sections) != len(populations):
            raise ValueError(
                "level_cross_sections and populations must have equal length, "
                f"got {len(level_cross_sections)} and {len(populations)}"
            )
        weights = np.asarray(xp.to_numpy(xp.asarray(populations, dtype=float)))
        if np.any(weights < 0.0):
            raise ValueError("populations must be non-negative")
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0:
            raise ValueError("populations must sum to a positive value")
        weights = weights / total_weight

        threshold_points = [
            point
            for level in level_cross_sections
            if level.threshold > 0.0
            for point in (level.threshold * (1.0 - 1.0e-9), level.threshold)
        ]
        energy = xp.unique(
            xp.concatenate(
                [level.energy for level in level_cross_sections]
                + ([xp.asarray(threshold_points)] if threshold_points else [])
            )
        )
        sigma = xp.zeros_like(energy)
        for level, weight in zip(level_cross_sections, weights, strict=True):
            if weight > 0.0:
                sigma = sigma + float(weight) * level(energy)
        return cls(
            energy=energy,
            sigma=sigma,
            threshold=0.0,  # already applied in each level's contribution
            label=label or "vibrational mixture",
        )

    @classmethod
    def from_table(
        cls,
        path: PathLikeStr,
        threshold: float = 0.0,
        delimiter: str | None = None,
        energy_unit: float = 1.0,
        sigma_unit: float = 1.0,
        label: str = "",
    ) -> CrossSection:
        """Read a two-column (energy, sigma) text table.

        Lines that do not start with two numbers (headers, comments) are skipped.

        Args:
            path: The text file.
            threshold: Energy in eV below which the cross section is zero.
            delimiter: Column separator; whitespace if None.
            energy_unit: Factor converting the first column to eV.
            sigma_unit: Factor converting the second column to m^2.
            label: A name for plots and messages; the file name if empty.
        """
        energies = []
        sigmas = []
        for line in Path(path).read_text().splitlines():
            parts = line.replace(delimiter, " ").split() if delimiter else line.split()
            if len(parts) < 2:
                continue
            try:
                energy, sigma = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            energies.append(energy)
            sigmas.append(sigma)
        return cls(
            energy=xp.asarray(energies) * energy_unit,
            sigma=xp.asarray(sigmas) * sigma_unit,
            threshold=threshold,
            label=label or Path(path).stem,
        )

    @classmethod
    def elastic_from_effective(
        cls,
        effective: CrossSection,
        inelastic: Sequence[CrossSection],
        label: str = "",
    ) -> CrossSection:
        """Return the elastic momentum-transfer cross section ``sigma_eff - sum sigma_inel``.

        An LXCat ``EFFECTIVE`` cross section is the elastic momentum-transfer
        cross section plus every inelastic cross section of the set. Used as an
        ``elastic`` process next to the inelastic ones it would count their
        collisions twice, so subtract them first. The result is tabulated on
        the energies of all the tables, with points on either side of every
        threshold, and clipped at zero; a warning is issued if the inelastic
        sum exceeds the effective cross section by more than 1 %, which means
        the set is inconsistent.

        Args:
            effective: The effective (total momentum-transfer) cross section.
            inelastic: The excitation, ionization and attachment cross sections
                of the same set.
            label: A name for plots and messages.
        """
        threshold_points = [
            point
            for table in inelastic
            if table.threshold > 0.0
            for point in (table.threshold * (1.0 - 1.0e-9), table.threshold)
        ]
        energy = xp.unique(
            xp.concatenate(
                [effective.energy]
                + [table.energy for table in inelastic]
                + ([xp.asarray(threshold_points)] if threshold_points else [])
            )
        )
        reference = effective(energy)
        sigma = reference
        for table in inelastic:
            sigma = sigma - table(energy)
        if bool(xp.any(sigma < -0.01 * reference)):
            warnings.warn(
                "The inelastic cross sections exceed the effective cross section; "
                "the elastic cross section is clipped at zero there.",
                RuntimeWarning,
                stacklevel=2,
            )
        return cls(
            energy=energy,
            sigma=xp.clip(sigma, 0.0, None),
            threshold=effective.threshold,
            label=label or f"elastic from {effective.label or 'effective'}",
        )

    def with_threshold(self, threshold: float) -> CrossSection:
        """Return a copy of this table with another threshold (eV)."""
        return CrossSection(
            energy=self.energy,
            sigma=self.sigma,
            threshold=threshold,
            label=self.label,
        )

    def __call__(self, energy: Array | float) -> Array:
        """Return the cross section in m^2 at ``energy`` (eV)."""
        energy = xp.asarray(energy, dtype=float)
        sigma = xp.interp(energy, self.energy, self.sigma)
        return xp.where(energy < self.threshold, 0.0, sigma)


@dataclass(frozen=True)
class LXCatProcess:
    """One process block of an LXCat cross-section file.

    Attributes:
        kind: The block keyword, one of :data:`LXCAT_PROCESS_KINDS`.
        target: The left side of the reaction line, e.g. ``Ar``.
        product: The right side of the reaction line, e.g. ``Ar^+`` (empty if none).
        parameter: The mass ratio ``m_e/M`` for ``ELASTIC`` and ``EFFECTIVE``,
            the threshold in eV for ``EXCITATION`` and ``IONIZATION``, zero for
            ``ATTACHMENT``.
        cross_section: The table.
        comments: The ``KEY: value`` lines of the block, e.g. ``PROCESS``.
    """

    kind: str
    target: str
    product: str
    parameter: float
    cross_section: CrossSection
    comments: dict[str, str] = field(default_factory=dict)

    @property
    def mass_ratio(self) -> float | None:
        """Return ``m_e/M`` for ELASTIC and EFFECTIVE processes, None otherwise."""
        if self.kind in ("ELASTIC", "EFFECTIVE"):
            return self.parameter
        return None

    @property
    def threshold(self) -> float:
        """Return the energy threshold in eV (zero for momentum transfer)."""
        return self.cross_section.threshold


def _split_reaction(target_line: str) -> tuple[str, str]:
    """Split an LXCat target line such as ``Ar -> Ar^+`` into its two sides."""
    for arrow in ("<->", "->"):
        if arrow in target_line:
            target, _, product = target_line.partition(arrow)
            return target.strip(), product.strip()
    return target_line, ""


def read_lxcat(path: PathLikeStr) -> list[LXCatProcess]:
    """Parse an LXCat cross-section file.

    Each process block starts with a keyword (``ELASTIC``, ``EFFECTIVE``,
    ``EXCITATION``, ``IONIZATION`` or ``ATTACHMENT``), followed by the target
    line, a parameter line for all but ``ATTACHMENT`` (mass ratio for
    momentum transfer, threshold in eV otherwise), optional ``KEY: value``
    comment lines, and a table of energy (eV) and cross section (m^2) framed
    by lines of dashes. Everything outside the blocks is skipped.

    Args:
        path: The file, as downloaded from https://lxcat.net.

    Returns:
        The processes in the order of the file.
    """
    lines = Path(path).read_text(errors="replace").splitlines()
    processes = []
    index = 0
    while index < len(lines):
        kind = lines[index].strip().upper()
        if kind not in LXCAT_PROCESS_KINDS:
            index += 1
            continue
        target_line = lines[index + 1].strip()
        index += 2
        target, product = _split_reaction(target_line)
        parameter = 0.0
        if kind != "ATTACHMENT":
            parameter = float(lines[index].split()[0])
            index += 1
        comments: dict[str, str] = {}
        while index < len(lines) and not lines[index].startswith("-----"):
            key, sep, value = lines[index].partition(":")
            if sep:
                comments[key.strip()] = value.strip()
            index += 1
        index += 1
        energies = []
        sigmas = []
        while index < len(lines) and not lines[index].startswith("-----"):
            parts = lines[index].split()
            if len(parts) >= 2:
                energies.append(float(parts[0]))
                sigmas.append(float(parts[1]))
            index += 1
        index += 1
        threshold = parameter if kind in ("EXCITATION", "IONIZATION") else 0.0
        processes.append(
            LXCatProcess(
                kind=kind,
                target=target,
                product=product,
                parameter=parameter,
                cross_section=CrossSection(
                    energy=xp.asarray(energies),
                    sigma=xp.asarray(sigmas),
                    threshold=threshold,
                    label=comments.get("PROCESS", f"{kind} {target_line}"),
                ),
                comments=comments,
            ),
        )
    return processes


def find_lxcat_process(
    processes: Sequence[LXCatProcess],
    wanted: str,
) -> LXCatProcess:
    """Return the first process whose kind or ``PROCESS`` comment is ``wanted``.

    Args:
        processes: Processes from :func:`read_lxcat`.
        wanted: A kind such as ``"IONIZATION"`` or a full ``PROCESS`` comment
            such as ``"E + Ar -> E + E + Ar+, Ionization"``.

    Raises:
        KeyError: If no process matches.
    """
    for process in processes:
        if wanted in (process.kind, process.comments.get("PROCESS", "")):
            return process
    raise KeyError(f"No LXCat process {wanted!r}")


def elastic_from_lxcat(
    processes: Sequence[LXCatProcess],
    target: str | None = None,
) -> CrossSection:
    """Return the elastic momentum-transfer cross section of an LXCat set.

    If the set has an ``ELASTIC`` block it is returned as is. Otherwise the
    ``EXCITATION``, ``IONIZATION`` and ``ATTACHMENT`` cross sections are
    subtracted from the ``EFFECTIVE`` one with
    :meth:`CrossSection.elastic_from_effective`.

    Args:
        processes: Processes from :func:`read_lxcat`.
        target: Use only the blocks of this target (e.g. ``"Ar"``); all if None.

    Raises:
        KeyError: If the set has neither an ``ELASTIC`` nor an ``EFFECTIVE`` block.
    """
    selected = [
        process for process in processes if target is None or process.target == target
    ]
    for process in selected:
        if process.kind == "ELASTIC":
            return process.cross_section
    for process in selected:
        if process.kind == "EFFECTIVE":
            inelastic = [
                other.cross_section
                for other in selected
                if other.kind in ("EXCITATION", "IONIZATION", "ATTACHMENT")
            ]
            return CrossSection.elastic_from_effective(process.cross_section, inelastic)
    raise KeyError(f"No ELASTIC or EFFECTIVE LXCat process for target {target!r}")
