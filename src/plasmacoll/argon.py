"""Electrons and Ar+ ions in argon, the gas of magnetron and glow-discharge models.

Electron-argon cross sections come from a set the user downloads from LXCat:
the standard ones are the Phelps, SIGLO, Biagi, Hayashi and IST-Lisbon
databases. LXCat does not allow third parties to redistribute its data, so
none is included here; :func:`argon_electron_processes` turns a downloaded
file into the processes of :class:`~plasmacoll.mcc.MonteCarloCollisions`.

The Ar+ + Ar processes are the isotropic and backward elastic scattering
components of Phelps:

- ``revision="1994"``: the fits of A. V. Phelps, J. Appl. Phys. 76, 747
  (1994), Eqs. (2) and (3), which the LXCat Phelps database tabulates and
  which Boeuf's magnetron simulations use (arXiv:2211.17051). With the
  laboratory ion energy ``eps`` in eV (twice the centre-of-mass energy)::

      Q_i = 2e-19 / (eps^0.5 (1 + eps)) + 3e-19 eps / (1 + eps/3)^2.3   m^2
      Q_m = 1.15e-18 eps^-0.1 (1 + 0.015/eps)^0.6                       m^2
      Q_b = (Q_m - Q_i) / 2

- ``revision="2005"``: Phelps' later fits from his collision-data pages
  (``collision_data/ionneutral/IONATOM.TXT``, updated 2005; cite as A. V.
  Phelps, private communication), with the centre-of-mass energy ``E`` in eV::

      Q_i = 21.6e-20 / E^0.5                                            m^2
      Q_b = 52e-20 / E^0.08 / (1 + 0.08/E) / (1 + E/1000)^0.3           m^2

In both, the momentum-transfer cross section is ``Q_m = Q_i + 2 Q_b`` and at
high energies ``Q_b`` is the symmetric charge-exchange cross section.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from plasmacoll._types import PathLikeStr
from plasmacoll.constants import ATOMIC_MASS
from plasmacoll.cross_sections import (
    CrossSection,
    LXCatProcess,
    elastic_from_lxcat,
    read_lxcat,
)
from plasmacoll.process import CollisionProcess

__all__ = [
    "ARGON_IONIZATION_ENERGY",
    "ARGON_MASS",
    "ARGON_OPAL_ENERGY",
    "PHELPS_REVISIONS",
    "argon_electron_processes",
    "argon_ion_cross_sections",
    "argon_ion_processes",
    "phelps_argon_ion_backscatter",
    "phelps_argon_ion_isotropic",
]

#: Mass of an argon atom in kg (standard atomic weight 39.948 u).
ARGON_MASS = 39.948 * ATOMIC_MASS
#: First ionization energy of argon in eV (NIST Atomic Spectra Database).
ARGON_IONIZATION_ENERGY = 15.7596119
#: The parameter ``w`` in eV of Opal-Peterson-Beaty energy sharing in argon
#: (Opal, Peterson and Beaty, J. Chem. Phys. 55, 4100 (1971)).
ARGON_OPAL_ENERGY = 10.0
#: The Phelps fits :func:`argon_ion_cross_sections` offers.
PHELPS_REVISIONS = ("1994", "2005")


def _check_revision(revision: str) -> None:
    if revision not in PHELPS_REVISIONS:
        raise ValueError(f"revision must be one of {PHELPS_REVISIONS}")


def _phelps_1994(energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Q_i, Q_b)`` in m^2 of Phelps (1994) at centre-of-mass energies (eV)."""
    eps = 2.0 * energy  # laboratory energy of the ion, the target at rest
    isotropic = (
        2.0e-19 / (np.sqrt(eps) * (1.0 + eps))
        + 3.0e-19 * eps / (1.0 + eps / 3.0) ** 2.3
    )
    momentum = 1.15e-18 * eps**-0.1 * (1.0 + 0.015 / eps) ** 0.6
    return isotropic, 0.5 * (momentum - isotropic)


def _phelps_2005(energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Q_i, Q_b)`` in m^2 of Phelps' 2005 fits at centre-of-mass energies (eV)."""
    isotropic = 21.6e-20 / np.sqrt(energy)
    backscatter = (
        52.0e-20 / energy**0.08 / (1.0 + 0.08 / energy) / (1.0 + energy / 1000.0) ** 0.3
    )
    return isotropic, backscatter


def _phelps(energy: np.ndarray, revision: str) -> tuple[np.ndarray, np.ndarray]:
    _check_revision(revision)
    energy = np.asarray(energy, dtype=float)
    if np.any(energy <= 0.0):
        raise ValueError("the Phelps fits need energies > 0")
    return _phelps_1994(energy) if revision == "1994" else _phelps_2005(energy)


def phelps_argon_ion_isotropic(
    energy: np.ndarray, revision: str = "1994"
) -> np.ndarray:
    """Return the isotropic Ar+ + Ar cross section ``Q_i`` in m^2.

    Args:
        energy: Centre-of-mass energies in eV, > 0.
        revision: One of :data:`PHELPS_REVISIONS`, see the module docstring.
    """
    return _phelps(energy, revision)[0]


def phelps_argon_ion_backscatter(
    energy: np.ndarray, revision: str = "1994"
) -> np.ndarray:
    """Return the backward-scattering Ar+ + Ar cross section ``Q_b`` in m^2.

    Args:
        energy: Centre-of-mass energies in eV, > 0.
        revision: One of :data:`PHELPS_REVISIONS`, see the module docstring.
    """
    return _phelps(energy, revision)[1]


def argon_ion_cross_sections(
    revision: str = "1994",
    min_energy: float = 1.0e-4,
    max_energy: float = 1.0e4,
    num: int = 401,
) -> dict[str, CrossSection]:
    """Tabulate Phelps' Ar+ + Ar fits against the centre-of-mass energy.

    The tables have ``num`` logarithmically spaced points from ``min_energy``
    to ``max_energy`` (eV), the range of the LXCat tables; below and above
    them the end values are used.

    Args:
        revision: One of :data:`PHELPS_REVISIONS`, see the module docstring.
        min_energy: First table energy in eV.
        max_energy: Last table energy in eV.
        num: Number of table points.

    Returns:
        ``{"isotropic": Q_i, "backscatter": Q_b}``.
    """
    _check_revision(revision)
    if not 0.0 < min_energy < max_energy:
        raise ValueError("need 0 < min_energy < max_energy")
    energy = np.geomspace(min_energy, max_energy, num)
    source = f"Phelps {revision}"
    return {
        "isotropic": CrossSection.from_function(
            lambda e: phelps_argon_ion_isotropic(e, revision),
            energy,
            label=f"Ar+ + Ar isotropic ({source})",
        ),
        "backscatter": CrossSection.from_function(
            lambda e: phelps_argon_ion_backscatter(e, revision),
            energy,
            label=f"Ar+ + Ar backscatter ({source})",
        ),
    }


def _ion_tables_from_lxcat(
    processes: Sequence[LXCatProcess], target: str
) -> dict[str, CrossSection]:
    tables = {}
    for key, kind in (("isotropic", "ISOTROPIC"), ("backscatter", "BACKSCAT")):
        matches = [p for p in processes if p.kind == kind and p.target == target]
        if not matches:
            raise KeyError(f"No LXCat {kind} block for target {target!r}")
        tables[key] = matches[0].cross_section
    return tables


def argon_ion_processes(
    cross_sections: Mapping[str, CrossSection]
    | Sequence[LXCatProcess]
    | PathLikeStr
    | None = None,
    *,
    background: str = "Ar",
    neutral: str | None = None,
    revision: str = "1994",
) -> list[CollisionProcess]:
    """Return the isotropic and backscatter processes of Ar+ in argon.

    Both are looked up at the centre-of-mass energy, as Phelps and LXCat give
    them. Backscatter reverses the relative velocity, so the ion leaves with
    the neutral's velocity: the charge exchange.

    Args:
        cross_sections: ``None`` for Phelps' fits (:func:`argon_ion_cross_sections`),
            a mapping with the keys ``"isotropic"`` and ``"backscatter"``, or
            the ``ISOTROPIC`` and ``BACKSCAT`` blocks of the LXCat Phelps
            database, read with :func:`~plasmacoll.cross_sections.read_lxcat`
            or given as the path of the file.
        background: The name of the argon background.
        neutral: A species name for the fast neutral of the charge exchange;
            dropped if None.
        revision: The Phelps fits if ``cross_sections`` is None.

    Raises:
        KeyError: If a table is missing.
    """
    if cross_sections is None:
        tables: Mapping[str, CrossSection] = argon_ion_cross_sections(revision)
    elif isinstance(cross_sections, Mapping):
        tables = cross_sections
    else:
        processes = (
            cross_sections
            if isinstance(cross_sections, Sequence)
            and not isinstance(cross_sections, str)
            else read_lxcat(cross_sections)
        )
        tables = _ion_tables_from_lxcat(processes, background)
    missing = {"isotropic", "backscatter"} - set(tables)
    if missing:
        raise KeyError(f"Missing cross sections for {sorted(missing)}")
    return [
        CollisionProcess(
            kind="elastic",
            background=background,
            cross_section=tables["isotropic"],
            name="ar_ion_isotropic",
            energy_frame="center_of_mass",
        ),
        CollisionProcess(
            kind="backscatter",
            background=background,
            cross_section=tables["backscatter"],
            name="ar_ion_backscatter",
            energy_frame="center_of_mass",
            products={} if neutral is None else {"neutral": neutral},
        ),
    ]


def argon_electron_processes(
    lxcat: Sequence[LXCatProcess] | PathLikeStr,
    *,
    background: str = "Ar",
    target: str = "Ar",
    electron: str = "e",
    ion: str = "Ar+",
    energy_sharing: str = "opal",
    sharing_energy: float | None = ARGON_OPAL_ENERGY,
) -> list[CollisionProcess]:
    """Return the electron-argon processes of an LXCat set.

    The set must have an ``ELASTIC`` or an ``EFFECTIVE`` block (the latter
    less the inelastic blocks, see
    :func:`~plasmacoll.cross_sections.elastic_from_lxcat`) and at least one
    ``IONIZATION`` block; every ``EXCITATION`` block becomes an excitation
    process (the SIGLO set lumps them into one level at 11.5 eV). All are
    looked up at the electron's energy (``energy_frame="lab"``) with isotropic
    scattering, which is what a momentum-transfer cross section is for.

    Args:
        lxcat: The processes from :func:`~plasmacoll.cross_sections.read_lxcat`,
            or the path of the downloaded file.
        background: The name of the argon background.
        target: The target name of the blocks to use (other gases and ion
            blocks in the same file are skipped).
        electron: The electron species.
        ion: The ion species ionization creates.
        energy_sharing: Ionization energy sharing, see
            :class:`~plasmacoll.process.CollisionProcess`.
        sharing_energy: The ``w`` of ``"opal"`` sharing in eV.

    Raises:
        KeyError: If the set has no momentum-transfer or no ionization block.
    """
    processes = (
        lxcat
        if isinstance(lxcat, Sequence) and not isinstance(lxcat, str)
        else read_lxcat(lxcat)
    )
    selected = [p for p in processes if p.target == target]
    result = [
        CollisionProcess(
            kind="elastic",
            background=background,
            cross_section=elastic_from_lxcat(selected, target=target),
            name="elastic",
            energy_frame="lab",
        )
    ]
    excitations = [p for p in selected if p.kind == "EXCITATION"]
    ionizations = [p for p in selected if p.kind == "IONIZATION"]
    if not ionizations:
        raise KeyError(f"No LXCat IONIZATION block for target {target!r}")
    for number, process in enumerate(excitations):
        result.append(
            CollisionProcess(
                kind="excitation",
                background=background,
                cross_section=process.cross_section,
                name=f"excitation_{number}:{process.product or process.threshold}",
                energy_frame="lab",
            )
        )
    for number, process in enumerate(ionizations):
        result.append(
            CollisionProcess(
                kind="ionization",
                background=background,
                cross_section=process.cross_section,
                name="ionization" if number == 0 else f"ionization_{number}",
                energy_frame="lab",
                products={"electron": electron, "ion": ion},
                energy_sharing=energy_sharing,
                sharing_energy=sharing_energy,
            )
        )
    return result
