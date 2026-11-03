"""Model gases for the comparison with LoKI-B, and the script that made its reference.

LoKI-B (Tejero-del-Caz et al., Plasma Sources Sci. Technol. 28, 043001
(2019)) is the LisbOn KInetics two-term Boltzmann solver. Its reference
results in ``data/lokib_reference.json`` were computed with the C++ version,
LoKI-B++ (https://github.com/LoKI-Suite/LoKI-B-cpp, GPL-3.0), built from
source, on the synthetic cross sections defined here: LXCat does not allow
redistributing its data, so no measured set is used.

Two gases:

- ``Atom``, argon-like: an elastic momentum-transfer cross section, one
  excitation, ionization and attachment. It is solved with both growth models
  of LoKI-B (temporal and spatial) and both ways of sharing the energy in an
  ionization (equal sharing and the Opal-Peterson-Beaty secondary-electron
  distribution, LoKI-B's ``usingSDCS``).
- ``Mol``, H2-like: rotational levels ``J = 0..9`` of ``v = 0`` populated at
  the gas temperature, ``J -> J + 2`` excitations of the Gerjuoy-Stein form, a
  vibrational excitation ``v = 0 -> 1`` with ``v = 1`` populated at a
  vibrational temperature, an electronic excitation and an ionization. Every
  rotational and the vibrational excitation is reversible, so both codes add
  the superelastic collisions by detailed balance.

All thresholds are multiples of LoKI-B's (uniform) energy step, which rounds an
energy loss down to a whole number of cells.

Regenerate the reference with a LoKI-B++ executable (``loki``)::

    python -I src/plasmacoll/tests/unit/_lokib_models.py --loki <path to loki> \\
        --workdir <empty scratch directory>

It writes the cross sections and setup files into ``workdir``, runs LoKI-B on
each case at two energy steps (the difference estimates LoKI-B's
discretisation error), and writes the JSON next to this file.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: The reference data file.
REFERENCE = Path(__file__).parent / "data" / "lokib_reference.json"

#: Physical constants as in LoKI-B++ (Constant.h, CODATA 2014).
ATOMIC_MASS = 1.660539040e-27
BOLTZMANN_EV = 1.38064852e-23 / 1.6021766208e-19
BOHR_RADIUS = 5.2917721067e-11

GAS_TEMPERATURE = 300.0  # K
VIBRATIONAL_TEMPERATURE = 2000.0  # K, of the Mol(v=1) population

ATOM_MASS = 40.0 * ATOMIC_MASS
MOL_MASS = 2.016 * ATOMIC_MASS
OPB_WIDTH = 10.0  # eV, the Opal-Peterson-Beaty parameter of Atom

ROTATIONAL_CONSTANT = 2.0**-7  # eV, close to H2's 7.5e-3 eV
MAX_J = 9
VIBRATIONAL_ENERGY = 0.5  # eV
#: Gerjuoy-Stein: 8 pi Q^2 a0^2 / 15 with the quadrupole moment Q = 0.47 e a0^2 of H2.
GERJUOY_STEIN = 8.0 * math.pi * 0.47**2 * BOHR_RADIUS**2 / 15.0


@dataclass(frozen=True)
class Process:
    """One process of a model gas, in LoKI-B's notation.

    Attributes:
        kind: The plasmacoll kind: ``elastic``, ``excitation``, ``ionization``
            or ``attachment``.
        lower: The target state, e.g. ``Mol(X,v=0,J=0)``.
        upper: The product state of a reversible excitation, else ``""``.
        loss: The energy loss (threshold) in eV.
        energy: The table energies in eV.
        sigma: The cross section in m^2 at ``energy``.
        label: The LoKI-B type: ``Elastic``, ``Excitation``, ``Vibrational``,
            ``Rotational``, ``Ionization`` or ``Attachment``.
        weight_ratio: ``g_upper / g_lower`` of a reversible excitation, None
            for an irreversible process.
    """

    kind: str
    lower: str
    upper: str
    loss: float
    energy: np.ndarray
    sigma: np.ndarray
    label: str
    weight_ratio: float | None = None

    @property
    def reaction(self) -> str:
        """Return the reaction as LoKI-B writes it."""
        if self.kind == "elastic":
            return f"e + {self.lower} -> e + {self.lower}"
        if self.kind == "ionization":
            gas = self.lower.split("(")[0]
            return f"e + {self.lower} -> e + e + {gas}(+,X)"
        if self.kind == "attachment":
            gas = self.lower.split("(")[0]
            return f"e + {self.lower} -> {gas}(-,X)"
        arrow = "->" if self.weight_ratio is None else "<->"
        return f"e + {self.lower} {arrow} e + {self.upper}"


def _above(
    threshold: float,
    sigma: Callable[[np.ndarray], np.ndarray],
    top: float = 1.0e3,
    num: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a table from ``threshold`` (where it is zero) up to ``top``."""
    energy = threshold + np.concatenate(
        [[0.0], np.geomspace(1e-5, top - threshold, num)]
    )
    return energy, np.where(energy > threshold, sigma(energy), 0.0)


def _bethe(threshold: float, peak: float) -> Callable[[np.ndarray], np.ndarray]:
    """Return ``peak e ln(x) / x`` with ``x = eps / threshold``, zero at threshold."""

    def sigma(energy: np.ndarray) -> np.ndarray:
        x = np.maximum(energy / threshold, 1.0)
        return peak * math.e * np.log(x) / x

    return sigma


def atom_processes() -> list[Process]:
    """Return the argon-like model gas."""
    energy = np.concatenate([[0.0], np.geomspace(1e-3, 1e3, 300)])
    elastic = 8e-20 / (1.0 + (energy / 10.0) ** 0.8)
    attachment = 1e-22 * np.exp(-np.minimum(((energy - 6.0) / 1.5) ** 2, 50.0))
    attachment[attachment < 1e-40] = 0.0
    return [
        Process("elastic", "Atom(X)", "", 0.0, energy, elastic, "Elastic"),
        Process(
            "excitation",
            "Atom(X)",
            "Atom(E1)",
            11.5,
            *_above(11.5, lambda e: 2e-21 * (e - 11.5) / (1.0 + (e / 30.0) ** 1.5)),
            "Excitation",
        ),
        Process(
            "ionization",
            "Atom(X)",
            "",
            15.75,
            *_above(15.75, _bethe(15.75, 3e-20 / math.e)),
            "Ionization",
        ),
        Process("attachment", "Atom(X)", "", 0.0, energy, attachment, "Attachment"),
    ]


def rotational_energy(j: int) -> float:
    """Return the rigid-rotor energy ``B J (J + 1)`` of level ``J`` in eV."""
    return ROTATIONAL_CONSTANT * j * (j + 1)


def rotational_weight(j: int) -> float:
    """Return ``(2J + 1) g_ns`` with the nuclear-spin weights of H2 (1 para, 3 ortho)."""
    return (2 * j + 1) * (3.0 if j % 2 else 1.0)


def rotational_populations(temperature: float = GAS_TEMPERATURE) -> np.ndarray:
    """Return the Boltzmann populations of ``J = 0..MAX_J`` at ``temperature`` (K)."""
    kt = BOLTZMANN_EV * temperature
    weights = np.array(
        [
            rotational_weight(j) * math.exp(-rotational_energy(j) / kt)
            for j in range(MAX_J + 1)
        ]
    )
    return weights / weights.sum()


def vibrational_populations(temperature: float = VIBRATIONAL_TEMPERATURE) -> tuple:
    """Return the populations of ``v = 0`` and ``v = 1`` (two levels) at ``temperature``."""
    ratio = math.exp(-VIBRATIONAL_ENERGY / (BOLTZMANN_EV * temperature))
    return 1.0 / (1.0 + ratio), ratio / (1.0 + ratio)


def rotational_state(j: int) -> str:
    """Return LoKI-B's name of rotational level ``J`` of ``v = 0``."""
    return f"Mol(X,v=0,J={j})"


def molecule_processes() -> list[Process]:
    """Return the H2-like model gas."""
    energy = np.concatenate([[0.0], np.geomspace(1e-3, 1e3, 300)])
    elastic = (
        6e-20 * (1.0 + 2.0 * energy / (1.0 + energy)) / (1.0 + (energy / 15.0) ** 1.2)
    )
    processes = [Process("elastic", "Mol(X)", "", 0.0, energy, elastic, "Elastic")]
    for j in range(MAX_J - 1):
        loss = rotational_energy(j + 2) - rotational_energy(j)
        strength = GERJUOY_STEIN * (j + 2) * (j + 1) / ((2 * j + 3) * (2 * j + 1))

        def sigma(e: np.ndarray, loss=loss, strength=strength) -> np.ndarray:
            return strength * np.sqrt(np.clip(1.0 - loss / e, 0.0, None))

        processes.append(
            Process(
                "excitation",
                rotational_state(j),
                rotational_state(j + 2),
                loss,
                *_above(loss, sigma),
                "Rotational",
                rotational_weight(j + 2) / rotational_weight(j),
            )
        )

    def vibrational(e: np.ndarray) -> np.ndarray:
        x = (e - VIBRATIONAL_ENERGY) / 2.5
        return 5e-21 * x * np.exp(1.0 - x)

    processes += [
        Process(
            "excitation",
            "Mol(X,v=0)",
            "Mol(X,v=1)",
            VIBRATIONAL_ENERGY,
            *_above(VIBRATIONAL_ENERGY, vibrational),
            "Vibrational",
            1.0,
        ),
        Process(
            "excitation",
            "Mol(X)",
            "Mol(B)",
            11.25,
            *_above(11.25, _bethe(11.25, 4e-21)),
            "Excitation",
        ),
        Process(
            "ionization",
            "Mol(X)",
            "",
            15.5,
            *_above(15.5, _bethe(15.5, 1e-20)),
            "Ionization",
        ),
    ]
    return processes


#: The cases: (name, gas, E/N in Td, growth model, ionization operator,
#: maximum energy in eV, number of cells), with the energy step a power of two.
CASES = [
    *[
        (
            f"atom_{growth}_{sharing}_{field:g}",
            "Atom",
            field,
            growth,
            sharing,
            top,
            4096,
        )
        for growth in ("temporal", "spatial")
        for sharing in ("equalSharing", "usingSDCS")
        for field, top in ((100.0, 128.0), (300.0, 256.0), (1000.0, 512.0))
    ],
    *[
        (f"mol_{field:g}", "Mol", field, "temporal", "equalSharing", top, cells)
        for field, top, cells in (
            (1.0, 4.0, 4096),
            (10.0, 16.0, 4096),
            (30.0, 32.0, 4096),
            (100.0, 64.0, 4096),
        )
    ],
]


def plasmacoll_model(gas: str, sharing: str = "equalSharing") -> tuple:
    """Return the processes, the mixture and the states of a model gas for plasmacoll.

    LoKI-B's ``Atom(X)`` and ``Mol(X)`` are the gases ``Atom`` and ``Mol``; its
    vibrational and rotational levels are states of ``Mol``, the rotational
    populations relative to ``v = 0``. Processes are named by their LoKI-B
    reaction.

    Args:
        gas: ``"Atom"`` or ``"Mol"``.
        sharing: LoKI-B's ionization operator, ``"equalSharing"`` or
            ``"usingSDCS"`` (plasmacoll's ``"opal"``).
    """
    from plasmacoll import CollisionProcess, CrossSection, NeutralBackground

    processes = []
    for process in atom_processes() if gas == "Atom" else molecule_processes():
        options: dict = {}
        if process.kind == "ionization":
            options = {
                "products": {"electron": "e", "ion": f"{gas}+"},
                "energy_sharing": "equal" if sharing == "equalSharing" else "opal",
                "sharing_energy": OPB_WIDTH if sharing == "usingSDCS" else None,
            }
        elif process.kind == "attachment":
            options = {"products": {"negative_ion": f"{gas}-"}}
        if process.weight_ratio is not None:
            options["upper_state"] = process.upper
            options["statistical_weight_ratio"] = process.weight_ratio
        processes.append(
            CollisionProcess(
                process.kind,
                gas if process.lower == f"{gas}(X)" else process.lower,
                CrossSection(process.energy, process.sigma, threshold=process.loss),
                name=process.reaction,
                **options,
            )
        )
    states = None
    if gas == "Mol":
        v0, v1 = vibrational_populations()
        states = {"Mol(X,v=0)": ("Mol", v0), "Mol(X,v=1)": ("Mol", v1)}
        for j, population in enumerate(rotational_populations()):
            states[rotational_state(j)] = ("Mol", v0 * float(population))
    mass = ATOM_MASS if gas == "Atom" else MOL_MASS
    mixture = [NeutralBackground(gas, 1e22, BOLTZMANN_EV * GAS_TEMPERATURE, mass)]
    return processes, mixture, states


# --- LoKI-B input and output ---------------------------------------------------------


def lokib_lxcat(processes: list[Process]) -> str:
    """Return the cross sections in the LXCat format LoKI-B reads."""
    blocks = []
    for process in processes:
        keyword = {"elastic": "ELASTIC", "attachment": "ATTACHMENT"}.get(
            process.kind, process.kind.upper()
        )
        gas = process.lower.split("(")[0]
        mass = ATOM_MASS if gas == "Atom" else MOL_MASS
        parameter = (
            f"m/M = {9.1093835e-31 / mass:.6e}"
            if process.kind == "elastic"
            else f"E = {process.loss!r} eV"
        )
        table = "\n".join(
            f"{e:.15e}\t{s:.15e}"
            for e, s in zip(process.energy, process.sigma, strict=True)
        )
        blocks.append(
            f"{keyword}\n{process.lower}\n"
            f"PARAM.:  {parameter}\n"
            f"COMMENT: [{process.reaction}, {process.label}]\n"
            f"-----------------------------\n{table}\n-----------------------------\n"
        )
    return "\n".join(blocks)


def lokib_setup(case: tuple, cells: int) -> str:
    """Return the LoKI-B setup file of ``case`` on ``cells`` cells."""
    name, gas, field, growth, sharing, top, _ = case
    if gas == "Atom":
        states = """    population:
      - Atom(X) = 1.0"""
    else:
        rotational = rotational_populations()
        v0, v1 = vibrational_populations()
        lines = [
            "    energy:",
            "      - Mol(X,v=0) = 0.0",
            f"      - Mol(X,v=1) = {VIBRATIONAL_ENERGY!r}",
        ]
        lines += [
            f"      - {rotational_state(j)} = {rotational_energy(j)!r}"
            for j in range(MAX_J + 1)
        ]
        lines += [
            "    statisticalWeight:",
            "      - Mol(X,v=0) = 1.0",
            "      - Mol(X,v=1) = 1.0",
        ]
        lines += [
            f"      - {rotational_state(j)} = {rotational_weight(j)!r}"
            for j in range(MAX_J + 1)
        ]
        lines += [
            "    population:",
            "      - Mol(X) = 1.0",
            f"      - Mol(X,v=0) = {v0!r}",
            f"      - Mol(X,v=1) = {v1!r}",
        ]
        lines += [
            f"      - {rotational_state(j)} = {float(p)!r}"
            for j, p in enumerate(rotational)
        ]
        states = "\n".join(lines)
    return f"""workingConditions:
  reducedField: {field!r}
  electronTemperature: 1.0
  excitationFrequency: 0
  gasPressure: 133.32
  gasTemperature: {GAS_TEMPERATURE!r}
  electronDensity: 1e10
  chamberLength: 1.0
  chamberRadius: 1.0
electronKinetics:
  isOn: true
  eedfType: boltzmann
  ionizationOperatorType: {sharing}
  growthModelType: {growth}
  includeEECollisions: false
  LXCatFiles:
    - {gas}.txt
  gasProperties:
    mass: masses.txt
    fraction:
      - {gas} = 1
    OPBParameter: opb.txt
  stateProperties:
{states}
  numerics:
    energyGrid:
      maxEnergy: {top!r}
      cellNumber: {cells}
    maxPowerBalanceRelError: 1e-6
    nonLinearRoutines:
      algorithm: mixingDirectSolutions
      mixingParameter: 0.7
      maxEedfRelError: 1e-10
gui:
  isOn: false
output:
  isOn: true
  folder: {name}_{cells}
  dataFiles:
    - eedf
    - swarmParameters
    - rateCoefficients
    - powerBalance
"""


def _read_values(path: Path) -> dict[str, float]:
    """Return the ``name = value (unit)`` lines of a LoKI-B output file."""
    values = {}
    for line in path.read_text().splitlines():
        match = re.match(r"\s*(.+?) = ([-+0-9.eE]+)", line)
        if match:
            values.setdefault(match.group(1).strip(), float(match.group(2)))
    return values


def _read_rates(path: Path) -> dict[str, list[float]]:
    """Return the inelastic and superelastic rate coefficients by reaction."""
    rates = {}
    for line in path.read_text().splitlines()[1:]:
        parts = line.split(maxsplit=2)
        if len(parts) == 3 and parts[2].startswith("e +"):
            reaction = parts[2].rsplit(",", 1)[0]
            rates[reaction] = [float(parts[0]), float(parts[1])]
    return rates


def _results(folder: Path) -> dict:
    """Return what the tests compare from the output of one LoKI-B run."""
    (run,) = folder.glob("reducedField_*")
    swarm = _read_values(run / "swarmParameters.txt")
    power = _read_values(run / "powerBalance.txt")
    eedf = np.loadtxt(run / "eedf.txt", skiprows=1)
    keep = np.unique(np.linspace(0, eedf.shape[0] - 1, 64).astype(int))
    return {
        "mean_energy": swarm["Mean energy"],
        "mobility": swarm["Reduced mobility coefficient"],
        "diffusion": swarm["Reduced diffusion coefficient"],
        "townsend": swarm["Reduced Townsend coefficient"],
        "attachment": swarm["Reduced attachment coefficient"],
        "relative_power_balance": power["Relative Power Balance"] / 100.0,
        "rates": _read_rates(run / "rateCoefficients.txt"),
        "eedf_energy": eedf[keep, 0].tolist(),
        "eedf": eedf[keep, 1].tolist(),
    }


def _one_line(match: re.Match) -> str:
    """Return a JSON array of numbers on one line."""
    return "[" + ", ".join(item.strip() for item in match.group(1).split(",")) + "]"


def main() -> None:
    """Run LoKI-B on every case and write the reference file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--loki", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--version", default="", help="LoKI-B++ version or commit")
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "Atom.txt").write_text(lokib_lxcat(atom_processes()))
    (workdir / "Mol.txt").write_text(lokib_lxcat(molecule_processes()))
    (workdir / "masses.txt").write_text(f"Atom {ATOM_MASS!r}\nMol {MOL_MASS!r}\n")
    (workdir / "opb.txt").write_text(f"Atom {OPB_WIDTH!r}\nMol {OPB_WIDTH!r}\n")
    cases = {}
    for case in CASES:
        name, cells = case[0], case[-1]
        runs = {}
        for num in (cells, cells // 2):
            setup = workdir / f"{name}_{num}.in"
            setup.write_text(lokib_setup(case, num))
            subprocess.run(
                [str(args.loki.resolve()), setup.name],
                cwd=workdir,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            runs[num] = _results(workdir / "output" / f"{name}_{num}")
        fine, coarse = runs[cells], runs[cells // 2]
        cases[name] = {
            "gas": case[1],
            "reduced_field": case[2],
            "growth": case[3],
            "ionization_operator": case[4],
            "max_energy": case[5],
            "cells": cells,
            **fine,
            "coarse": {
                key: coarse[key]
                for key in (
                    "mean_energy",
                    "mobility",
                    "diffusion",
                    "townsend",
                    "attachment",
                    "rates",
                )
            },
        }
    REFERENCE.parent.mkdir(exist_ok=True)
    text = json.dumps(
        {
            "provenance": (
                "LoKI-B++ (https://github.com/LoKI-Suite/LoKI-B-cpp) "
                f"{args.version}, run by _lokib_models.py on its synthetic "
                "model gases; 'coarse' is the same case on half the cells"
            ),
            "gas_temperature": GAS_TEMPERATURE,
            "vibrational_temperature": VIBRATIONAL_TEMPERATURE,
            "cases": cases,
        },
        indent=1,
    )
    # one line per array of numbers
    text = re.sub(r"\[\s+([^\[\]{}\"]*?)\s+\]", _one_line, text)
    REFERENCE.write_text(text + "\n")


if __name__ == "__main__":
    main()
