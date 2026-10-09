r"""Case 1 of the Turner et al. (2013) capacitively coupled helium discharge benchmark.

M. M. Turner et al., "Simulation benchmarks for low-pressure plasmas: capacitive
discharges", Phys. Plasmas 20, 013507 (2013). A 1D3V electrostatic
particle-in-cell simulation of electrons and He+ between two plane electrodes,
one driven at 13.56 MHz, with plasmacoll as the collision step.

The physical parameters below are those of case 1. Check them against Table I
of the paper before comparing results, and compare the time-averaged ion
density with the paper's figures.

The benchmark prescribes its cross sections. plasmacoll does not ship data, so
download the sets the paper specifies from LXCat and pass them:

    uv run python examples/turner_ccp/turner_case1.py \\
        --electron-lxcat He_electrons.txt --ion-lxcat He_ions.txt

The electron file needs an ELASTIC or EFFECTIVE block, the excitation blocks
and an IONIZATION block; the ion file an isotropic (ELASTIC) and a
backscattering (charge exchange) block, given with ``--ion-isotropic`` and
``--ion-backscatter`` (their ``PROCESS`` comments or kinds).

``--synthetic`` replaces the data with made-up constant cross sections. That
only checks that the simulation runs; its results mean nothing. The full run
(1280 RF cycles at 400 steps per cycle) takes hours in Python; ``--rf-cycles``
shortens it.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

import plasmacoll
from plasmacoll.constants import ELECTRON_MASS, ELEMENTARY_CHARGE

VACUUM_PERMITTIVITY = 8.8541878128e-12

# Case 1 of the benchmark (Table I of Turner et al. 2013).
GAP = 0.067  # m
FREQUENCY = 13.56e6  # Hz
GAS_DENSITY = 9.64e20  # m^-3
GAS_TEMPERATURE = 300.0  # K
VOLTAGE = 450.0  # V
INITIAL_DENSITY = 2.56e14  # m^-3
ELECTRON_TEMPERATURE = 30000.0  # K, initial
ION_TEMPERATURE = 300.0  # K, initial
ION_MASS = 6.67e-27  # kg
STEPS_PER_CYCLE = 400
NUM_CELLS = 128
PARTICLES_PER_CELL = 512
RF_CYCLES = 1280
AVERAGING_CYCLES = 32


def electron_processes(path: Path | None) -> list[plasmacoll.CollisionProcess]:
    """Return the electron-helium processes from an LXCat file (or synthetic ones)."""
    if path is None:
        return [
            plasmacoll.CollisionProcess(
                "elastic",
                "He",
                plasmacoll.CrossSection.constant(6e-20),
                energy_frame="lab",
            ),
            plasmacoll.CollisionProcess(
                "excitation",
                "He",
                plasmacoll.CrossSection.constant(1e-21, threshold=19.82),
                energy_frame="lab",
            ),
            plasmacoll.CollisionProcess(
                "ionization",
                "He",
                plasmacoll.CrossSection.constant(3e-21, threshold=24.59),
                energy_frame="lab",
                products={"electron": "e", "ion": "He+"},
            ),
        ]
    blocks = plasmacoll.read_lxcat(path)
    processes = [
        plasmacoll.CollisionProcess(
            "elastic", "He", plasmacoll.elastic_from_lxcat(blocks), energy_frame="lab"
        )
    ]
    for index, block in enumerate(blocks):
        if block.kind == "EXCITATION":
            processes.append(
                plasmacoll.CollisionProcess(
                    "excitation",
                    "He",
                    block.cross_section,
                    name=f"excitation:{index}",
                    energy_frame="lab",
                )
            )
        elif block.kind == "IONIZATION":
            processes.append(
                plasmacoll.CollisionProcess(
                    "ionization",
                    "He",
                    block.cross_section,
                    name=f"ionization:{index}",
                    energy_frame="lab",
                    products={"electron": "e", "ion": "He+"},
                )
            )
    return processes


def ion_processes(
    path: Path | None, isotropic: str, backscatter: str
) -> list[plasmacoll.CollisionProcess]:
    """Return the He+ - He processes from an LXCat file (or synthetic ones)."""
    if path is None:
        return [
            plasmacoll.CollisionProcess(
                "elastic",
                "He",
                plasmacoll.CrossSection.constant(3e-19),
                energy_frame="lab",
            ),
            plasmacoll.CollisionProcess(
                "backscatter",
                "He",
                plasmacoll.CrossSection.constant(3e-19),
                energy_frame="lab",
            ),
        ]
    blocks = plasmacoll.read_lxcat(path)
    return [
        plasmacoll.CollisionProcess(
            "elastic",
            "He",
            plasmacoll.find_lxcat_process(blocks, isotropic).cross_section,
            energy_frame="lab",
        ),
        plasmacoll.CollisionProcess(
            "backscatter",
            "He",
            plasmacoll.find_lxcat_process(blocks, backscatter).cross_section,
            energy_frame="lab",
        ),
    ]


class Field:
    """Poisson's equation on a uniform 1D grid with Dirichlet electrodes."""

    def __init__(self, num_cells: int, gap: float) -> None:
        """Invert the Laplacian of the inner nodes once."""
        self.num_nodes = num_cells + 1
        self.dx = gap / num_cells
        inner = num_cells - 1
        laplacian = (
            np.diag(np.full(inner, -2.0))
            + np.diag(np.ones(inner - 1), 1)
            + np.diag(np.ones(inner - 1), -1)
        ) / self.dx**2
        self.inverse = np.linalg.inv(laplacian)

    def solve(self, rho: np.ndarray, left: float, right: float) -> np.ndarray:
        """Return the potential at the nodes for charge density ``rho`` (C/m^3)."""
        rhs = -rho[1:-1] / VACUUM_PERMITTIVITY
        rhs[0] -= left / self.dx**2
        rhs[-1] -= right / self.dx**2
        potential = np.empty(self.num_nodes)
        potential[0], potential[-1] = left, right
        potential[1:-1] = self.inverse @ rhs
        return potential

    def electric_field(self, potential: np.ndarray) -> np.ndarray:
        """Return ``-d phi / dx`` at the nodes (one-sided at the electrodes)."""
        return -np.gradient(potential, self.dx)


def deposit(
    positions: np.ndarray, weights: np.ndarray, dx: float, num_nodes: int
) -> np.ndarray:
    """Return the cloud-in-cell density at the nodes (per m^3 of a unit area)."""
    cell = np.clip((positions / dx).astype(int), 0, num_nodes - 2)
    fraction = positions / dx - cell
    density = np.bincount(cell, weights * (1 - fraction), minlength=num_nodes)
    density += np.bincount(cell + 1, weights * fraction, minlength=num_nodes)
    density /= dx
    density[0] *= 2.0  # half cells at the electrodes
    density[-1] *= 2.0
    return density


def interpolate(field: np.ndarray, positions: np.ndarray, dx: float) -> np.ndarray:
    """Return the cloud-in-cell interpolation of node values to the positions."""
    cell = np.clip((positions / dx).astype(int), 0, field.shape[0] - 2)
    fraction = positions / dx - cell
    return field[cell] * (1 - fraction) + field[cell + 1] * fraction


def main() -> None:
    """Run the benchmark and write the time-averaged ion density."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--electron-lxcat", type=Path)
    parser.add_argument("--ion-lxcat", type=Path)
    parser.add_argument("--ion-isotropic", default="ELASTIC")
    parser.add_argument("--ion-backscatter", default="EFFECTIVE")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--rf-cycles", type=int, default=RF_CYCLES)
    parser.add_argument("--averaging-cycles", type=int, default=AVERAGING_CYCLES)
    parser.add_argument("--particles-per-cell", type=int, default=PARTICLES_PER_CELL)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("turner_case1_density.txt"))
    args = parser.parse_args()
    if not args.synthetic and (args.electron_lxcat is None or args.ion_lxcat is None):
        parser.error("give --electron-lxcat and --ion-lxcat, or --synthetic")

    dt = 1.0 / (STEPS_PER_CYCLE * FREQUENCY)
    field = Field(NUM_CELLS, GAP)
    num = NUM_CELLS * args.particles_per_cell
    weight = INITIAL_DENSITY * GAP / num  # particles per m^2 of electrode
    rng = np.random.default_rng(args.seed)

    def initial(mass: float, temperature: float) -> plasmacoll.ParticleArrays:
        markers = plasmacoll.ParticleArrays.maxwellian(
            num, mass, temperature, weight=weight, rng=rng
        )
        markers.positions = rng.uniform(0.0, GAP, (num, 1))
        return markers

    species = {
        "e": initial(ELECTRON_MASS, ELECTRON_TEMPERATURE),
        "He+": initial(ION_MASS, ION_TEMPERATURE),
    }
    charges = {"e": -ELEMENTARY_CHARGE, "He+": ELEMENTARY_CHARGE}
    masses = {"e": ELECTRON_MASS, "He+": ION_MASS}
    synthetic = args.synthetic
    mcc = plasmacoll.MonteCarloCollisions(
        species_masses=masses,
        backgrounds=[
            plasmacoll.NeutralBackground("He", GAS_DENSITY, GAS_TEMPERATURE, ION_MASS)
        ],
        processes={
            "e": electron_processes(None if synthetic else args.electron_lxcat),
            "He+": ion_processes(
                None if synthetic else args.ion_lxcat,
                args.ion_isotropic,
                args.ion_backscatter,
            ),
        },
        seed=args.seed,
    )

    total_steps = args.rf_cycles * STEPS_PER_CYCLE
    averaging_steps = min(args.averaging_cycles, args.rf_cycles) * STEPS_PER_CYCLE
    average = np.zeros(field.num_nodes)
    start = time.perf_counter()
    for step in range(total_steps):
        rho = sum(
            charges[name]
            * deposit(
                markers.positions[:, 0], markers.weights, field.dx, field.num_nodes
            )
            for name, markers in species.items()
        )
        voltage = VOLTAGE * math.sin(2.0 * math.pi * FREQUENCY * step * dt)
        efield = field.electric_field(field.solve(rho, voltage, 0.0))
        for name, markers in species.items():
            local = interpolate(efield, markers.positions[:, 0], field.dx)
            markers.velocities[:, 0] += charges[name] / masses[name] * local * dt
            markers.positions[:, 0] += markers.velocities[:, 0] * dt
            outside = (markers.positions[:, 0] < 0.0) | (markers.positions[:, 0] > GAP)
            if np.any(outside):
                markers.remove(outside)
        mcc.collide_species(species, dt)
        if step >= total_steps - averaging_steps:
            ions = species["He+"]
            average += deposit(
                ions.positions[:, 0], ions.weights, field.dx, field.num_nodes
            )
        if (step + 1) % (10 * STEPS_PER_CYCLE) == 0:
            elapsed = time.perf_counter() - start
            print(
                f"cycle {(step + 1) // STEPS_PER_CYCLE:5d}  "
                f"electrons {len(species['e']):7d}  ions {len(species['He+']):7d}  "
                f"{elapsed:7.1f} s"
            )
    average /= max(averaging_steps, 1)
    x = np.linspace(0.0, GAP, field.num_nodes)
    np.savetxt(
        args.output,
        np.column_stack([x, average]),
        header="x (m)    time-averaged ion density (m^-3)",
    )
    mean_energy = species["e"].mean_energy_ev(ELECTRON_MASS)
    print(
        f"peak ion density {average.max():.4g} m^-3, "
        f"mean electron energy {mean_energy:.3g} eV; wrote {args.output}"
    )


if __name__ == "__main__":
    main()
