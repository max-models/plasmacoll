"""Proton collisions with atomic and molecular hydrogen in a negative hydrogen ion source.

The six processes that dominate the transport of protons in the driver and
expansion region of an RF-driven negative hydrogen ion source (the test-particle
model of Wünderlich et al.):

========  ==============================  =================  =============
key       reaction                        plasmacoll kind        energy frame
========  ==============================  =================  =============
mt_h      ``H+ + H  -> H+ + H``           ``elastic``        centre of mass
mt_h2     ``H+ + H2 -> H+ + H2``          ``elastic``        lab
cx_h      ``H+ + H  -> H  + H+``          ``backscatter``    lab
cx_h2     ``H+ + H2 -> H  + H2+``         ``charge_transfer``  lab
rot_h2    ``H+ + H2(J=0) -> H+ + H2(J=2)``  ``excitation``   lab
vib_h2    ``H+ + H2(v=0) -> H+ + H2(v>0)``  ``excitation``   lab
========  ==============================  =================  =============

"Lab" is the proton's energy in the rest frame of the target, as the cross
sections of the model are tabulated; momentum transfer with atoms uses the
centre-of-mass energy.

This module supplies the process definitions, not the cross sections: the
tables are inputs (the model's sources are Krstic and Schultz for H+ + H2 and
Janev et al. for H+ + H; see also AMJUEL/HYDHEL), because numbers are not
reproduced here that cannot be checked against their source. The charge
exchange of protons with H2 depends strongly on the vibrational level of the
molecule; the model uses a population-weighted mean for a vibrational
temperature of 5000 K. Build it with::

    levels = [CrossSection.from_table(f"cx_h2_v{v}.dat") for v in range(15)]
    dist = VibrationalDistribution.h2_ground_state_from_kelvin(5000.0)
    cx_h2 = CrossSection.vibrational_mixture(levels, dist.populations)
"""

from __future__ import annotations

from collections.abc import Mapping

from plasmacoll.cross_sections import CrossSection
from plasmacoll.process import CollisionProcess

__all__ = [
    "H2_ROTATIONAL_EXCITATION_EV",
    "H2_VIBRATIONAL_EXCITATION_EV",
    "PROTON_HYDROGEN_PROCESS_KEYS",
    "proton_hydrogen_processes",
]

#: The keys of the cross sections :func:`proton_hydrogen_processes` needs.
PROTON_HYDROGEN_PROCESS_KEYS = (
    "mt_h",
    "mt_h2",
    "cx_h",
    "cx_h2",
    "rot_h2",
    "vib_h2",
)

#: Energy of the H2 rotational transition J = 0 -> 2 in eV (6 B, B = 60.85 /cm).
H2_ROTATIONAL_EXCITATION_EV = 0.0453
#: Energy of the H2 vibrational transition v = 0 -> 1 in eV.
H2_VIBRATIONAL_EXCITATION_EV = 0.516


def proton_hydrogen_processes(
    cross_sections: Mapping[str, CrossSection],
    *,
    atom: str = "H",
    molecule: str = "H2",
    h2_ion: str | None = None,
    rotational_energy: float = H2_ROTATIONAL_EXCITATION_EV,
    vibrational_energy: float = H2_VIBRATIONAL_EXCITATION_EV,
) -> list[CollisionProcess]:
    """Return the six proton processes with the given cross sections.

    The excitation thresholds ``rotational_energy`` and ``vibrational_energy``
    override whatever threshold the tables carry; the other tables are used
    with threshold zero.

    Args:
        cross_sections: Tables (energies in eV, cross sections in m^2) for
            every key in :data:`PROTON_HYDROGEN_PROCESS_KEYS`.
        atom: The name of the atomic hydrogen background.
        molecule: The name of the molecular hydrogen background.
        h2_ion: A species name such as ``"H2+"``: the charge exchange with H2
            then creates that ion with the molecule's velocity; otherwise it
            only removes the proton.
        rotational_energy: Threshold of the rotational excitation in eV.
        vibrational_energy: Threshold of the vibrational excitation in eV.

    Raises:
        KeyError: If a table is missing.
    """
    missing = set(PROTON_HYDROGEN_PROCESS_KEYS) - set(cross_sections)
    if missing:
        raise KeyError(f"Missing cross sections for {sorted(missing)}")

    def table(key: str, threshold: float = 0.0) -> CrossSection:
        return cross_sections[key].with_threshold(threshold)

    return [
        CollisionProcess(
            kind="elastic",
            background=atom,
            cross_section=table("mt_h"),
            name="mt_h",
            energy_frame="center_of_mass",
        ),
        CollisionProcess(
            kind="elastic",
            background=molecule,
            cross_section=table("mt_h2"),
            name="mt_h2",
            energy_frame="lab",
        ),
        CollisionProcess(
            kind="backscatter",
            background=atom,
            cross_section=table("cx_h"),
            name="cx_h",
            energy_frame="lab",
        ),
        CollisionProcess(
            kind="charge_transfer",
            background=molecule,
            cross_section=table("cx_h2"),
            name="cx_h2",
            energy_frame="lab",
            products={} if h2_ion is None else {"ion": h2_ion},
        ),
        CollisionProcess(
            kind="excitation",
            background=molecule,
            cross_section=table("rot_h2", rotational_energy),
            name="rot_h2",
            energy_frame="lab",
        ),
        CollisionProcess(
            kind="excitation",
            background=molecule,
            cross_section=table("vib_h2", vibrational_energy),
            name="vib_h2",
            energy_frame="lab",
        ),
    ]
