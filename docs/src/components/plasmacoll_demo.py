"""The live demo of the docs: the energy distribution of electrons in hydrogen.

It runs in the browser (Pyodide). The e + H2 cross sections are the published fits
the package ships: the elastic momentum transfer of Park et al. (2022) and the
vibrational and electronic excitation and ionization of Janev, Reiter and Samm
(2003), all from H2(v=0) (no superelastic collisions). `plasmacoll.solve_eedf`
solves the two-term Boltzmann equation for them; `run` takes the values of the
sliders and returns the figure and the swarm parameters.
"""

import functools

import numpy as np

import plasmacoll
from plasmacoll.constants import ATOMIC_MASS, kelvin_to_ev
from plasmacoll.hydrogen_elastic import ELASTIC_FITS
from plasmacoll.janev import JANEV_FITS

#: e + H2(v=0) -> e + H2(v=1, 2), and the strongest electronic excitations (b 3Sigma_u
#: dissociates the molecule).
VIBRATIONAL = ("e_h2_vibrational_1", "e_h2_vibrational_2")
ELECTRONIC = ("e_h2_triplet_b", "e_h2_singlet_B", "e_h2_singlet_C")
#: The ionization threshold of H2 in eV.
IONIZATION_THRESHOLD = JANEV_FITS["e_h2_ionization"].threshold


@functools.cache
def processes(vibrational=True):
    """Return the e + H2 processes, with or without the vibrational excitation."""
    table = {"max_energy": 1e3, "num": 120}
    result = [
        plasmacoll.CollisionProcess(
            "elastic", "H2", ELASTIC_FITS["H2"].cross_section(**table), name="elastic"
        )
    ]
    for key in (VIBRATIONAL if vibrational else ()) + ELECTRONIC:
        cross_section = JANEV_FITS[key].cross_section(**table)
        result.append(
            plasmacoll.CollisionProcess("excitation", "H2", cross_section, name=key)
        )
    result.append(
        plasmacoll.CollisionProcess(
            "ionization",
            "H2",
            JANEV_FITS["e_h2_ionization"].cross_section(**table),
            name="ionization",
            products={"electron": "e", "ion": "H2+"},
        )
    )
    return tuple(result)


def run(reduced_field=30.0, gas_temperature=300.0, vibrational="on"):
    """Solve the two-term Boltzmann equation for electrons in H2.

    Args:
        reduced_field: The reduced field E/N in Td (1 Td = 1e-21 V m^2).
        gas_temperature: The temperature of the gas in K.
        vibrational: ``"on"`` with the vibrational excitation of H2, ``"off"``
            without it.

    Returns:
        The Plotly figure of the EEPF and of the Maxwellian with the same mean
        energy (``figure``), and the swarm parameters (``numbers``).
    """
    gas = plasmacoll.NeutralBackground(
        "H2", 1e22, kelvin_to_ev(gas_temperature), 2.016 * ATOMIC_MASS
    )
    solution = plasmacoll.solve_eedf(
        processes(vibrational == "on"), [gas], reduced_field, num_cells=150
    )
    # the Maxwellian with kT = 2/3 <eps>, normalised as F0: int eps^1/2 F0 deps = 1
    kt = 2.0 * solution.mean_energy / 3.0
    maxwellian = 2.0 / np.sqrt(np.pi) * kt**-1.5 * np.exp(-solution.energy / kt)
    shown = solution.eepf > 1e-12 * solution.eepf.max()
    energy, eepf = solution.energy[shown], solution.eepf[shown]
    power = solution.power
    lost = power["elastic"] + power["inelastic"] + power["growth"]
    figure = {
        "data": [
            {"x": energy, "y": eepf, "mode": "lines", "name": "two-term Boltzmann"},
            {
                "x": energy,
                "y": maxwellian[shown],
                "mode": "lines",
                "line": {"dash": "dash"},
                "name": "Maxwellian, same mean energy",
            },
            {
                "x": [IONIZATION_THRESHOLD, IONIZATION_THRESHOLD],
                "y": [float(eepf.min()), 2.0 * float(eepf.max())],
                "mode": "lines",
                "line": {"dash": "dot", "width": 1, "color": "gray"},
                "name": "ionization threshold",
            },
        ],
        "layout": {
            "xaxis": {
                "title": {"text": "electron energy (eV)"},
                "range": [0.0, float(energy[-1])],
            },
            "yaxis": {"title": {"text": "EEPF (eV<sup>-3/2</sup>)"}, "type": "log"},
            "legend": {"x": 1, "xanchor": "right", "y": 1},
        },
    }
    return {
        "figure": figure,
        "numbers": {
            "Mean energy (eV)": solution.mean_energy,
            "Drift velocity (m/s)": solution.drift_velocity,
            "Power lost in inelastic collisions (%)": 100 * power["inelastic"] / lost,
            "Ionization rate coefficient (m³/s)": solution.rate_coefficients[
                "ionization"
            ],
            "Townsend coefficient α/N (m²)": solution.townsend_coefficient,
        },
    }
