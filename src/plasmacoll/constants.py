"""Physical constants in SI units (CODATA 2018).

The elementary charge and the Boltzmann constant are exact since the 2019
redefinition of the SI; the masses are the CODATA 2018 recommended values.
"""

from __future__ import annotations

#: Elementary charge in C, also the number of J in one eV (exact).
ELEMENTARY_CHARGE = 1.602176634e-19
#: Boltzmann constant in J/K (exact).
BOLTZMANN = 1.380649e-23
#: Electron mass in kg.
ELECTRON_MASS = 9.1093837015e-31
#: Proton mass in kg.
PROTON_MASS = 1.67262192369e-27
#: Atomic mass constant (1 u, 1/12 of the mass of carbon-12) in kg.
ATOMIC_MASS = 1.66053906660e-27


def kelvin_to_ev(temperature: float) -> float:
    """Return the thermal energy ``k T`` of ``temperature`` (K) in eV."""
    return BOLTZMANN * temperature / ELEMENTARY_CHARGE


def ev_to_kelvin(energy: float) -> float:
    """Return the temperature (K) whose thermal energy ``k T`` is ``energy`` (eV)."""
    return energy * ELEMENTARY_CHARGE / BOLTZMANN
