"""Astropy quantities in the setup: converted once, stored as plain numbers."""

from __future__ import annotations

import subprocess
import sys

import astropy.units as u
import cunumpy as xp
import numpy as np
import pytest

from plasmacoll import (
    BeamGasReactions,
    ChargeStateTransitions,
    CollisionProcess,
    CrossSection,
    DensityProfile,
    MonteCarloCollisions,
    NeutralBackground,
    ParticleArrays,
    RateTable,
    VibrationalDistribution,
    ZeroDReactor,
    argon_ion_cross_sections,
    charge_state_fractions,
    hydrogen_beam_on_h2,
    rate_coefficient,
    solve_eedf,
    swarm_table,
)
from plasmacoll.atomic_hydrogen import (
    vriens_smeets_excitation_rate,
    vriens_smeets_ionization_rate,
)
from plasmacoll.caesium import langevin_rate_coefficient
from plasmacoll.constants import (
    ATOMIC_MASS,
    BOLTZMANN,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    PROTON_MASS,
)
from plasmacoll.hydrogen_elastic import ELASTIC_FITS
from plasmacoll.janev import JANEV_FITS, dissociative_attachment_levels
from plasmacoll.metals import LotzSubshell, lotz_table
from plasmacoll.secondaries import RUDD_H2, RuddIonization, RuddShell
from plasmacoll.vibrational import estimate_electronegativity

KELVIN_PER_EV = ELEMENTARY_CHARGE / BOLTZMANN
ARGON = 39.948 * ATOMIC_MASS


def _gas(**kwargs) -> NeutralBackground:
    params = {"density": 1e21, "temperature": 0.025, "mass": ARGON} | kwargs
    return NeutralBackground("Ar", **params)


@pytest.mark.parametrize(
    "density", [1e15 * u.cm**-3, "1e15 cm-3", 1e21, "1e21"], ids=str
)
def test_background_density(density) -> None:
    assert _gas(density=density).density == pytest.approx(1e21)
    assert type(_gas(density=density).density) is float


@pytest.mark.parametrize(
    "temperature",
    [300 * u.K, "300 K", 300 / KELVIN_PER_EV * u.eV, 300 * BOLTZMANN * u.J],
    ids=str,
)
def test_background_temperature_in_kelvin_ev_or_joule(temperature) -> None:
    gas = _gas(temperature=temperature)
    assert gas.temperature == pytest.approx(300 / KELVIN_PER_EV)


def test_background_mass_and_drift() -> None:
    gas = _gas(mass="39.948 u", drift=[1, 0, -2] * u.km / u.s)
    assert gas.mass == pytest.approx(ARGON)
    assert gas.drift == pytest.approx((1000.0, 0.0, -2000.0))
    assert _gas(drift=["1 km/s", 0, 0]).drift == pytest.approx((1000.0, 0.0, 0.0))
    # bare numbers keep their meaning
    assert _gas().density == 1e21
    assert _gas().temperature == 0.025
    assert _gas().drift == (0.0, 0.0, 0.0)


def test_background_rejects_wrong_dimensions() -> None:
    with pytest.raises(ValueError, match=r"density must be in units of m\^-3"):
        _gas(density=1 * u.m)
    with pytest.raises(ValueError, match="temperature must be in units of eV"):
        _gas(temperature="1 m")
    with pytest.raises(ValueError, match="mass must be in units of kg"):
        _gas(mass=1 * u.s)
    with pytest.raises(ValueError, match="drift must be in units of m/s"):
        _gas(drift=[1, 0, 0] * u.m)


def test_density_profile_positions() -> None:
    profile = DensityProfile(0, [0, 5, 10] * u.cm, [1.0, 0.5, 0.0])
    assert xp.to_numpy(profile.positions) == pytest.approx([0.0, 0.05, 0.1])
    assert xp.to_numpy(DensityProfile(1, "[0, 2] mm", [1, 1]).positions) == (
        pytest.approx([0.0, 0.002])
    )
    plain = DensityProfile(0, xp.asarray([0.0, 1.0]), [1.0, 2.0])
    assert xp.to_numpy(plain.positions) == pytest.approx([0.0, 1.0])
    with pytest.raises(ValueError, match="positions must be in units of m"):
        DensityProfile(0, [0, 1] * u.s, [1.0, 1.0])


def test_cross_section_table() -> None:
    table = CrossSection(
        energy=[0.01, 1, 10] * u.keV, sigma="[1, 2, 3] cm2", threshold="15 eV"
    )
    assert xp.to_numpy(table.energy) == pytest.approx([10.0, 1000.0, 10000.0])
    assert xp.to_numpy(table.sigma) == pytest.approx([1e-4, 2e-4, 3e-4])
    assert table.threshold == pytest.approx(15.0)
    assert table.with_threshold(2 * u.eV).threshold == pytest.approx(2.0)
    plain = CrossSection([0.0, 1.0], [1e-20, 1e-20])
    assert plain.threshold == 0.0
    with pytest.raises(ValueError, match="sigma must be in units of m\\^2"):
        CrossSection([0.0, 1.0] * u.eV, [1, 1] * u.m)
    with pytest.raises(ValueError, match="threshold must be in units of eV"):
        CrossSection([0.0, 1.0], [0.0, 1.0], threshold=1 * u.m)


def test_cross_section_builders() -> None:
    constant = CrossSection.constant(
        "1e-16 cm2", threshold="11.5 eV", max_energy=1 * u.MeV
    )
    assert xp.to_numpy(constant.sigma) == pytest.approx([1e-20, 1e-20])
    assert constant.threshold == pytest.approx(11.5)
    assert constant.max_energy == pytest.approx(1e6)
    maxwell = CrossSection.maxwell_molecule(
        "1e-9 cm3/s", 4 * u.u, min_energy="1 meV", max_energy=1 * u.keV, num=11
    )
    reference = CrossSection.maxwell_molecule(
        1e-15, 4 * ATOMIC_MASS, min_energy=1e-3, max_energy=1e3, num=11
    )
    assert xp.to_numpy(maxwell.sigma) == pytest.approx(xp.to_numpy(reference.sigma))
    tabulated = CrossSection.from_function(
        lambda e: 1e-20 * np.ones_like(e), [1, 2, 3] * u.keV, threshold="1 keV"
    )
    assert xp.to_numpy(tabulated.energy) == pytest.approx([1e3, 2e3, 3e3])
    assert tabulated.threshold == pytest.approx(1e3)
    with pytest.raises(ValueError, match="mass must be in units of kg"):
        CrossSection.maxwell_molecule(1e-15, 1 * u.m)


def test_cross_section_from_table_units(tmp_path) -> None:
    path = tmp_path / "table.txt"
    path.write_text("1 1e-16\n2 2e-16\n")
    table = CrossSection.from_table(path, energy_unit="1 keV", sigma_unit=1 * u.cm**2)
    assert xp.to_numpy(table.energy) == pytest.approx([1e3, 2e3])
    assert xp.to_numpy(table.sigma) == pytest.approx([1e-20, 2e-20])
    factors = CrossSection.from_table(path, energy_unit=1e3, sigma_unit=1e-4)
    assert xp.to_numpy(factors.sigma) == pytest.approx([1e-20, 2e-20])


def test_collision_process_energies() -> None:
    sigma = CrossSection.constant(1e-20, threshold=15.8)
    process = CollisionProcess(
        "ionization",
        "Ar",
        sigma,
        energy_loss="15.8 eV",
        products={"electron": "e", "ion": "Ar+"},
        energy_sharing="opal",
        sharing_energy=10 * u.eV,
    )
    assert process.energy_loss == pytest.approx(15.8)
    assert process.sharing_energy == pytest.approx(10.0)
    assert CollisionProcess("elastic", "Ar", sigma).energy_loss is None
    with pytest.raises(ValueError, match="energy_loss must be in units of eV"):
        CollisionProcess("excitation", "Ar", sigma, energy_loss=1 * u.m)


def test_monte_carlo_collisions_masses() -> None:
    sigma = CrossSection.constant(1e-19)
    mcc = MonteCarloCollisions(
        {"e": ELECTRON_MASS * u.kg, "Ar+": "39.948 u"},
        [_gas(temperature=300 * u.K)],
        {"e": [CollisionProcess("elastic", "Ar", sigma)]},
        seed=1,
    )
    assert mcc.species_masses["e"] == pytest.approx(ELECTRON_MASS)
    assert mcc.species_masses["Ar+"] == pytest.approx(ARGON)
    with pytest.raises(ValueError, match="must be in units of kg"):
        MonteCarloCollisions({"e": 1 * u.m}, [], {})


def test_maxwellian_markers() -> None:
    markers = ParticleArrays.maxwellian(
        2000, "39.948 u", 3000 * u.K, drift=[1, 0, 0] * u.km / u.s, seed=3
    )
    assert markers.temperature(ARGON) == pytest.approx(3000 / KELVIN_PER_EV, rel=0.1)
    mean = xp.to_numpy(xp.mean(markers.velocities, axis=0))
    assert mean[0] == pytest.approx(1000.0, rel=0.1)
    with pytest.raises(ValueError, match="temperature must be in units of eV"):
        ParticleArrays.maxwellian(10, ARGON, 1 * u.m)


def test_zero_d_reactor_setup() -> None:
    sigma = CrossSection.constant(1e-19)
    mcc = MonteCarloCollisions(
        {"e": ELECTRON_MASS},
        [_gas()],
        {"e": [CollisionProcess("elastic", "Ar", sigma, energy_frame="lab")]},
        seed=2,
    )
    electrons = ParticleArrays.maxwellian(100, ELECTRON_MASS, "2 eV", seed=2)
    reactor = ZeroDReactor(
        mcc,
        {"e": electrons},
        volume=1 * u.cm**3,
        electric_field="[0, 0, 1] kV/m",
        charges={"e": -1},
    )
    assert reactor.volume == pytest.approx(1e-6)
    assert reactor.electric_field == pytest.approx((0.0, 0.0, 1000.0))
    history = reactor.run("0.1 ns", 2)
    assert history.time[-1] == pytest.approx(2e-10)
    with pytest.raises(ValueError, match="volume must be in units of m\\^3"):
        ZeroDReactor(mcc, {}, volume=1 * u.m)


def test_beam_gas_reactions() -> None:
    reactions = hydrogen_beam_on_h2(max_energy="100 keV", num=21)
    assert reactions.reactions[0].cross_section.max_energy == pytest.approx(1e5)
    again = BeamGasReactions(reactions.reactions, sharing_energy="15.43 eV")
    assert again.sharing_energy == pytest.approx(15.43)
    at_keV = reactions.transitions(10 * u.keV).cross_sections
    assert at_keV == reactions.transitions(1e4).cross_sections
    assert reactions.ionization("10 keV") == reactions.ionization(1e4)
    with pytest.raises(ValueError, match="max_energy must be in units of eV"):
        hydrogen_beam_on_h2(max_energy=1 * u.m)


def test_charge_states() -> None:
    transitions = ChargeStateTransitions(
        {"H2": {("-", "0"): "1e-16 cm2", ("0", "+"): 2e-20 * u.m**2}}
    )
    assert transitions.cross_sections["H2"] == pytest.approx(
        {("-", "0"): 1e-20, ("0", "+"): 2e-20}
    )
    fractions = charge_state_fractions(
        transitions, {"H2": [0, 1e16] * u.cm**-2}, initial=[1.0, 0.0, 0.0]
    )
    plain = charge_state_fractions(transitions, {"H2": [0, 1e20]}, [1.0, 0.0, 0.0])
    assert fractions == pytest.approx(plain)
    with pytest.raises(ValueError, match="must be in units of m\\^2"):
        ChargeStateTransitions({"H2": {("-", "0"): 1 * u.m}})


def test_table_builders_max_energy() -> None:
    assert argon_ion_cross_sections(min_energy="0.1 meV", max_energy=10 * u.keV)[
        "isotropic"
    ].max_energy == pytest.approx(1e4)
    shell = LotzSubshell("7.7 eV", 1, a=4.5e-14 * u.cm**2 * u.eV**2)
    assert shell.binding_energy == pytest.approx(7.7)
    assert shell.a == pytest.approx(4.5e-18)
    assert lotz_table([shell], max_energy="1 keV").max_energy == pytest.approx(1e3)
    fit = JANEV_FITS["hminus_h2_detachment"]
    assert fit.cross_section("1 keV").max_energy == pytest.approx(1e3)
    assert dissociative_attachment_levels(20 * u.eV)[0].max_energy == pytest.approx(
        20.0
    )
    elastic = ELASTIC_FITS["H2"].cross_section(max_energy="100 eV", num=11)
    assert elastic.max_energy == pytest.approx(100.0)
    with pytest.raises(ValueError, match="binding_energy must be in units of eV"):
        LotzSubshell(1 * u.m, 1)


def test_rates_take_temperatures_in_kelvin() -> None:
    sigma = CrossSection.constant(1e-19)
    in_kelvin = rate_coefficient(sigma, np.array([1, 2]) * KELVIN_PER_EV * u.K, "1 u")
    assert in_kelvin == pytest.approx(rate_coefficient(sigma, [1.0, 2.0], ATOMIC_MASS))
    table = RateTable.from_cross_section(
        sigma, ATOMIC_MASS * u.kg, min_temperature="100 K", max_temperature=1 * u.keV
    )
    assert table.temperature[0] == pytest.approx(100 / KELVIN_PER_EV)
    arrhenius = RateTable.arrhenius(1e-14, c="1 eV", max_temperature="10 eV")
    assert arrhenius.temperature[-1] == pytest.approx(10.0)
    direct = RateTable([1, 10] * u.eV, "[1e-8, 2e-8] cm3/s")
    assert direct.rate == pytest.approx([1e-14, 2e-14])
    assert vriens_smeets_ionization_rate(2, "3 eV") == pytest.approx(
        vriens_smeets_ionization_rate(2, 3.0)
    )
    assert vriens_smeets_excitation_rate(1, 2, [2, 4] * u.eV) == pytest.approx(
        vriens_smeets_excitation_rate(1, 2, np.array([2.0, 4.0]))
    )
    with pytest.raises(ValueError, match="temperature must be in units of eV"):
        rate_coefficient(sigma, 1 * u.m, ATOMIC_MASS)


def test_boltzmann_reduced_field() -> None:
    sigma = CrossSection.constant(1e-19)
    processes = [CollisionProcess("elastic", "Ar", sigma, energy_frame="lab")]
    mixture = [_gas()]
    td = 1e-21 * u.V * u.m**2
    solution = solve_eedf(processes, mixture, 10 * td, "300 K", num_cells=20)
    reference = solve_eedf(processes, mixture, 10.0, 300 / KELVIN_PER_EV, num_cells=20)
    assert solution.reduced_field == pytest.approx(10.0)
    assert solution.mean_energy == pytest.approx(reference.mean_energy)
    table = swarm_table(processes, mixture, [10, 20] * td, num_cells=20)
    assert table.reduced_field == pytest.approx([10.0, 20.0])
    with pytest.raises(ValueError, match="reduced_field must be in units of"):
        solve_eedf(processes, mixture, 1 * u.V)


def test_vibrational_and_secondaries() -> None:
    from_kelvin = VibrationalDistribution.h2_ground_state_from_kelvin(5000 * u.K)
    from_quantity = VibrationalDistribution.h2_ground_state(5000 * u.K)
    assert from_quantity.temperature_ev == pytest.approx(from_kelvin.temperature_ev)
    assert VibrationalDistribution.h2_ground_state_from_kelvin(
        "0.5 eV"
    ).temperature_ev == pytest.approx(0.5)
    ratio = estimate_electronegativity(from_kelvin, "1e13 cm-3", 1e19 * u.m**-3)
    assert ratio == pytest.approx(estimate_electronegativity(from_kelvin, 1e19, 1e19))
    deuteron = RUDD_H2.with_projectile("2.014 u")
    assert deuteron.projectile_mass == pytest.approx(2.014 * ATOMIC_MASS)
    assert RuddIonization(RUDD_H2.shells).projectile_mass == PROTON_MASS
    shell = RUDD_H2.shells[0]
    assert RuddShell(**(vars(shell) | {"binding_energy": "15.43 eV"})) == shell
    table = RUDD_H2.cross_section("[10, 100] keV")
    assert xp.to_numpy(table.energy) == pytest.approx([1e4, 1e5])
    with pytest.raises(ValueError, match="projectile_mass must be in units of kg"):
        RUDD_H2.with_projectile(1 * u.eV)
    assert langevin_rate_coefficient("0.8 Angstrom3", 1 * u.u) == pytest.approx(
        langevin_rate_coefficient(0.8e-30, ATOMIC_MASS)
    )


def test_numbers_do_not_import_astropy() -> None:
    """A setup of plain numbers runs without importing astropy."""
    code = (
        "import sys\n"
        "import plasmacoll\n"
        "from plasmacoll.constants import ELECTRON_MASS\n"
        "gas = plasmacoll.NeutralBackground('Ar', 1e21, 0.025, 6.6e-26)\n"
        "process = plasmacoll.CollisionProcess(\n"
        "    'elastic', 'Ar', plasmacoll.CrossSection.constant(1e-19))\n"
        "plasmacoll.MonteCarloCollisions({'e': ELECTRON_MASS}, [gas], {'e': [process]})\n"
        "assert 'astropy' not in sys.modules, 'astropy was imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_builders_pass_quantities_on() -> None:
    from plasmacoll.caesium import electron_background
    from plasmacoll.hydrogen import (
        PROTON_HYDROGEN_PROCESS_KEYS,
        proton_hydrogen_processes,
    )

    tables = dict.fromkeys(PROTON_HYDROGEN_PROCESS_KEYS, CrossSection.constant(1e-20))
    processes = proton_hydrogen_processes(tables, vibrational_energy="516 meV")
    vibrational = next(p for p in processes if p.name == "vib_h2")
    assert vibrational.cross_section.threshold == pytest.approx(0.516)
    electrons = electron_background("1e11 cm-3", 2 * u.eV)
    assert electrons.density == pytest.approx(1e17)
    assert electrons.temperature == pytest.approx(2.0)
