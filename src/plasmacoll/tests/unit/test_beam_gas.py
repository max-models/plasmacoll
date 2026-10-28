"""The beam-gas reactions of a hydrogen beam on H2: data, fractions, markers."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest

from plasmacoll import (
    BeamGasReaction,
    BeamGasReactions,
    CollisionProcess,
    CrossSection,
    DensityProfile,
    MonteCarloCollisions,
    NeutralBackground,
    ParticleArrays,
    charge_state_fractions,
    hydrogen_beam_on_h2,
)
from plasmacoll.constants import ATOMIC_MASS, ELECTRON_MASS, ELEMENTARY_CHARGE
from plasmacoll.janev import JANEV_FITS

from ._helpers import beam

M_H = 1.00782503207 * ATOMIC_MASS
MASSES = {
    "H-": M_H + ELECTRON_MASS,
    "H": M_H,
    "H+": M_H - ELECTRON_MASS,
    "e": ELECTRON_MASS,
    "H2+": 2.01565 * ATOMIC_MASS - ELECTRON_MASS,
}
M_H2 = 2.01565 * ATOMIC_MASS
SPECIES = {"-": "H-", "0": "H", "+": "H+"}
NAMES = ("H-", "H", "H+")
CM2 = 1.0e-4


@pytest.mark.parametrize(
    ("key", "energy", "expected", "tolerance"),
    [
        # read off Figure 34 of the report (H- + H2 detachment)
        ("hminus_h2_detachment", 30.0, 4.4e-16, 0.15),
        ("hminus_h2_detachment", 2.0e3, 1.15e-15, 0.15),
        ("hminus_h2_detachment", 1.0e5, 3.5e-16, 0.15),
        # read off Figure 32 (H+ + H2(v=0) charge exchange)
        ("proton_h2_charge_exchange", 30.0, 1.35e-17, 0.15),
        ("proton_h2_charge_exchange", 5.0e3, 1.2e-15, 0.15),
        ("proton_h2_charge_exchange", 1.0e5, 2.7e-17, 0.15),
    ],
)
def test_the_fast_projectile_fits_follow_the_report(key, energy, expected, tolerance):
    fit = JANEV_FITS[key]
    assert fit.energy_frame == "lab"
    assert float(fit(energy)) == pytest.approx(expected * CM2, rel=tolerance)
    assert float(fit(0.5 * fit.threshold)) == 0.0


def test_the_detachment_maximum_is_where_the_report_says() -> None:
    """A broad maximum of about 1e-15 cm^2 in 2-8 keV; the energy is in eV."""
    energy = np.geomspace(10.0, 1.0e6, 2001)
    sigma = JANEV_FITS["hminus_h2_detachment"](energy)
    assert 1.0e3 < energy[np.argmax(sigma)] < 8.0e3
    assert sigma.max() == pytest.approx(1.2e-15 * CM2, rel=0.1)


def test_the_reactions_and_their_sources() -> None:
    reactions = hydrogen_beam_on_h2()
    assert reactions.targets == ("H2",)
    assert set(reactions.placeholders) == {
        "-->+:H2", "0->+:H2", "-:ionization:H2", "0:ionization:H2",
    }  # fmt: skip
    sigma = reactions.transitions(1.0e5).cross_sections["H2"]
    assert sigma[("-", "0")] == pytest.approx(
        float(JANEV_FITS["hminus_h2_detachment"](1.0e5)), rel=1e-3
    )
    assert sigma[("-", "+")] == pytest.approx(sigma[("-", "0")] * 5.0e-22 / 1.2e-20)
    # Rudd's proton ionization of H2 peaks at about 2e-16 cm^2 near 50-70 keV
    ionization = reactions.ionization(6.0e4)["H2"]
    assert ionization["+"] == pytest.approx(2.0e-16 * CM2, rel=0.1)
    assert ionization["0"] == pytest.approx(ionization["+"], rel=1e-3)


def test_deuterium_takes_the_hydrogen_data_at_the_same_speed() -> None:
    hydrogen, deuterium = hydrogen_beam_on_h2("H"), hydrogen_beam_on_h2("D")
    ratio = 2.01410177812 / 1.00782503207
    for energy in (1.0e4, 2.0e5, 1.0e6):
        h = hydrogen.transitions(energy / ratio).cross_sections["H2"]
        d = deuterium.transitions(energy).cross_sections["H2"]
        assert d[("-", "0")] == pytest.approx(h[("-", "0")], rel=0.02)
        h_iz = hydrogen.ionization(energy / ratio)["H2"]["+"]
        assert deuterium.ionization(energy)["H2"]["+"] == pytest.approx(h_iz, rel=0.02)
    source = deuterium.reactions[0].source
    assert "same speed" in source
    with pytest.raises(ValueError, match="isotope"):
        hydrogen_beam_on_h2("T")
    with pytest.raises(ValueError, match="max_energy"):
        hydrogen_beam_on_h2(max_energy=0.0)


def test_the_processes_follow_the_reactions() -> None:
    reactions = hydrogen_beam_on_h2()
    processes = reactions.processes(
        SPECIES, {"H2": "gas"}, electron="e", ions={"H2": "H2+"}
    )
    by_name = {p.name: p for ps in processes.values() for p in ps}
    assert {p.kind for p in processes["H+"]} == {"stripping", "beam_ionization"}
    capture = by_name["+->0:H2"]
    assert capture.charge_change == -1 and capture.products == {
        "projectile": "H",
        "ion": "H2+",
    }
    assert by_name["-->+:H2"].charge_change == 2
    ionization = by_name["0:ionization:H2"]
    assert ionization.products == {"ion": "H2+", "electron": "e"}
    assert ionization.background == "gas" and ionization.energy_frame == "lab"
    # without an ion the ionization is left out; states without a species too
    only = reactions.processes({"-": "H-", "0": "H"})
    names = {p.name for ps in only.values() for p in ps}
    assert names == {"-->0:H2"}


def test_the_reactions_are_checked() -> None:
    sigma = CrossSection.constant(1e-20)
    with pytest.raises(ValueError, match="unknown state"):
        BeamGasReactions([BeamGasReaction("-", "2", "gas", sigma, "")])
    with pytest.raises(ValueError, match="changes the state"):
        BeamGasReactions([BeamGasReaction("-", "-", "gas", sigma, "")])
    with pytest.raises(ValueError, match="once"):
        BeamGasReactions([BeamGasReaction("-", None, "gas", sigma, "")] * 2)
    with pytest.raises(ValueError, match="no charge"):
        BeamGasReactions([], states=("-", "x"))
    with pytest.raises(ValueError, match="sharing_energy"):
        CollisionProcess(
            "beam_ionization", "gas", sigma, products={"ion": "I", "electron": "e"}
        )


def test_the_ionization_keeps_the_projectile_and_leaves_slow_products() -> None:
    energy = 1.0e6
    reactions = BeamGasReactions(
        [BeamGasReaction("+", None, "gas", CrossSection.constant(1e-19), "")]
    )
    mcc = MonteCarloCollisions(
        MASSES,
        [NeutralBackground("gas", 1.0e20, 0.025, M_H2)],
        reactions.processes(SPECIES, electron="e", ions={"gas": "H2+"}),
        seed=5,
    )
    num = 20000
    velocities = beam(num, energy, MASSES["H+"])
    weights = xp.asarray(np.linspace(0.5, 2.0, num))
    result = mcc.collide("H+", xp.zeros((num, 1)), velocities, weights, dt=2e-9)
    events = result.diagnostics.counts["+:ionization:gas"]
    assert events > 1000
    assert not bool(xp.any(result.removed))
    assert xp.to_numpy(result.velocities) == pytest.approx(xp.to_numpy(velocities))
    (ion,) = result.created["H2+"]
    (electron,) = result.created["e"]
    assert int(ion.weights.shape[0]) == int(electron.weights.shape[0]) == events
    assert xp.to_numpy(ion.weights) == pytest.approx(xp.to_numpy(electron.weights))
    # the ion is a thermal molecule: far slower than the beam
    ion_speed = np.linalg.norm(xp.to_numpy(ion.velocities), axis=1)
    assert ion_speed.max() < 1.0e4
    # the electron's energy is below the binary-encounter maximum 2 m_e v^2 and
    # mostly of the order of the binding energy
    speed = float(beam(1, energy, MASSES["H+"])[0, 0])
    e_energy = (
        0.5
        * ELECTRON_MASS
        * np.sum(xp.to_numpy(electron.velocities) ** 2, axis=1)
        / ELEMENTARY_CHARGE
    )
    assert e_energy.max() <= 2.0 * ELECTRON_MASS * speed**2 / ELEMENTARY_CHARGE * 1.01
    assert 5.0 < np.median(e_energy) < 30.0


def _run_through_gas(num: int, energy: float, seed: int):
    """Push an H- beam through a triangular gas profile; return the state shares."""
    reactions = hydrogen_beam_on_h2(max_energy=2.0 * energy)
    length, peak = 1.0, 5.0e19
    profile = DensityProfile(0, [0.0, 0.5 * length, length], [0.0, 1.0, 0.0])
    mcc = MonteCarloCollisions(
        MASSES,
        [NeutralBackground("gas", peak, 0.0, M_H2, profile=profile)],
        reactions.processes(SPECIES, {"H2": "gas"}, electron="e", ions={"H2": "H2+"}),
        seed=seed,
    )
    species = {name: ParticleArrays.empty(ndim=1) for name in MASSES}
    species["H-"].add(xp.zeros((num, 1)), beam(num, energy, MASSES["H-"]), xp.ones(num))
    speed = float(beam(1, energy, MASSES["H-"])[0, 0])
    steps = 240
    dt = length / speed / steps
    checkpoints = {60, 120, 180, 240}
    shares = {}
    for k in range(1, steps + 1):
        for name in NAMES:  # half a push, collide at the midpoint, half a push
            species[name].positions = species[name].positions + 0.5 * dt * speed
        mcc.collide_species(species, dt, incident=NAMES)
        for name in NAMES:
            species[name].positions = species[name].positions + 0.5 * dt * speed
        if k in checkpoints:
            shares[k * dt * speed] = (
                np.array([species[n].total_weight for n in NAMES]) / num
            )
    return reactions, profile, peak, species, shares


def test_the_populations_follow_the_fractions_on_a_gas_profile() -> None:
    """Markers through a 1D gas profile against ``charge_state_fractions``."""
    num, energy = 20000, 1.0e5
    reactions, profile, peak, species, shares = _run_through_gas(num, energy, seed=7)
    transitions = reactions.transitions(energy)
    z = np.array(sorted(shares))
    density = peak * xp.to_numpy(profile(xp.asarray(z[:, None])))
    # the triangle's line density, exact for the piecewise-linear profile
    fine = np.linspace(0.0, z[-1], 4001)
    n_fine = peak * np.interp(fine, [0.0, 0.5, 1.0], [0.0, 1.0, 0.0])
    column = np.concatenate(
        [[0.0], np.cumsum(0.5 * (n_fine[1:] + n_fine[:-1]) * np.diff(fine))]
    )
    line_density = np.interp(np.concatenate([[0.0], z]), fine, column)
    expected = charge_state_fractions(transitions, {"H2": line_density})[1:]
    assert density[-1] == pytest.approx(0.0, abs=1e-6 * peak)
    for row, position in enumerate(z):
        spread = np.sqrt(expected[row] * (1.0 - expected[row]) / num)
        assert np.all(np.abs(shares[position] - expected[row]) < 5.0 * spread + 1e-12)
    # the beam has evolved: a good part of the H- stripped
    assert expected[-1][0] < 0.6 and expected[-1][1] > 0.3


def test_the_gas_ions_follow_the_ionization_rate_of_the_column() -> None:
    """The H2+ made along the path: ionization by every state plus capture."""
    num, energy = 20000, 1.0e5
    reactions, _, peak, species, _ = _run_through_gas(num, energy, seed=11)
    fine = np.linspace(0.0, 1.0, 4001)
    n_fine = peak * np.interp(fine, [0.0, 0.5, 1.0], [0.0, 1.0, 0.0])
    column = np.concatenate(
        [[0.0], np.cumsum(0.5 * (n_fine[1:] + n_fine[:-1]) * np.diff(fine))]
    )
    transitions = reactions.transitions(energy)
    fractions = charge_state_fractions(transitions, {"H2": column})
    ionization = reactions.ionization(energy)["H2"]
    sigma = transitions.cross_sections["H2"]
    rate = (
        sum(ionization[s] * fractions[:, k] for k, s in enumerate(("-", "0", "+")))
        + sigma[("+", "0")] * fractions[:, 2]
    )
    integrand = rate * n_fine
    expected = num * float(
        np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(fine))
    )
    made = species["H2+"].total_weight
    assert expected > 1000
    assert made == pytest.approx(expected, abs=5.0 * np.sqrt(expected))
    # the electrons: one per ionization and per single stripping, two per double
    assert species["e"].total_weight > made
