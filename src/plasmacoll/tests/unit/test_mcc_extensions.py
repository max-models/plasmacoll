"""Anisotropic scattering, ionization energy sharing, ion impact and the new process kinds."""

from __future__ import annotations

import math
import warnings

import cunumpy as xp
import numpy as np
import pytest

from plasmacoll import (
    CollisionProcess,
    MonteCarloCollisions,
    NeutralBackground,
    scattering_cosine,
)
from plasmacoll.constants import ELEMENTARY_CHARGE
from plasmacoll.mcc import _rotate, _unit

from ._helpers import HEAVY, M_E, M_HE, beam, constant, energy_ev, operator


def _host(array) -> np.ndarray:
    return np.asarray(xp.to_numpy(array))


def _ones(num: int):
    return xp.ones(num)


def _origin(num: int):
    return xp.zeros((num, 1))


def _all_collide(mcc: MonteCarloCollisions, species: str, velocities):
    """Collide with dt long enough that every marker collides."""
    num = int(velocities.shape[0])
    return mcc.collide(species, _origin(num), velocities, _ones(num), dt=1.0)


# --------------------------------------------------------------------- #
# Scattering models
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("model", ["vahedi_surendra", "okhrimovskyy"])
def test_scattering_is_isotropic_at_low_energy(model: str) -> None:
    uniform = xp.linspace(0.0, 1.0, 101)
    cosine = scattering_cosine(model, xp.zeros(101), uniform)
    np.testing.assert_allclose(_host(cosine), _host(1.0 - 2.0 * uniform), atol=1e-12)


@pytest.mark.parametrize("model", ["vahedi_surendra", "okhrimovskyy"])
def test_scattering_is_forward_peaked_at_high_energy(model: str) -> None:
    uniform = (np.arange(200_000) + 0.5) / 200_000
    cosine = _host(
        scattering_cosine(model, xp.full(200_000, 1.0e3), xp.asarray(uniform))
    )
    assert np.all((cosine >= -1.0) & (cosine <= 1.0))
    assert cosine.mean() > 0.5
    # The inverse CDF is monotonic in the random number.
    assert np.all(np.diff(cosine) <= 1e-12)


def test_isotropic_scattering_cosine_and_unknown_model() -> None:
    uniform = xp.asarray([0.0, 0.25, 1.0])
    np.testing.assert_allclose(
        _host(scattering_cosine("isotropic", xp.ones(3), uniform)), [1.0, 0.5, -1.0]
    )
    with pytest.raises(ValueError, match="Unknown scattering model"):
        scattering_cosine("rutherford", xp.ones(3), uniform)


def test_rotation_gives_the_requested_angle_for_any_direction() -> None:
    directions = xp.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.6, 0.0, 0.8]])
    cos_chi = xp.asarray([0.3, -0.5, 0.9])
    rotated = _rotate(directions, cos_chi, xp.asarray([0.1, 2.0, 4.0]))
    np.testing.assert_allclose(_host(xp.linalg.norm(rotated, axis=1)), 1.0)
    np.testing.assert_allclose(
        _host(xp.sum(rotated * directions, axis=1)), _host(cos_chi)
    )


def test_unit_vectors_of_zero_vectors_point_along_z() -> None:
    vectors = xp.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 4.0]])
    units = _unit(vectors, xp.linalg.norm(vectors, axis=1))
    np.testing.assert_allclose(_host(units), [[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]])


@pytest.mark.parametrize("model", ["vahedi_surendra", "okhrimovskyy"])
def test_anisotropic_elastic_scattering_keeps_energy_and_goes_forward(
    model: str,
) -> None:
    process = CollisionProcess("elastic", "gas", constant(1e-18), scattering=model)
    mcc = operator([process], {"e": M_E})
    initial = beam(20_000, 1.0e3, M_E)
    result = _all_collide(mcc, "e", initial)
    np.testing.assert_allclose(
        _host(energy_ev(result.velocities, M_E)), 1.0e3, rtol=1e-6
    )
    cosine = _host(result.velocities[:, 0]) / math.sqrt(
        2.0 * 1.0e3 * ELEMENTARY_CHARGE / M_E
    )
    expected = _host(
        scattering_cosine(
            model,
            xp.full(400_000, 1.0e3),
            xp.asarray(np.random.default_rng(0).random(400_000)),
        )
    ).mean()
    assert cosine.mean() == pytest.approx(expected, abs=0.01)


def test_anisotropic_excitation_scatters_with_the_incident_energy() -> None:
    process = CollisionProcess(
        "excitation", "gas", constant(1e-18, 10.0), scattering="vahedi_surendra"
    )
    mcc = operator([process], {"e": M_E})
    result = _all_collide(mcc, "e", beam(5000, 500.0, M_E))
    np.testing.assert_allclose(
        _host(energy_ev(result.velocities, M_E)), 490.0, rtol=1e-6
    )
    assert _host(result.velocities[:, 0]).mean() > 0.0


# --------------------------------------------------------------------- #
# Ionization
# --------------------------------------------------------------------- #
def _ionization(**kwargs) -> CollisionProcess:
    return CollisionProcess(
        "ionization",
        "gas",
        constant(1e-18, 15.0),
        products={"electron": "e", "ion": "ion"},
        **kwargs,
    )


def test_opal_sharing_conserves_energy_and_follows_its_distribution() -> None:
    width = 10.0
    mcc = operator(
        [_ionization(energy_sharing="opal", sharing_energy=width)],
        {"e": M_E, "ion": M_HE},
    )
    energy = 215.0
    result = _all_collide(mcc, "e", beam(100_000, energy, M_E))
    primary = _host(energy_ev(result.velocities, M_E))
    secondary = _host(energy_ev(result.created["e"][0].velocities, M_E))
    residual = energy - 15.0
    np.testing.assert_allclose(primary + secondary, residual, rtol=1e-9)
    assert secondary.max() <= residual / 2.0 + 1e-9
    half = residual / 2.0
    mean = width * math.log(1.0 + (half / width) ** 2) / (2.0 * math.atan(half / width))
    assert secondary.mean() == pytest.approx(mean, rel=0.02)


def test_opal_sharing_without_residual_energy_gives_nothing_to_the_secondary() -> None:
    process = CollisionProcess(
        "ionization",
        "gas",
        constant(1e-18),
        energy_loss=1.0e3,
        products={"electron": "e", "ion": "ion"},
        energy_sharing="opal",
        sharing_energy=10.0,
    )
    mcc = operator([process], {"e": M_E, "ion": M_HE})
    result = _all_collide(mcc, "e", beam(100, 10.0, M_E))
    np.testing.assert_allclose(_host(result.velocities), 0.0)
    np.testing.assert_allclose(_host(result.created["e"][0].velocities), 0.0)


def test_anisotropic_ionization_scatters_both_electrons() -> None:
    mcc = operator([_ionization(scattering="okhrimovskyy")], {"e": M_E, "ion": M_HE})
    result = _all_collide(mcc, "e", beam(5000, 2015.0, M_E))
    primary = _host(energy_ev(result.velocities, M_E))
    np.testing.assert_allclose(primary, 1000.0, rtol=1e-9)
    assert _host(result.velocities[:, 0]).mean() > 0.0
    assert _host(result.created["e"][0].velocities[:, 0]).mean() > 0.0


def test_ion_impact_ionization_conserves_energy_and_momentum() -> None:
    """``He+ + He -> He+ + He+ + e`` with the target at rest."""
    loss = 24.6
    process = CollisionProcess(
        "ionization",
        "gas",
        constant(1e-18, loss),
        products={"electron": "e", "ion": "He+"},
        energy_frame="lab",
    )
    mcc = operator(
        [process], {"He+": M_HE, "e": M_E}, neutral_mass=M_HE, incident="He+"
    )
    energy = 1.0e3
    initial = beam(2000, energy, M_HE)
    result = _all_collide(mcc, "He+", initial)
    incident = _host(result.velocities)
    target = _host(result.created["He+"][0].velocities)
    electron = _host(result.created["e"][0].velocities)
    kinetic = (
        0.5 * M_HE * (incident**2).sum(1)
        + 0.5 * M_HE * (target**2).sum(1)
        + 0.5 * M_E * (electron**2).sum(1)
    ) / ELEMENTARY_CHARGE
    # Per marker up to the electron's cross term m_e V_cm . v_e, which averages out.
    np.testing.assert_allclose(kinetic + loss, energy, rtol=math.sqrt(M_E / M_HE))
    assert (kinetic + loss).mean() == pytest.approx(energy, rel=1e-3)
    momentum = M_HE * (incident + target) + M_E * electron
    initial_momentum = M_HE * _host(initial)
    np.testing.assert_allclose(
        momentum[:, 0], initial_momentum[:, 0], rtol=math.sqrt(M_E / M_HE)
    )
    # Equal sharing: the electron has half the centre-of-mass energy left over.
    electron_cm = electron - (incident + target) / 2.0
    np.testing.assert_allclose(
        0.5 * M_E * (electron_cm**2).sum(1) / ELEMENTARY_CHARGE,
        0.5 * (energy / 2.0 - loss),
        rtol=1e-9,
    )


# --------------------------------------------------------------------- #
# Dissociative processes and optional products
# --------------------------------------------------------------------- #
def test_dissociation_loses_energy_and_creates_the_fragment() -> None:
    process = CollisionProcess(
        "dissociation", "gas", constant(1e-18, 8.8), products={"fragment": "H"}
    )
    mcc = operator([process], {"e": M_E, "H": M_HE})
    result = _all_collide(mcc, "e", beam(1000, 20.0, M_E))
    np.testing.assert_allclose(
        _host(energy_ev(result.velocities, M_E)), 11.2, rtol=1e-6
    )
    assert int(result.created["H"][0].velocities.shape[0]) == 1000
    assert result.diagnostics.counts == {"dissociation:gas": 1000}


def test_dissociation_without_fragment_species_tracks_nothing() -> None:
    process = CollisionProcess("dissociation", "gas", constant(1e-18, 8.8))
    result = _all_collide(operator([process], {"e": M_E}), "e", beam(100, 20.0, M_E))
    assert result.created == {}


def test_dissociative_attachment_creates_ion_and_fragment() -> None:
    process = CollisionProcess(
        "dissociative_attachment",
        "gas",
        constant(1e-18, 3.0),
        products={"negative_ion": "H-", "fragment": "H"},
    )
    assert process.loss == 0.0
    mcc = operator([process], {"e": M_E, "H-": M_HE, "H": M_HE})
    result = _all_collide(mcc, "e", beam(500, 5.0, M_E))
    assert bool(xp.all(result.removed))
    assert set(result.created) == {"H-", "H"}


def test_dissociative_ionization_creates_electron_ion_and_fragment() -> None:
    process = CollisionProcess(
        "dissociative_ionization",
        "gas",
        constant(1e-18, 18.0),
        products={"electron": "e", "ion": "H+", "fragment": "H"},
    )
    assert process.is_ionization
    mcc = operator([process], {"e": M_E, "H+": M_HE, "H": M_HE})
    result = _all_collide(mcc, "e", beam(500, 38.0, M_E))
    np.testing.assert_allclose(
        _host(energy_ev(result.velocities, M_E)), 10.0, rtol=1e-9
    )
    assert set(result.created) == {"e", "H+", "H"}


def test_charge_exchange_can_track_the_fast_neutral() -> None:
    initial = beam(800, 100.0, M_HE)
    backscatter = CollisionProcess(
        "backscatter", "gas", constant(1e-18), products={"neutral": "He"}
    )
    mcc = operator(
        [backscatter], {"He+": M_HE, "He": M_HE}, neutral_mass=M_HE, incident="He+"
    )
    result = _all_collide(mcc, "He+", initial)
    np.testing.assert_allclose(_host(result.velocities), 0.0, atol=1e-9)
    np.testing.assert_allclose(
        _host(result.created["He"][0].velocities), _host(initial)
    )

    transfer = CollisionProcess(
        "charge_transfer",
        "gas",
        constant(1e-18),
        products={"ion": "H2+", "neutral": "H"},
    )
    mcc = operator([transfer], {"H+": M_HE, "H2+": M_HE, "H": M_HE}, incident="H+")
    result = _all_collide(mcc, "H+", initial)
    assert bool(xp.all(result.removed))
    np.testing.assert_allclose(_host(result.created["H"][0].velocities), _host(initial))


# --------------------------------------------------------------------- #
# Operator settings
# --------------------------------------------------------------------- #
def test_large_collision_probability_warns_once() -> None:
    mcc = MonteCarloCollisions(
        {"e": M_E},
        [NeutralBackground("gas", 1.0e20, 0.0, HEAVY)],
        {"e": [CollisionProcess("elastic", "gas", constant(1e-19))]},
        seed=1,
        max_collision_probability=0.05,
    )
    velocities = beam(100, 10.0, M_E)
    speed = float(xp.to_numpy(velocities)[0, 0])
    small_dt = 0.01 / (1.0e20 * 1e-19 * speed)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quiet = mcc.collide("e", _origin(100), velocities, _ones(100), small_dt)
    assert quiet.diagnostics.collision_probability < 0.05
    with pytest.warns(RuntimeWarning, match="collision probability per step"):
        loud = mcc.collide("e", _origin(100), velocities, _ones(100), 100 * small_dt)
    assert loud.diagnostics.collision_probability > 0.05
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mcc.collide("e", _origin(100), velocities, _ones(100), 100 * small_dt)


def test_operator_rejects_bad_probability_limit_and_duplicate_backgrounds() -> None:
    gas = NeutralBackground("gas", 1.0e20, 0.0, HEAVY)
    elastic = {"e": [CollisionProcess("elastic", "gas", constant(1e-19))]}
    with pytest.raises(ValueError, match="max_collision_probability"):
        MonteCarloCollisions({"e": M_E}, [gas], elastic, max_collision_probability=0.0)
    with pytest.raises(ValueError, match="unique"):
        MonteCarloCollisions({"e": M_E}, [gas, gas], elastic)


def test_set_background_retabulates_the_bound() -> None:
    mcc = operator([CollisionProcess("elastic", "gas", constant(1e-19))], {"e": M_E})
    before = mcc.frequency_bound("e", 1.0e6)
    mcc.set_background(NeutralBackground("gas", 2.0e20, 0.0, HEAVY))
    assert mcc.backgrounds["gas"].density == 2.0e20
    assert mcc.frequency_bound("e", 1.0e6) == pytest.approx(2.0 * before)
    with pytest.raises(KeyError, match="No background"):
        mcc.set_background(NeutralBackground("other", 1.0, 0.0, HEAVY))


def test_process_validates_sharing_and_scattering() -> None:
    with pytest.raises(ValueError, match="sharing_energy"):
        _ionization(energy_sharing="opal")
    with pytest.raises(ValueError, match="scattering"):
        CollisionProcess("elastic", "gas", constant(1e-19), scattering="forward")
    with pytest.raises(ValueError, match="no product role"):
        CollisionProcess("elastic", "gas", constant(1e-19), products={"fragment": "H"})
