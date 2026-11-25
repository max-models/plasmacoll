"""The yields of particle-surface interactions."""

import astropy.units as u
import numpy as np
import pytest

from plasmacoll import ConstantYield, TabulatedYield, VaughanYield, surface_yield


def test_a_constant_yield_has_the_shape_of_its_arguments() -> None:
    """One value at every energy and angle."""
    model = ConstantYield(0.25)
    assert np.allclose(model(np.array([1.0, 100.0, 1.0e4])), 0.25)
    assert model(np.ones((2, 3)), np.full((2, 3), 0.3)).shape == (2, 3)
    with pytest.raises(ValueError):
        ConstantYield(-0.1)


def test_a_table_interpolates_and_holds_its_end_values() -> None:
    """Linear between the points, the end values outside them."""
    model = TabulatedYield((10.0, 100.0), (0.0, 0.2))
    assert model(55.0) == pytest.approx(0.1)
    assert model(1.0) == pytest.approx(0.0)
    assert model(1.0e4) == pytest.approx(0.2)
    assert TabulatedYield("[0.01, 0.1] keV", (0.0, 0.2)).energy == (10.0, 100.0)
    with pytest.raises(ValueError):
        TabulatedYield((100.0, 10.0), (0.0, 0.2))
    with pytest.raises(ValueError):
        TabulatedYield((10.0, 100.0), (0.0, -0.2))


def test_vaughan_peaks_at_its_maximum_and_is_zero_below_threshold() -> None:
    """At normal incidence the yield is max_yield at energy_at_max, 0 below E_0."""
    model = VaughanYield(2.0, 300.0 * u.eV, threshold=12.5)
    energy = np.linspace(13.0, 3000.0, 4000)
    yields = model(energy)
    assert energy[np.argmax(yields)] == pytest.approx(300.0, rel=1e-2)
    assert model(300.0) == pytest.approx(2.0)
    assert model(np.array([0.0, 12.5])) == pytest.approx([0.0, 0.0])
    # continuous where the fit switches to its high-energy tail (v = 3.6)
    switch = 12.5 + 3.6 * (300.0 - 12.5)
    assert model(switch - 1e-6) == pytest.approx(model(switch + 1e-6), rel=1e-3)


def test_vaughan_rises_at_oblique_incidence() -> None:
    """Grazing electrons release more secondaries; smoothness 0 removes the effect."""
    model = VaughanYield(1.5, 400.0)
    assert model(400.0, 0.2) > model(400.0, 1.0)
    flat = VaughanYield(1.5, 400.0, smoothness=0.0)
    assert flat(400.0, 0.2) == pytest.approx(flat(400.0, 1.0))
    with pytest.raises(ValueError):
        VaughanYield(1.5, 400.0, threshold=500.0)


@pytest.mark.parametrize(
    "model",
    [
        ConstantYield(0.1),
        TabulatedYield((10.0, 100.0), (0.0, 0.2)),
        VaughanYield(2.2, 350.0, threshold=10.0, smoothness=2.0),
    ],
)
def test_the_configuration_file_form_reads_back(model) -> None:
    """``surface_yield(model.to_dict())`` is the same model."""
    assert surface_yield(model.to_dict()) == model


def test_surface_yield_reads_numbers_and_rejects_unknown_models() -> None:
    """A number is a constant yield; a yield passes through."""
    assert surface_yield(0.3) == ConstantYield(0.3)
    model = VaughanYield(1.0, 300.0)
    assert surface_yield(model) is model
    with pytest.raises(ValueError):
        surface_yield({"model": "magic"})
    with pytest.raises(TypeError):
        surface_yield("0.3")
