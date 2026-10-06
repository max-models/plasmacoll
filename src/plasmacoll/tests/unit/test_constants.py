import pytest

from plasmacoll.constants import (
    BOLTZMANN,
    ELEMENTARY_CHARGE,
    ev_to_kelvin,
    kelvin_to_ev,
)


def test_kelvin_and_ev_are_inverse() -> None:
    assert ev_to_kelvin(kelvin_to_ev(300.0)) == pytest.approx(300.0)
    assert kelvin_to_ev(1.0) == pytest.approx(BOLTZMANN / ELEMENTARY_CHARGE)
    assert ev_to_kelvin(1.0) == pytest.approx(11604.518, rel=1e-6)
