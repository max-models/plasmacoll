"""Charge states of a fast beam against closed forms; the cross sections are synthetic."""

from __future__ import annotations

import numpy as np
import pytest

from plasmacoll import ChargeStateTransitions, charge_state_fractions

PI = np.linspace(0.0, 4.0e20, 41)  # line density, m^-2


def test_single_stripping_alone_decays_exponentially() -> None:
    sigma = 1.0e-20
    beam = ChargeStateTransitions({"gas": {("-", "0"): sigma}})
    fractions = charge_state_fractions(beam, {"gas": PI})
    assert fractions[:, 0] == pytest.approx(np.exp(-sigma * PI), rel=1e-12)
    assert fractions[:, 1] == pytest.approx(1.0 - np.exp(-sigma * PI), abs=1e-14)
    assert np.all(fractions[:, 2] == 0.0)


@pytest.mark.parametrize("b", [3.0e-21, 1.0e-20])  # the second: equal cross sections
def test_stripping_then_ionization_gives_the_two_step_solution(b: float) -> None:
    a = 1.0e-20
    beam = ChargeStateTransitions({"gas": {("-", "0"): a, ("0", "+"): b}})
    neutral = charge_state_fractions(beam, {"gas": PI})[:, 1]
    if a == b:
        expected = a * PI * np.exp(-a * PI)
    else:
        expected = a / (b - a) * (np.exp(-a * PI) - np.exp(-b * PI))
    assert neutral == pytest.approx(expected, rel=1e-10, abs=1e-15)
    # the neutral fraction peaks at ln(b / a) / (b - a), or 1 / a when equal
    peak = 1.0 / a if a == b else np.log(b / a) / (b - a)
    fine = np.linspace(0.0, 4.0e20, 4001)
    found = fine[np.argmax(charge_state_fractions(beam, {"gas": fine})[:, 1])]
    assert found == pytest.approx(peak, rel=1e-3)


def test_fractions_are_conserved_and_reach_equilibrium() -> None:
    sigma = {
        ("-", "0"): 1.2e-20,
        ("-", "+"): 1e-21,
        ("0", "+"): 4e-21,
        ("+", "0"): 2e-22,
    }
    beam = ChargeStateTransitions({"gas": sigma})
    fractions = charge_state_fractions(beam, {"gas": np.linspace(0, 1e22, 200)})
    assert fractions.sum(axis=1) == pytest.approx(1.0, abs=1e-12)
    equilibrium = beam.equilibrium()
    assert beam.matrix("gas") @ equilibrium == pytest.approx(0.0, abs=1e-30)
    assert fractions[-1] == pytest.approx(equilibrium, abs=1e-9)
    assert equilibrium[0] == pytest.approx(0.0, abs=1e-12)  # nothing returns to -


def test_two_targets_at_a_fixed_ratio_act_as_one_with_summed_cross_sections() -> None:
    gas = {("-", "0"): 1e-20, ("0", "+"): 3e-21}
    plasma = {("-", "0"): 4e-20, ("0", "+"): 2e-20}
    beam = ChargeStateTransitions({"gas": gas, "plasma": plasma})
    both = charge_state_fractions(beam, {"gas": PI, "plasma": 0.25 * PI})
    summed = ChargeStateTransitions(
        {"mix": {k: gas[k] + 0.25 * plasma[k] for k in gas}}
    )
    assert both == pytest.approx(charge_state_fractions(summed, {"mix": PI}), rel=1e-10)
    # a plasma target strips more for the same line density
    only_plasma = charge_state_fractions(beam, {"plasma": PI})
    only_gas = charge_state_fractions(beam, {"gas": PI})
    assert only_plasma[10, 0] < only_gas[10, 0]


def test_initial_fractions_and_bad_inputs() -> None:
    beam = ChargeStateTransitions({"gas": {("0", "+"): 1e-20}})
    fractions = charge_state_fractions(beam, {"gas": PI}, initial={"0": 1.0})
    assert fractions[0] == pytest.approx([0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="no cross sections"):
        charge_state_fractions(beam, {"plasma": PI})
    with pytest.raises(ValueError, match="non-decreasing"):
        charge_state_fractions(beam, {"gas": PI[::-1]})
    with pytest.raises(ValueError, match="no transition"):
        ChargeStateTransitions({"gas": {("-", "2+"): 1e-20}})
    with pytest.raises(ValueError, match=">= 0"):
        ChargeStateTransitions({"gas": {("-", "0"): -1e-20}})
