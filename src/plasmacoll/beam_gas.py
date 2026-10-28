"""The reactions of a fast hydrogen or deuterium beam with the gas of its beam line.

A negative-ion beam on its way through the accelerator and the neutralizer
meets the gas that leaks from the source: its ions are stripped (H- -> H, the
loss of the accelerator), the stripped neutrals ionized (H -> H+), the positive
ions neutralized by capture (H+ + H2 -> H + H2+), and every state ionizes the gas
(X + H2 -> X + H2+ + e), whose slow ions the accelerator's field sends back
towards the source (backstreaming). :class:`BeamGasReactions` holds these
reactions with their cross sections against the projectile's energy and gives
them both as charge-state cross sections at one energy
(:meth:`~BeamGasReactions.transitions`, for
:func:`~plasmacoll.charge_states.charge_state_fractions`) and as processes of
:class:`~plasmacoll.mcc.MonteCarloCollisions` (:meth:`~BeamGasReactions.processes`,
``stripping`` and ``beam_ionization``), so that markers and fractions use the
same data. A gas profile along the beam is a
:class:`~plasmacoll.background.DensityProfile` of the gas background.

:func:`hydrogen_beam_on_h2` assembles them for H2 gas from their sources:

=====================  ==================================================  ===============
reaction               cross section                                       status
=====================  ==================================================  ===============
H- -> H                Janev, Reiter and Samm (2003), eq. (159)            data
H+ -> H (H2+ left)     Janev, Reiter and Samm (2003), eqs. (144), (145)    data to 200 keV
H+ + H2 -> H2+ + e     Rudd et al. (1992)                                  data
H- -> H+               0.042 times H- -> H                                 placeholder
H -> H+                0.31 times H- -> H                                  placeholder
H, H- + H2 -> H2+ + e  Rudd's proton cross section at the same speed       placeholder
=====================  ==================================================  ===============

The placeholders' ratios are those of the round 1 MeV D- numbers of
wellbalanced's illustrative neutralizer (freeofcharge's ``GAS_NEUTRALIZER``);
they give the right order of magnitude, not data, and every reaction says
whether it is one (:attr:`BeamGasReaction.placeholder`). The capture fit is
fitted to about 200 keV; above, its power-law tail is kept, an extrapolation.
A deuterium beam takes the hydrogen data at the same speed (half the energy),
as BBCNI (Hurlbatt et al., Plasma Phys. Control. Fusion 61, 105012, 2019) does
for D- with the H- data; the reactions' sources say so.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import cunumpy as xp
import numpy as np

from plasmacoll.charge_states import (
    HYDROGEN_CHARGES,
    HYDROGEN_STATES,
    ChargeStateTransitions,
)
from plasmacoll.constants import ATOMIC_MASS, ELECTRON_MASS
from plasmacoll.cross_sections import CrossSection
from plasmacoll.janev import JANEV_FITS
from plasmacoll.process import CollisionProcess
from plasmacoll.secondaries import RUDD_H2

__all__ = [
    "PLACEHOLDER_RATIOS",
    "BeamGasReaction",
    "BeamGasReactions",
    "hydrogen_beam_on_h2",
]

#: Placeholder cross sections as fractions of H- -> H on H2: double stripping
#: and the stripping of the neutrals (the ratios of 5.0e-22 and 3.7e-21 to
#: 1.2e-20 m^2, wellbalanced's illustrative 1 MeV D- values).
PLACEHOLDER_RATIOS = {("-", "+"): 5.0e-22 / 1.2e-20, ("0", "+"): 3.7e-21 / 1.2e-20}
#: Width ``w`` in eV of the ejected electrons' energy distribution in H2: the
#: binding energy (15.43 eV), the scale of Rudd's spectrum.
H2_SHARING_ENERGY = 15.43
#: The masses in kg of the hydrogen and deuterium atoms (CODATA 2018).
_ATOM_MASS = {"H": 1.00782503207 * ATOMIC_MASS, "D": 2.01410177812 * ATOMIC_MASS}


@dataclass(frozen=True)
class BeamGasReaction:
    """One reaction of a beam state with a gas target.

    Attributes:
        start: The beam state before (``"-"``, ``"0"`` or ``"+"``).
        end: The state after, or None for the ionization of the target, which
            leaves the state unchanged.
        target: The target's name.
        cross_section: Against the projectile's energy in eV with the target at
            rest (the lab frame), for the beam's isotope.
        source: Where the cross section comes from.
        placeholder: Whether the cross section is a stand-in, not data.
    """

    start: str
    end: str | None
    target: str
    cross_section: CrossSection
    source: str
    placeholder: bool = False

    @property
    def is_ionization(self) -> bool:
        """Whether the reaction ionizes the target and keeps the beam's state."""
        return self.end is None

    @property
    def name(self) -> str:
        """``"<start>-><end>:<target>"``, or ``"<start>:ionization:<target>"``."""
        if self.end is None:
            return f"{self.start}:ionization:{self.target}"
        return f"{self.start}->{self.end}:{self.target}"


@dataclass(frozen=True)
class BeamGasReactions:
    """The reactions of a beam with its gas, as fractions and as Monte Carlo processes.

    Attributes:
        reactions: The reactions.
        states: The beam's charge states, in the order of the fractions.
        charges: The charge of every state in e.
        sharing_energy: The width ``w`` in eV of the energy of the electrons
            the ionization ejects (see ``beam_ionization`` in
            :mod:`plasmacoll.mcc`).
    """

    reactions: Sequence[BeamGasReaction]
    states: Sequence[str] = HYDROGEN_STATES
    charges: Mapping[str, int] = field(default_factory=lambda: dict(HYDROGEN_CHARGES))
    sharing_energy: float = H2_SHARING_ENERGY

    def __post_init__(self) -> None:
        """Validate the states of the reactions."""
        object.__setattr__(self, "reactions", tuple(self.reactions))
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "charges", dict(self.charges))
        missing = set(self.states) - set(self.charges)
        if missing:
            raise ValueError(f"no charge given for the state(s) {sorted(missing)}")
        for reaction in self.reactions:
            ends = (reaction.start,) + (() if reaction.end is None else (reaction.end,))
            if any(state not in self.states for state in ends):
                raise ValueError(f"{reaction.name}: unknown state")
            if reaction.end == reaction.start:
                raise ValueError(f"{reaction.name}: a transition changes the state")
        names = [reaction.name for reaction in self.reactions]
        if len(set(names)) != len(names):
            raise ValueError("every reaction may appear once")

    @property
    def targets(self) -> tuple[str, ...]:
        """The target names, in the order they first appear."""
        return tuple(dict.fromkeys(reaction.target for reaction in self.reactions))

    @property
    def placeholders(self) -> tuple[str, ...]:
        """The names of the reactions whose cross sections are placeholders."""
        return tuple(r.name for r in self.reactions if r.placeholder)

    def transitions(self, energy: float) -> ChargeStateTransitions:
        """Return the charge-changing cross sections at the projectile energy (eV)."""
        table: dict[str, dict[tuple[str, str], float]] = {t: {} for t in self.targets}
        for reaction in self.reactions:
            if reaction.end is not None:
                sigma = float(xp.to_numpy(reaction.cross_section(energy)))
                table[reaction.target][(reaction.start, reaction.end)] = sigma
        return ChargeStateTransitions(table, self.states)

    def ionization(self, energy: float) -> dict[str, dict[str, float]]:
        """Return ``{target: {state: sigma}}`` (m^2) of the ionization of the targets."""
        table: dict[str, dict[str, float]] = {t: {} for t in self.targets}
        for reaction in self.reactions:
            if reaction.end is None:
                sigma = float(xp.to_numpy(reaction.cross_section(energy)))
                table[reaction.target][reaction.start] = sigma
        return table

    def processes(
        self,
        species: Mapping[str, str],
        backgrounds: Mapping[str, str] | None = None,
        *,
        electron: str | None = None,
        ions: Mapping[str, str] | None = None,
    ) -> dict[str, list[CollisionProcess]]:
        """Return the processes of every state for the Monte Carlo collisions.

        A transition becomes a ``stripping`` process that moves the marker to
        the species of its end state, the ionization of a target a
        ``beam_ionization`` process that creates the target's ion; both look up
        their cross sections in the lab frame.

        Args:
            species: The species name of every state, e.g.
                ``{"-": "H-", "0": "H", "+": "H+"}``; states without a species
                are left out with their reactions.
            backgrounds: The background name of every target (the target name
                if missing).
            electron: The species of the released and ejected electrons; not
                tracked if None.
            ions: ``{target: species}``, the ion that a capture or an
                ionization leaves behind. The ionization of a target without one
                is left out; a capture on it leaves no ion.

        Returns:
            ``{species: [processes]}``, named as the reactions.
        """
        backgrounds = dict(backgrounds or {})
        ions = dict(ions or {})
        processes: dict[str, list[CollisionProcess]] = {}
        for reaction in self.reactions:
            if reaction.start not in species:
                continue
            background = backgrounds.get(reaction.target, reaction.target)
            products: dict[str, str] = {}
            end = reaction.end
            if end is None:
                if reaction.target not in ions:
                    continue
                products["ion"] = ions[reaction.target]
                if electron is not None:
                    products["electron"] = electron
                process = CollisionProcess(
                    "beam_ionization",
                    background,
                    reaction.cross_section,
                    name=reaction.name,
                    energy_frame="lab",
                    products=products,
                    sharing_energy=self.sharing_energy,
                )
            else:
                if end not in species:
                    continue
                change = int(self.charges[end]) - int(self.charges[reaction.start])
                products["projectile"] = species[end]
                if change > 0 and electron is not None:
                    products["electron"] = electron
                if change < 0 and reaction.target in ions:
                    products["ion"] = ions[reaction.target]
                process = CollisionProcess(
                    "stripping",
                    background,
                    reaction.cross_section,
                    name=reaction.name,
                    energy_frame="lab",
                    products=products,
                    charge_change=change,
                )
            processes.setdefault(species[reaction.start], []).append(process)
        return processes


def _scaled(sigma: CrossSection, factor: float, label: str) -> CrossSection:
    """``sigma`` times ``factor``, with a new label."""
    return CrossSection(
        energy=sigma.energy,
        sigma=sigma.sigma * factor,
        threshold=sigma.threshold,
        label=label,
    )


def hydrogen_beam_on_h2(
    isotope: str = "H",
    max_energy: float = 2.0e6,
    target: str = "H2",
    num: int = 241,
) -> BeamGasReactions:
    """Return the reactions of a hydrogen or deuterium beam with H2 (or D2) gas.

    See the module docstring for the sources and the placeholders.

    Args:
        isotope: ``"H"`` or ``"D"``; a deuterium beam takes the hydrogen data at
            the same speed.
        max_energy: The largest projectile energy in eV of the tables; beyond
            it the last values hold.
        target: The name of the gas target.
        num: The number of table points per cross section.

    Returns:
        The reactions, with their cross sections against the projectile's lab
        energy.
    """
    if isotope not in _ATOM_MASS:
        raise ValueError(f"isotope must be 'H' or 'D', not {isotope!r}")
    if max_energy <= 0.0:
        raise ValueError("max_energy must be > 0")
    # the hydrogen data at the energy of a hydrogen projectile of the same speed
    speed_ratio = _ATOM_MASS["H"] / _ATOM_MASS[isotope]
    flag = "" if isotope == "H" else "; hydrogen data at the same speed"
    energy = np.geomspace(1.0, max_energy, num)

    def tabulated(function, threshold: float, label: str) -> CrossSection:
        grid = (
            np.concatenate([[threshold / speed_ratio], energy]) if threshold else energy
        )
        grid = np.unique(grid[grid >= threshold / speed_ratio])
        return CrossSection.from_function(
            lambda e: function(e * speed_ratio),
            grid,
            threshold=threshold / speed_ratio,
            label=label,
        )

    detachment = JANEV_FITS["hminus_h2_detachment"]
    capture = JANEV_FITS["proton_h2_charge_exchange"]
    stripping = tabulated(detachment, detachment.threshold, detachment.label + flag)
    # the masses of the three states differ by electrons only: one speed, one table
    masses = {
        "-": _ATOM_MASS[isotope] + ELECTRON_MASS,
        "0": _ATOM_MASS[isotope],
        "+": _ATOM_MASS[isotope] - ELECTRON_MASS,
    }

    def rudd(state: str) -> CrossSection:
        model = RUDD_H2.with_projectile(masses[state])
        label = f"{RUDD_H2.label or 'Rudd ionization of H2'} by {state}"
        return CrossSection.from_function(
            lambda e: np.array([model.total(float(x)) for x in e]), energy, label=label
        )

    placeholder = "placeholder: {ratio:.3g} times H- -> H"
    reactions = [
        BeamGasReaction("-", "0", target, stripping, detachment.label + flag),
        BeamGasReaction(
            "+", "0", target,
            tabulated(capture, capture.threshold, capture.label + flag),
            capture.label + "; above 200 keV the fit's tail, extrapolated" + flag,
        ),
        BeamGasReaction("+", None, target, rudd("+"), "Rudd et al. (1992)" + flag),
    ]  # fmt: skip
    for pair, ratio in PLACEHOLDER_RATIOS.items():
        label = placeholder.format(ratio=ratio)
        reactions.append(
            BeamGasReaction(
                *pair, target, _scaled(stripping, ratio, label), label, True
            )
        )
    for state in ("-", "0"):
        label = "placeholder: Rudd's proton ionization at the same speed"
        reactions.append(BeamGasReaction(state, None, target, rudd(state), label, True))
    return BeamGasReactions(reactions)
