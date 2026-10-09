---
title: Negative-ion sources
description: Proton transport in hydrogen and the vibrational population behind H⁻ volume production.
sidebar:
  order: 6
---

## Protons in hydrogen

`plasmacoll.proton_hydrogen_processes(cross_sections, h2_ion=None)` builds the six processes that
dominate proton transport in the driver and expansion region of an RF-driven negative hydrogen
ion source (the test-particle model of Wünderlich et al.):

| Key      | Reaction                           | Kind              | Energy frame   |
| -------- | ---------------------------------- | ----------------- | -------------- |
| `mt_h`   | H⁺ + H → H⁺ + H                    | `elastic`         | centre of mass |
| `mt_h2`  | H⁺ + H₂ → H⁺ + H₂                  | `elastic`         | lab            |
| `cx_h`   | H⁺ + H → H + H⁺                    | `backscatter`     | lab            |
| `cx_h2`  | H⁺ + H₂ → H + H₂⁺                  | `charge_transfer` | lab            |
| `rot_h2` | H⁺ + H₂(J=0) → H⁺ + H₂(J=2)        | `excitation`      | lab            |
| `vib_h2` | H⁺ + H₂(v=0) → H⁺ + H₂(v>0)        | `excitation`      | lab            |

The function supplies the process definitions, not the cross sections. The model's sources are
Krstić and Schultz for H⁺ + H₂ and Janev et al. for H⁺ + H (see also AMJUEL/HYDHEL). The
rotational (0.0453 eV) and vibrational (0.516 eV) thresholds replace those of the tables. With
`h2_ion="H2+"` the charge exchange with H₂ creates H₂⁺ ions. Otherwise it only removes the proton.

The fit formulas of AMJUEL/HYDHEL and of Janev's compilations can be evaluated with
`plasmacoll.cross_section_from_log_polynomial` and `log_polynomial`; copy the coefficients and
ranges of validity from the source. See [cross sections](/plasmacoll/guides/cross-sections/#fits).

## H⁻ losses between charged particles

Mutual neutralization with positive ions, $\mathrm{H^-} + \mathrm{H^+} \to 2\mathrm{H}$, and
detachment by electrons, $e + \mathrm{H^-} \to \mathrm{H} + 2e$, are collisions between charged
particles. `ChargedReaction("mutual_neutralization", ("H-", "H+"), ...)` and
`ChargedReaction("electron_detachment", ("e", "H-"), ...)` add them, with a rate coefficient or a
cross section from the literature. See [charged-particle collisions](/plasmacoll/guides/charged-collisions/).

## Vibrationally excited H₂

H⁻ is produced in the volume by dissociative attachment, $e + \mathrm{H_2}(v) \to \mathrm{H^-} +
\mathrm{H}$. The cross section rises by orders of magnitude from $v=0$ to the highest bound levels
(Bardsley and Wadehra 1979; Wadehra 1984). A model that uses one ground-state cross section for
all of H₂ gets the production rate wrong by orders of magnitude.

`plasmacoll.vibrational` supplies a *prescribed* population: the H₂(X) vibrational ladder of
Fantz and Wünderlich (At. Data Nucl. Data Tables **92**, 853 (2006)) with a Boltzmann distribution
at one vibrational temperature.

```python
from plasmacoll.vibrational import VibrationalDistribution

distribution = VibrationalDistribution.h2_ground_state_from_kelvin(5000.0)
distribution.populations  # n_v / n for v = 0..14
distribution.fraction_at_least(4)  # [H2(v >= 4)] / [H2]

attachment = plasmacoll.CrossSection.vibrational_mixture(
    level_tables, distribution.populations
)
```

A real population deviates from a single Boltzmann distribution at high $v$, so this
underestimates the tail that drives attachment. It is a first approximation, not a solution of
the vibrational kinetics. `estimate_electronegativity` turns the excited fraction into an
order-of-magnitude $[\mathrm{H^-}]/[e]$ from the balance of attachment against detachment by H
atoms (Kimura and Kasugai 2010, as given by Krištof et al. 2019).

plasmacoll does not ship level-resolved attachment cross sections. Read them from tables with
`CrossSection.from_table`. [Tutorial 5](/plasmacoll/tutorials/05-hydrogen-negative-ions/) runs the
pipeline with synthetic levels.
