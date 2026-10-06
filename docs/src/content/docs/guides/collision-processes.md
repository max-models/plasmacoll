---
title: Collision processes
description: The process kinds, their products and energy bookkeeping, and the energy frames of cross sections.
sidebar:
  order: 2
---

A `CollisionProcess` names its `kind`, the `background` it collides with and its
`cross_section`. All kinematics are evaluated with a neutral partner drawn from the background's
Maxwellian, in the centre-of-mass frame where that matters. $m$ is the incident mass, $M$ the
neutral mass, $\mu = mM/(m+M)$ the reduced mass, $\mathbf V$ the centre-of-mass velocity and
$\mathbf g$ the relative velocity.

| Kind              | Reaction                         | What happens                                                                                       | Product roles            |
| ----------------- | -------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------ |
| `elastic`         | $A + B \to A + B$                | $\mathbf g$ rotated isotropically, $\lvert\mathbf g\rvert$ kept                                    | none                     |
| `backscatter`     | $A^+ + A \to A + A^+$            | $\mathbf g$ reversed: for equal masses the ion leaves with the neutral's velocity                  | none                     |
| `excitation`      | $A + B \to A + B^*$              | isotropic, $\tfrac12\mu g^2$ reduced by the energy loss                                            | none                     |
| `ionization`      | $e + B \to 2e + B^+$             | $E - E_{iz}$ shared by the two electrons, both isotropic in the neutral frame; ion at $\mathbf w$  | `electron`, `ion`        |
| `attachment`      | $e + B \to B^-$                  | electron removed, negative ion created at $\mathbf w$                                              | `negative_ion`           |
| `detachment`      | $A^- + B \to A + B + e$          | negative ion removed, electron emitted isotropically about $\mathbf V$ with $\tfrac12\mu g^2 - E_{th}$ | `electron`           |
| `charge_transfer` | $A^+ + B \to A + B^+$            | incident removed (fast neutrals are not tracked); with an `ion` product, $B^+$ created at $\mathbf w$ | optional `ion`        |

`products` maps each role to a species name, for example
`{"electron": "e", "ion": "Ar+"}`. Every species involved needs a mass in `species_masses`.
Products take the position and the weight of the incident marker, so species that exchange
particles must have equal marker weights to conserve charge.

## Energy loss

Excitation, ionization and detachment lose `energy_loss` (eV), which defaults to the threshold
of the cross section. Ionization shares the residual energy equally (`energy_sharing="equal"`)
or with a uniformly distributed fraction (`"uniform"`).

## Energy frames

`energy_frame` selects the energy the cross section is looked up at:

- `"center_of_mass"` (default): $E = \tfrac12\mu g^2$. Use it for ion-neutral data given in the
  centre-of-mass frame.
- `"lab"`: $E = \tfrac12 m g^2$, the incident particle's energy in the rest frame of the target.
  This is the convention of electron cross-section compilations such as LXCat, and of most
  ion-beam data.

For electrons the two agree to $O(m_e/M)$. For ions on a gas of similar mass they differ by a
factor $(m+M)/M$, so pick the one the data was tabulated in.

## Process sets for hydrogen

`plasmacoll.proton_hydrogen_processes` returns the six processes of a test-particle model for protons
in negative-ion sources: momentum transfer with H and H₂, charge exchange with H (backscatter) and
H₂ (charge transfer), and rotational and vibrational excitation of H₂. It takes the cross-section
tables as input. See [negative-ion sources](/plasmacoll/guides/negative-ion-sources/).
