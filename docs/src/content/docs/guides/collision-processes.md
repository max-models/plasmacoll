---
title: Collision processes
description: The process kinds, their products and energy bookkeeping, scattering models and the energy frames of cross sections.
sidebar:
  order: 2
---

A `CollisionProcess` names its `kind`, the `background` it collides with and its
`cross_section`. All kinematics are evaluated with a neutral partner drawn from the background's
Maxwellian, in the centre-of-mass frame where that matters. $m$ is the incident mass, $M$ the
neutral mass, $\mu = mM/(m+M)$ the reduced mass, $\mathbf V$ the centre-of-mass velocity,
$\mathbf w$ the neutral's velocity and $\mathbf g$ the relative velocity. "Scattered" means
isotropically, or with the process's [scattering model](#scattering-models).

| Kind                       | Reaction                     | What happens                                                                                                | Product roles                          |
| -------------------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `elastic`                  | $A + B \to A + B$            | $\mathbf g$ scattered, $\lvert\mathbf g\rvert$ kept                                                         | none                                   |
| `backscatter`              | $A^+ + A \to A + A^+$        | $\mathbf g$ reversed: for equal masses the ion leaves with the neutral's velocity                           | optional `neutral`                     |
| `excitation`               | $A + B \to A + B^*$          | scattered, $\tfrac12\mu g^2$ reduced by the energy loss                                                     | none                                   |
| `dissociation`             | $e + AB \to e + A + B$       | as excitation                                                                                               | optional `fragment`                    |
| `ionization`               | $e + B \to 2e + B^+$         | $E - E_{iz}$ shared by the two electrons, both scattered in the neutral frame; ion at $\mathbf w$           | `electron`, `ion`                      |
| `dissociative_ionization`  | $e + AB \to 2e + A^+ + B$    | as ionization                                                                                               | `electron`, `ion`, optional `fragment` |
| `attachment`               | $e + B \to B^-$              | electron removed, negative ion created at $\mathbf w$                                                       | `negative_ion`                         |
| `dissociative_attachment`  | $e + AB \to A^- + B$         | as attachment                                                                                               | `negative_ion`, optional `fragment`    |
| `detachment`               | $A^- + B \to A + B + e$      | negative ion removed, electron emitted isotropically about $\mathbf V$ with $\tfrac12\mu g^2 - E_{th}$      | `electron`                             |
| `charge_transfer`          | $A^+ + B \to A + B^+$        | incident removed; with an `ion` product, $B^+$ created at $\mathbf w$                                       | optional `ion`, `neutral`              |

`products` maps each role to a species name, for example
`{"electron": "e", "ion": "Ar+"}`. Every species involved needs a mass in `species_masses`.
Products take the position and the weight of the incident marker, which conserves charge and
particle number marker by marker, so markers may carry any weight. A code with one fixed weight
per species must give species that exchange particles the same weight. To keep the number of
markers in check, see [marker weights](#marker-weights).

## Tracking neutrals

Neutrals are a prescribed background, not markers, but the fast neutrals of charge exchange and
the fragments of dissociation can be kept as a species of their own:

- `neutral` (`backscatter`, `charge_transfer`): the fast neutral, with the incident ion's
  velocity before the collision.
- `fragment` (the dissociative kinds): one neutral fragment at the neutral's velocity.

Without a species for these roles they are dropped. Those species usually have no processes of
their own and only receive products.

## Energy loss and sharing

Excitation, dissociation, ionization and detachment lose `energy_loss` (eV), which defaults to
the threshold of the cross section. Ionization shares the residual energy $E - E_{iz}$ between the
primary and the secondary electron:

- `energy_sharing="equal"` (default): half each.
- `"uniform"`: a uniformly distributed fraction.
- `"opal"`: the secondary energy $E_s \in [0, (E-E_{iz})/2]$ drawn from the
  Opal–Peterson–Beaty distribution $\propto 1 / (1 + (E_s/w)^2)$, with `sharing_energy=w` in eV
  (about 10 eV for argon, 8.3 eV for H₂). This is the usual choice for electron swarms and
  discharges: most secondaries are slow.

If the incident species is not the product electron (ion impact, $A^+ + B \to A^+ + B^+ + e$),
the electron takes its share of the centre-of-mass energy $\tfrac12\mu g^2 - E_{iz}$ and is
emitted isotropically from $\mathbf V$, and incident and target ion separate along the incident
direction with the rest.

## Scattering models

`scattering` sets the angular distribution of elastic, excitation, dissociation and ionization
collisions about the incident direction:

- `"isotropic"` (default).
- `"vahedi_surendra"`: $\cos\chi = (2 + E - 2(1+E)^R)/E$ with $E$ in eV and $R$ uniform
  (Vahedi and Surendra, Comput. Phys. Commun. 87, 179 (1995)).
- `"okhrimovskyy"`: $\cos\chi = 1 - 2R(1-\xi)/(1 + \xi(1-2R))$ with
  $\xi = 4\varepsilon/(1+4\varepsilon)$ and $\varepsilon$ the energy in Hartree
  (Okhrimovskyy et al., Phys. Rev. E 65, 037402 (2002)).

Both are screened-Coulomb forms for electrons: isotropic at low energy and forward-peaked above a
few eV. They change which cross section belongs in the `elastic` process. With isotropic
scattering it must be the **momentum-transfer** cross section, so that momentum relaxes at the
right rate; with an anisotropic model it must be the **integral elastic** cross section. Ionized
electrons are scattered with their own energies after the collision.

`plasmacoll.scattering_cosine(model, energy, uniform)` evaluates the models directly.

## Energy frames

`energy_frame` selects the energy the cross section is looked up at:

- `"center_of_mass"` (default): $E = \tfrac12\mu g^2$. Use it for ion-neutral data given in the
  centre-of-mass frame.
- `"lab"`: $E = \tfrac12 m g^2$, the incident particle's energy in the rest frame of the target.
  This is the convention of electron cross-section compilations such as LXCat, and of most
  ion-beam data.

For electrons the two agree to $O(m_e/M)$. For ions on a gas of similar mass they differ by a
factor $(m+M)/M$, so pick the one the data was tabulated in.

## Marker weights

Every ionization doubles a marker, and attachment and charge transfer remove them, so the number
of markers drifts away from what a run can afford. `plasmacoll.population` controls it on any
marker container:

- `merge_markers(markers, target, cell_size=None)` groups markers in the same spatial cell and
  velocity octant by speed and replaces every group of three or more by two markers that keep
  the group's weight, momentum and kinetic energy exactly (Vranic et al., Comput. Phys. Commun.
  191, 65 (2015)).
- `split_markers(markers, max_weight)` divides markers heavier than `max_weight` into equal
  copies.

`ZeroDReactor(..., max_markers=N)` merges a species down to $N/2$ markers whenever it exceeds $N$.

## Process sets for hydrogen

`plasmacoll.proton_hydrogen_processes` returns the six processes of a test-particle model for protons
in negative-ion sources: momentum transfer with H and H₂, charge exchange with H (backscatter) and
H₂ (charge transfer), and rotational and vibrational excitation of H₂. It takes the cross-section
tables as input. See [negative-ion sources](/plasmacoll/guides/negative-ion-sources/).

## Out of scope

The targets are always neutral gases. Collisions between charged particles (Coulomb collisions
such as electron–electron or electron–ion, recombination, mutual neutralization) need a
different operator (e.g. Takizuka–Abe or Nanbu) and are not part of plasmacoll.
