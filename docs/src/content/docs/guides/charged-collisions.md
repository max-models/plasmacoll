---
title: Charged-particle collisions
description: Coulomb scattering and reactions between charged species with ChargedCollisions, their pairing in cells, and their accuracy.
sidebar:
  order: 5
---

`MonteCarloCollisions` collides charged particles with prescribed neutral gases.
`ChargedCollisions` handles what happens between the charged particles themselves: Coulomb
scattering and reactions such as mutual neutralization. It pairs markers within spatial cells and
works on the same marker containers.

```python
charged = plasmacoll.ChargedCollisions(
    species_masses={"e": ELECTRON_MASS, "H+": M_H, "H-": M_H, "H": M_H},
    charges={"e": -1, "H+": 1, "H-": -1},
    coulomb_pairs="all",  # or [("e", "e"), ("e", "H+"), ...]
    reactions=[
        plasmacoll.ChargedReaction(
            "mutual_neutralization", ("H-", "H+"), rate_coefficient=4e-13
        ),
    ],
    coulomb_logarithm=plasmacoll.coulomb_logarithm(n_e, T_e_ev),
    seed=1,
)
charged.collide_species(species, dt, cell_volume=dx * area, cell_size=dx)
```

`cell_size` is the size of the pairing cells in the units of the positions (one cell for all
markers if omitted), and `cell_volume` the volume of one cell in m³, which sets the densities.
`ZeroDReactor(..., charged=charged)` uses the reactor volume as a single cell.

## Pairing

Pairs follow Takizuka and Abe (1977). Within one species, the markers of a cell are shuffled and
paired; an odd count leaves a triple that collides as three pairs of half the time step. Between
two species, every marker of the more numerous one in a cell gets a partner of the other, which is
reused cyclically. Each side's scattering is accepted with the probability that gives it its
partner's density on average (Nanbu and Yonemura 1998), and a reused marker receives the sum of
its kicks. Pairing runs on the host; the kinematics run on the active backend.

## Coulomb scattering

Each pair's relative velocity $\mathbf u$ is rotated by an angle $\chi$ with $\tan(\chi/2)$ drawn
from a normal distribution of variance

$$\langle\delta^2\rangle = \frac{q_a^2 q_b^2\, n\, \ln\Lambda\, \Delta t}{8\pi\varepsilon_0^2\,\mu^2\,u^3},$$

and the velocities change by $\pm(\mu/m)\Delta\mathbf u$. This conserves momentum exactly, and
energy exactly for markers that are not reused (on average otherwise). `coulomb_logarithm(n_e,
T_e)` evaluates the NRL formulary's electron–ion $\ln\Lambda$; one value is used for every pair.

### Accuracy and the time step

Binary-collision methods converge slowly in $\Delta t$, and the right small parameter is the
collision frequency of slow pairs, not the energy-exchange rate. Measured against the Spitzer
temperature-relaxation rate $\nu_\varepsilon$ of two ion species started as exact Maxwellians:

| $\nu_\varepsilon\Delta t$ | initial rate / Spitzer | temperature change after $t = 1/\nu_\varepsilon$ / Spitzer |
| ------------------------- | ---------------------- | ----------------------------------------------------------- |
| $10^{-2}$                 | about 0.8              | about 0.82                                                  |
| $10^{-3}$                 | about 0.93             | about 0.87                                                  |
| $10^{-4}$                 | about 0.93             | not measured                                                |

The kernel itself is checked separately: a single step reproduces the analytic drag on a fast beam
to within 1 %, and the energy exchange per step matches the model's own expectation to 1 %. Part
of the gap is the known slow convergence of binary methods (the model's own expected exchange from
Maxwellians is 0.81 of Spitzer's at $10^{-2}$ and 0.94 at $10^{-3}$); the rest, 5–10 %, has not been
traced to a cause. Treat Coulomb relaxation rates as accurate to 10–15 % at
$\nu_\varepsilon\Delta t \lesssim 10^{-3}$, and compare with
[tutorial 7](/plasmacoll/tutorials/07-charged-collisions/). Keep like-species collisions on
(`coulomb_pairs="all"`): without them the distributions drift away from Maxwellians and the
exchange slows further.

## Reactions

| Kind                    | Reactants `species`         | Products                                                               |
| ----------------------- | --------------------------- | ---------------------------------------------------------------------- |
| `mutual_neutralization` | (negative ion, positive ion) | optional `neutral_1`, `neutral_2` with their parents' velocities       |
| `recombination`         | (electron, positive ion)    | optional `neutral` at the pair's centre of mass (radiative)            |
| `electron_detachment`   | (electron, negative ion)    | two `electron`s sharing the energy above `energy_loss`; optional `neutral` |

Give a constant `rate_coefficient` (m³/s) or a `cross_section` against the centre-of-mass energy.
A pair reacts with probability $K n_a n_b V\Delta t / S$, where $S$ sums the smaller weight of every
pair in the cell. That gives the expected number of physical reactions $K n_a n_b V\Delta t$ for any
marker weights. A reaction removes the smaller weight from both markers and creates every outgoing
particle as a new marker of that weight, so particle number and charge are conserved exactly. Each
marker reacts at most once per step; `ChargedDiagnostics.max_reaction_probability` reports the
largest pair probability, and the operator warns once above `max_reaction_probability` (0.1).

Partly consumed markers have their weight reduced in place, so a custom container's `weights`
must be a writable view of its storage.
