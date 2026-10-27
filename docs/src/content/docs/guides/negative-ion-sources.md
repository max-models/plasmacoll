---
title: Negative-ion sources
description: Proton transport in hydrogen and the vibrational population behind H⁻ volume production.
sidebar:
  order: 5
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

The level-resolved attachment cross sections of Janev, Reiter and Samm are in
`plasmacoll.janev` (next section); other data are read with `CrossSection.from_table`.
[Tutorial 5](/plasmacoll/tutorials/05-hydrogen-negative-ions/) runs the pipeline with synthetic
levels.

## Hydrogen cross sections

`plasmacoll.janev` implements the analytic fits of R. K. Janev, D. Reiter and U. Samm,
"Collision Processes in Low-Temperature Hydrogen Plasmas", Forschungszentrum Jülich report
Jül-4105 (2003), [openly available](https://eirene.de/Documentation/report_4105.pdf), for the
electron collisions with H₂ and the collisions of H⁻:

| Keys                                                                | Reaction                         | Report              |
| ------------------------------------------------------------------- | -------------------------------- | ------------------- |
| `e_h2_vibrational_1`, `_2`                                          | e + H₂(v=0) → e + H₂(v=1, 2)     | eq. (74)            |
| `e_h2_singlet_B`, `_Bprime`, `_Bdoubleprime`, `_C`, `_D`, `_Dprime` | e + H₂(X) → e + H₂(¹Σᵤ, ¹Πᵤ)     | eq. (80), table 15  |
| `e_h2_singlet_EF`, `_HH`, `_GK`, `_I`                               | e + H₂(X) → e + H₂(¹Σ_g, ¹Π_g)   | eq. (81), table 16  |
| `e_h2_triplet_a`, `_b`, `_c`, `_e`, `_d`                            | e + H₂(X) → e + H₂(³Λ) (b: → 2H) | eq. (95), table 22  |
| `e_h2_ionization`                                                   | e + H₂ → 2e + H₂⁺                | eq. (106)           |
| `e_h2_dissociative_ionization`                                      | e + H₂ → 2e + H⁺ + H             | eqs. (107a, b)      |
| `dissociative_attachment(distribution)`                             | e + H₂(v) → H⁻ + H, v = 0..14    | eq. (124), table 27 |
| `e_hminus_detachment`                                               | e + H⁻ → 2e + H                  | eq. (58)            |
| `mutual_neutralization`                                             | H⁺ + H⁻ → H(n=2, 3) + H          | eq. (63), table 10  |
| `hminus_h_associative_detachment`, `hminus_h_detachment`            | H⁻ + H → H₂ + e, → 2H + e        | eq. (71), table 11  |

```python
from plasmacoll.janev import JANEV_FITS, dissociative_attachment
from plasmacoll.vibrational import VibrationalDistribution

fit = JANEV_FITS["e_h2_singlet_B"]
fit(50.0)  # m^2 at 50 eV
table = fit.cross_section()  # a CrossSection with the threshold, for a process
attachment = dissociative_attachment(
    VibrationalDistribution.h2_ground_state_from_kelvin(5000.0)
)
```

Electron collisions with H₂ are fitted against the electron energy (`energy_frame="lab"`), the
collisions of H⁻ against the centre-of-mass energy. The polynomial fits of eqs. (63) and (71)
hold from 0.1 eV to 20 keV; the tables keep their end values beyond. The tests check every fit
against its equation evaluated by hand, against values printed in the report and in HYDHEL, and
the Maxwellian rate coefficients against HYDHEL's rate fits. Two points to know: the report gives
eqs. (63) and (71) against E_CM while HYDHEL, with the same coefficients, uses an energy scale of
1 u (twice E_CM), and the prefactor of the attachment rate formula, eq. (125), is 2.95 times
that of the Maxwellian average of eq. (124), so rates are computed from the cross sections.

The report leaves out elastic collisions. `plasmacoll.hydrogen_elastic` has them for e + H and
e + H₂: the closed-form fits of R. M. Park et al., "Anisotropic angular scattering models of
elastic electron-neutral collisions for Monte Carlo plasma simulations", Plasma Sources Sci.
Technol. 31, 065013 (2022) ([accepted manuscript](https://www.osti.gov/servlets/purl/2318946)),
to the convergent close-coupling (CCC) calculations of the Curtin and Los Alamos groups. They
give the integral elastic cross section (eq. (26), table II), a forward plus backward
screened-Coulomb angular distribution (eqs. (17), (19), (20), table I) and from the two the
momentum-transfer cross section in closed form (eqs. (28)–(30)), the one to use with isotropic
scattering and in `solve_eedf`:

```python
from plasmacoll import CollisionProcess
from plasmacoll.hydrogen_elastic import ELASTIC_FITS

fit = ELASTIC_FITS["H2"]  # or "H"
fit.momentum_transfer(2.0)  # m^2 at 2 eV
elastic = CollisionProcess(
    "elastic", "H2", fit.cross_section(), name="e_h2_elastic", energy_frame="lab"
)
```

The fits reproduce the CCC data from 0.001 to 500 eV for H and from 0.1 to 250 eV for H₂. The
tests check them against independent data:

| Check                                                          | Reference                                   | Agreement             |
| -------------------------------------------------------------- | ------------------------------------------- | --------------------- |
| e + H₂ momentum transfer, 0.02–5 eV                            | England, Elford, Crompton 1988 (swarm)      | −5 to +10 %           |
| e + H₂ integral and momentum transfer, 1–5 eV                  | Brunger et al. 1991 (beam, ±20 %)           | fit 3 to 15 % higher  |
| e + H integral cross section, 0–1000 eV                        | CCC database (atom.curtin.edu.au)           | within 3 %            |
| e + H momentum transfer at 0 eV, π(a_s² + 3a_t²)               | Schwartz 1961 scattering lengths            | fit 5 % higher        |

Below 0.02 eV the H₂ fit exceeds the swarm value (by 23 % at zero energy, outside the CCC
range). Above a few hundred eV the momentum-transfer cross section falls too slowly: for H it is
2.2 times the first Born approximation at 1 keV, where inelastic collisions dominate anyway.
The swarm-derived H₂ cross section is only reliable below 5 eV (Brunger et al. 1991): at 10 eV
it is 7.6 Å² against 5.3 Å² from the CCC fit.
