---
title: Cross sections
description: Tables, LXCat files, functions, Maxwell molecules and vibrational mixtures.
sidebar:
  order: 3
---

A `CrossSection` holds energies (eV, non-decreasing) and cross sections (m², non-negative). It is
evaluated by linear interpolation, which is the convention of the Turner et al. (2013) benchmark.
Beyond the last point the last value is used, and below `threshold` the result is zero.

```python
table = plasmacoll.CrossSection(
    energy=[0.0, 10.0, 100.0], sigma=[1e-20, 3e-20, 1e-20], threshold=5.0
)
table([2.0, 50.0])  # array([0.0, 2.11e-20])
```

## Sources

| Constructor                      | Use                                                                                 |
| -------------------------------- | ----------------------------------------------------------------------------------- |
| `CrossSection.from_table(path)`  | two-column text files; `delimiter`, `energy_unit` and `sigma_unit` convert the columns |
| `plasmacoll.read_lxcat(path)`         | every process block of an LXCat file, as `LXCatProcess` with the table and comments |
| `CrossSection.from_function(f, energies)` | an analytic fit sampled on a grid                                          |
| `CrossSection.constant(sigma)`   | a constant cross section above an optional threshold                                |
| `CrossSection.maxwell_molecule(K, mass)` | $\sigma = K/g$, a speed-independent frequency $nK$ for benchmarks          |
| `CrossSection.vibrational_mixture(levels, populations)` | population-weighted mean of level-resolved tables            |

`find_lxcat_process(processes, wanted)` picks an LXCat block by its kind (`"IONIZATION"`) or by
its full `PROCESS` comment. For excitation and ionization blocks the threshold is the block's
parameter line; for elastic blocks the parameter is the mass ratio $m_e/M$
(`LXCatProcess.mass_ratio`).

The ion-neutral blocks of the Phelps database (`Ar+ + Ar`) have no keyword, target or parameter
line. `read_lxcat` starts them at their `SPECIES:` line and takes the kind from the end of the
`PROCESS` line: `ISOTROPIC` or `BACKSCAT`. The target is the neutral (`Ar`) and the parameter the
mass ratio $M_i/M$. LXCat tabulates them against the centre-of-mass energy.

## Effective cross sections

Many LXCat sets give an `EFFECTIVE` cross section instead of an `ELASTIC` one: the elastic
momentum-transfer cross section **plus every inelastic cross section** of the set. Used directly
as an `elastic` process next to the inelastic processes, it counts their momentum transfer twice.
Subtract them first:

```python
processes = plasmacoll.read_lxcat("Ar.txt")
elastic = plasmacoll.elastic_from_lxcat(processes, target="Ar")
```

`elastic_from_lxcat` returns the `ELASTIC` block if the set has one, and otherwise
`CrossSection.elastic_from_effective(effective, inelastic)` of the `EFFECTIVE` block and the
set's excitation, ionization and attachment blocks. The result is tabulated on the energies of
all tables with points on both sides of every threshold, and clipped at zero with a warning
where the set is inconsistent.

The momentum-transfer cross section belongs with isotropic scattering. With an anisotropic
[scattering model](/plasmacoll/guides/collision-processes/#scattering-models), use the integral elastic
cross section instead.

## Data

plasmacoll does not ship tabulated cross-section data: LXCat does not allow third parties to
redistribute its data. Download sets from [LXCat](https://lxcat.net) and cite the databases as
their terms of use require, or use tables from the literature with `from_table`. The tutorials
and tests use synthetic tables and Maxwell molecules, and they say so wherever they appear.
Published analytic fits are included where there is no set to download, or for convenience:
Phelps' Ar⁺ + Ar cross sections and the Lotz ionization of metals, see
[argon and metals](/plasmacoll/guides/argon-and-metals/), and the hydrogen fits of Janev, Reiter and Samm (2003) in `plasmacoll.janev`, see
[Negative-ion sources](/plasmacoll/guides/negative-ion-sources/#hydrogen-cross-sections),
and the H₂ parameters of Rudd's model (below), checked in the tests against the review's own
table.

## Maxwell molecules

For $\sigma = K/g$ every marker collides at $\nu = nK$, independent of its speed. Moments then
relax with closed-form rates. For example, for isotropic elastic scattering

$$\frac{d\langle E\rangle}{dt} = -\frac{2mM}{(m+M)^2}\,\nu\,\left(\langle E\rangle - \tfrac32 kT_g\right),$$

and the number of collisions follows a Poisson process. The test suite and
[tutorial 2](/plasmacoll/tutorials/02-relaxation/) use this to check the kinematics. Pass the reduced
mass for `energy_frame="center_of_mass"` and the incident mass for `"lab"`, so that the table's
energy-to-speed conversion matches the process.

## Vibrational mixtures

`vibrational_mixture` evaluates every level through its own threshold and inserts table points
on both sides of each threshold, so the mixture keeps the steps sharp. The populations need not
be normalized. See [negative-ion sources](/plasmacoll/guides/negative-ion-sources/).

## Secondary electrons of ion impact

`RuddIonization` is Rudd's semi-empirical singly differential cross section $d\sigma/dW$ for
ejecting an electron of energy $W$ from a target by a fast ion (Rudd, Kim, Madison and Gay, Rev.
Mod. Phys. 64, 441, 1992, Eqs. (37)–(48)): a sum over the target's shells (`RuddShell`: binding
energy, occupation and ten fit parameters). It depends on the projectile only through its speed,
so `with_projectile(mass)` turns the proton model into one for deuterons. `total(energy)` and
`mean_energy(energy)` integrate it over $W$: the ionization cross section and the mean energy of
the secondary electrons, the source terms of a plasma that a fast beam ionizes.
`cross_section(energies)` tabulates the total as a `CrossSection` of the projectile energy.

`RUDD_H2` holds the H₂ parameters of the review (Table V, binding energy from Table I(a)); it
reproduces the review's Table VII of recommended H⁺ + H₂ cross sections to 1 %. For 1 MeV
deuterons on H₂ the total is 6.3 × 10⁻²¹ m² and the mean secondary energy 37 eV.
