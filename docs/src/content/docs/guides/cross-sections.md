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

## Data

plasmacoll does not ship cross-section data. Download sets from [LXCat](https://lxcat.net) and cite
the databases as their terms of use require, or use tables from the literature with
`from_table`. The tutorials and tests use synthetic tables and Maxwell molecules, and they say
so wherever they appear.

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
