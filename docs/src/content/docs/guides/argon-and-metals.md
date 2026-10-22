---
title: Argon and metals
description: Electron and Ar⁺ processes in argon from LXCat sets and Phelps' fits, and Lotz ionization of sputtered Cu and Ti, for magnetron models.
sidebar:
  order: 6
---

A magnetron discharge in argon needs electron-argon collisions, Ar⁺ + Ar scattering and, once the
target is sputtered, the ionization of the metal atoms. `plasmacoll.argon` and
`plasmacoll.metals` build these processes for `MonteCarloCollisions` and `ZeroDReactor`.

## What is included

| Data                                    | Source                                                   | In plasmacoll                                       |
| --------------------------------------- | -------------------------------------------------------- | --------------------------------------------------- |
| e + Ar (Phelps, SIGLO, Biagi, Hayashi, IST-Lisbon) | [LXCat](https://lxcat.net) databases          | no: download the file, then `argon_electron_processes` |
| Ar⁺ + Ar isotropic and backscatter      | Phelps, J. Appl. Phys. 76, 747 (1994), Eqs. (2) and (3)  | yes, analytic fit (`revision="1994"`, default)      |
| Ar⁺ + Ar, later fits                     | Phelps, collision-data pages (2005), private communication | yes, analytic fit (`revision="2005"`)             |
| Ar⁺ + Ar from LXCat                      | LXCat Phelps database (tables of the 1994 fits)          | no: download the file, then `argon_ion_processes`   |
| e + Cu                                  | LXCat SIGLO database (Tkachev et al., Laser Phys. 17, 775 (2007)) | no: download the file, then `read_lxcat`   |
| e + Cu, Cu⁺, Ti, Ti⁺ ionization          | Lotz, Z. Phys. 206, 205 (1967); NIST ionization energies | yes, analytic (`metal_ionization_cross_section`)    |

LXCat's [redistribution policy](https://us.lxcat.net/instructions/redistribution.php) does not
allow third parties to redistribute its data; the contributors keep ownership of their databases.
Phelps' own pages carry no licence. So plasmacoll ships no tables, only the published analytic
fits, and cites them. Cite the databases you download as their file headers ask.

## Electrons in argon

Download a complete argon set from LXCat ("Data center", "Download data", database, e / Ar,
accept the terms) and save the text file. Boeuf's magnetron simulations (arXiv:2211.17051) use
the SIGLO set: Phelps' compilation with every excitation level lumped into one at 11.5 eV, an
`EFFECTIVE` momentum-transfer cross section and ionization from Rapp and Schram with a 15.8 eV
threshold. The Phelps and Biagi sets put the ionization threshold at 15.76 eV
(`plasmacoll.argon.ARGON_IONIZATION_ENERGY` is NIST's 15.7596 eV).

```python
import plasmacoll
from plasmacoll.argon import ARGON_MASS
from plasmacoll.constants import ELECTRON_MASS, kelvin_to_ev

electrons = plasmacoll.argon_electron_processes("Ar_siglo.txt")
ions = plasmacoll.argon_ion_processes()  # Phelps (1994) fits

mcc = plasmacoll.MonteCarloCollisions(
    species_masses={"e": ELECTRON_MASS, "Ar+": ARGON_MASS},
    backgrounds=[
        plasmacoll.NeutralBackground("Ar", 1.4e20, kelvin_to_ev(300.0), ARGON_MASS)
    ],
    processes={"e": electrons, "Ar+": ions},
)
```

`argon_electron_processes` takes the path or the output of `read_lxcat`, and returns:

- `elastic`: the `ELASTIC` block, or the `EFFECTIVE` block less every excitation and ionization
  block (see [effective cross sections](/plasmacoll/guides/cross-sections/#effective-cross-sections)),
  with isotropic scattering;
- one `excitation` per `EXCITATION` block, losing its threshold;
- `ionization` with the products `{"electron": "e", "ion": "Ar+"}` and Opal–Peterson–Beaty energy
  sharing with $w = 10$ eV.

All are looked up at the electron's energy (`energy_frame="lab"`). Only the blocks of `target`
(`"Ar"`) are used, so one file may hold several gases. Every ionization block creates the `ion`
species; drop blocks for multiple ionization from the list if the set has them.

## Ar⁺ in argon

Phelps splits the elastic Ar⁺ + Ar differential cross section into an isotropic part $Q_i$ and a
backward (charge-exchange) part $Q_b$, with the momentum-transfer cross section
$Q_m = Q_i + 2Q_b$. With the laboratory ion energy $\varepsilon$ in eV (twice the centre-of-mass
energy), the 1994 fits are

$$Q_i = \frac{2\times10^{-19}}{\varepsilon^{1/2}(1+\varepsilon)} + \frac{3\times10^{-19}\,\varepsilon}{(1+\varepsilon/3)^{2.3}},\qquad
Q_m = 1.15\times10^{-18}\,\varepsilon^{-0.1}\left(1+\frac{0.015}{\varepsilon}\right)^{0.6}\ \mathrm{m^2},$$

and $Q_b = (Q_m - Q_i)/2$. These are the tables of the LXCat Phelps database. Phelps later revised
them to account for differential-scattering measurements (`revision="2005"`, centre-of-mass energy
$E$ in eV): $Q_i = 21.6\times10^{-20}/E^{1/2}$ and
$Q_b = 52\times10^{-20}/[E^{0.08}(1+0.08/E)(1+E/1000)^{0.3}]$ m².

`argon_ion_processes()` returns an isotropic `elastic` process and a `backscatter` process, both
looked up at the centre-of-mass energy. Backscatter reverses the relative velocity, so the ion
leaves with the neutral's velocity; `neutral="Ar_fast"` keeps the fast neutral as a species.
`argon_ion_cross_sections(revision)` gives the two tables, from $10^{-4}$ to $10^4$ eV. To use a
downloaded LXCat file instead, pass its path: `read_lxcat` reads the database's ion-neutral blocks,
which have no keyword line, as kinds `ISOTROPIC` and `BACKSCAT`.

## Sputtered metals

There are few cross-section sets for metal atoms. LXCat has e + Cu in the SIGLO database, which
`read_lxcat` reads like any other set; Ti has none. Where there is no set, the Lotz formula gives
the ionization cross section as a sum over subshells,

$$\sigma(E) = \sum_i a_i q_i \frac{\ln(E/P_i)}{E P_i}\left[1 - b_i e^{-c_i(E/P_i - 1)}\right],\qquad E \ge P_i,$$

with $q_i$ electrons of binding energy $P_i$. `lotz_cross_section(energy, subshells)` evaluates it
for any `LotzSubshell(binding_energy, electrons, a, b, c)`; with the defaults
$a = 4.5\times10^{-14}$ cm² eV² and $b = c = 0$ it is Lotz's simplified formula.

```python
from plasmacoll.metals import METAL_MASSES

# Cu -> Cu+ from 7.73 eV, and Cu+ -> Cu2+ from 20.29 eV
cu = plasmacoll.metal_ionization_cross_section("Cu")
cu_ion = plasmacoll.metal_ionization_cross_section("Cu", 1)
process = plasmacoll.CollisionProcess(
    "ionization",
    "Cu",
    cu,
    energy_frame="lab",
    products={"electron": "e", "ion": "Cu+"},
)
cu_mass = METAL_MASSES["Cu"]  # for the Cu background and the Cu+ species
```

`metal_ionization_cross_section` takes the outermost subshell of Cu, Cu⁺, Ti and Ti⁺ (4s¹, 3d¹⁰,
4s² and 4s¹) at the ionization energies of the NIST Atomic Spectra Database. The deeper subshells
are left out (the 3d electrons of Cu and Ti), so it is a lower estimate. Pass more subshells to
`lotz_table` when their binding energies are at hand, or use measured cross sections (Freund et
al., Phys. Rev. A 41, 3575 (1990) for Cu) with `CrossSection.from_table`.

Charge exchange of metal ions with argon and with their own atoms is not included.

## Checking a downloaded set

The tests read self-made files in the LXCat layout. With a downloaded SIGLO argon file, one more
test compares its ionization cross section with Rapp and Schram's values at 20, 50 and 100 eV:

```bash
PLASMACOLL_LXCAT_SIGLO_AR=Ar_siglo.txt uv run pytest -k siglo
```
