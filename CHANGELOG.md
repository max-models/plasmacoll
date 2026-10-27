# Changelog

All notable changes to this project are documented in this file and maintained manually. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `plasmacoll.atomic_hydrogen`: electron excitation `n -> m` and ionization of H(n) for any
  principal quantum numbers, from Janev, Reiter and Samm (2003), section 2.1, and Johnson's
  (1972) formulas; Johnson's oscillator strengths; `hydrogen_atom_processes` for the MCC.
- `eedf_rate_coefficient`: rate coefficients over any tabulated electron energy distribution.
- A process kind `stripping` for the charge changes of a fast beam (H⁻ → H, H⁻ → H⁺, H → H⁺ and
  the capture H⁺ → H): the incident marker moves to its product species at its own velocity, with
  the released electrons at the projectile's velocity and the target ion of a capture as optional
  products. `stripping_processes` builds these processes from a `ChargeStateTransitions`, so the
  markers and `charge_state_fractions` use the same cross sections; over many markers the
  populations follow the fractions along the path.

- Argon and metal process sets for magnetron models (`plasmacoll.argon`, `plasmacoll.metals`):
  `argon_electron_processes` builds the electron-argon processes of a downloaded LXCat set,
  `argon_ion_processes` the isotropic and backscatter Ar+ + Ar processes from Phelps' 1994 or
  2005 fits or the LXCat Phelps database, and `metal_ionization_cross_section` and
  `lotz_cross_section` the Lotz ionization of Cu, Ti and their ions. No LXCat data is included,
  as LXCat does not allow redistribution.
- `read_lxcat` reads the ion-neutral blocks of the LXCat Phelps database (kinds `ISOTROPIC` and
  `BACKSCAT`), which have no keyword line and were skipped before.
- `plasmacoll.janev`: the hydrogen cross-section fits of Janev, Reiter and Samm (Jül-4105, 2003)
  for e + H₂ vibrational and electronic excitation, ionization and dissociative ionization,
  dissociative attachment over an H₂(v) population, e + H⁻ detachment, H⁺ + H⁻ mutual
  neutralization and H⁻ + H detachment, checked against the report and HYDHEL.
- `plasmacoll.hydrogen_elastic`: the elastic e + H and e + H₂ cross sections of Park et al.
  (Plasma Sources Sci. Technol. 31, 065013, 2022), closed-form fits to convergent close-coupling
  calculations: the integral cross section, the forward/backward angular distribution and the
  momentum-transfer cross section for isotropic scattering and `solve_eedf`. Checked against the
  swarm-derived H₂ cross section of England et al. (1988), the beam data of Brunger et al. (1991)
  and the CCC e + H database.
- `RuddIonization` and `RuddShell`: Rudd's singly differential cross section for the secondary
  electrons of ionization by fast ions (Rev. Mod. Phys. 64, 441, 1992), its integrals over the
  secondary energy (`total`, `mean_energy`, `cross_section`) and any projectile mass at equal
  speed. `RUDD_H2`, the review's H2 parameters, reproduces its Table VII to 1%.
- A two-term Boltzmann solver for electrons in a DC field, after Hagelaar and Pitchford (2005):
  `solve_eedf` returns the energy distribution, mean energy, `mu N`, `D N`, the rate coefficient
  of every process and `alpha / N` for the same `CollisionProcess` and `NeutralBackground`
  objects the Monte Carlo collisions use, with temporal or spatial growth for ionization and
  attachment. `swarm_table` tabulates the results against the mean energy for fluid and global
  models. Checked against the Maxwell and Druyvesteyn limits, the Reid ramp and Lucas–Saelee
  model gases and BOLOS.
- `plasmacoll.constants.TOWNSEND`, 1 Td in V m².

## [0.3.0] - 2026-10-21

### Added

- Opt-in scope-profiler regions around the collision operator, the reactor, marker merging and
  splitting, rate coefficients, charge-state fractions and file reading, off unless the
  `plasmacoll` tag is enabled (e.g. `scope_profiler.setup(enable_tags=["plasmacoll"])`). Adds the
  `scope-profiler>=0.7.5` dependency.

### Changed

- The combined coverage of the NumPy and the fake CuPy runs must be at least 95% (was 100%).
- **Breaking:** every temperature is ``kT`` in eV, as the rate coefficients already were:
  `NeutralBackground.temperature`, the `temperature` of `ParticleArrays.maxwellian`, and
  `ParticleArrays.temperature()` and the reactor's `temperature` (were K). Convert with
  `plasmacoll.constants.kelvin_to_ev`.

## [0.2.0] - 2026-10-20

### Added

- Scattering models, dissociative processes, marker merging (`merge_markers`), flowing gas
  (`DynamicProfile`) and a field-driven `ZeroDReactor`.
- Rate tables (`RateTable`) and charge-state transitions in a gas
  (`ChargeStateTransitions`, `charge_state_fractions`), used by wellbalanced and freeofcharge.
- Collisions with a plasma-species target.

### Changed

- The docs use the shared max-models starlight-theme.

## [0.1.0] - 2026-10-05

### Added

- Package layout with a console entry point and pytest tests.
- Astro + Starlight documentation site with executed tutorials and a generated API reference.
- GitHub Actions for tests, ruff, pyright, tutorials, documentation and PyPI publishing.
