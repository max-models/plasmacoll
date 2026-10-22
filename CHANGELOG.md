# Changelog

All notable changes to this project are documented in this file and maintained manually. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `plasmacoll.janev`: the hydrogen cross-section fits of Janev, Reiter and Samm (Jül-4105, 2003)
  for e + H₂ vibrational and electronic excitation, ionization and dissociative ionization,
  dissociative attachment over an H₂(v) population, e + H⁻ detachment, H⁺ + H⁻ mutual
  neutralization and H⁻ + H detachment, checked against the report and HYDHEL.
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
