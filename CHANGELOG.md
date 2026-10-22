# Changelog

All notable changes to this project are documented in this file and maintained manually. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Argon and metal process sets for magnetron models (`plasmacoll.argon`, `plasmacoll.metals`):
  `argon_electron_processes` builds the electron-argon processes of a downloaded LXCat set,
  `argon_ion_processes` the isotropic and backscatter Ar+ + Ar processes from Phelps' 1994 or
  2005 fits or the LXCat Phelps database, and `metal_ionization_cross_section` and
  `lotz_cross_section` the Lotz ionization of Cu, Ti and their ions. No LXCat data is included,
  as LXCat does not allow redistribution.
- `read_lxcat` reads the ion-neutral blocks of the LXCat Phelps database (kinds `ISOTROPIC` and
  `BACKSCAT`), which have no keyword line and were skipped before.

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
