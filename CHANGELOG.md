# Changelog

All notable changes to this project are documented in this file and maintained manually. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Opt-in scope-profiler regions around the collision operator, the reactor, marker merging and
  splitting, rate coefficients, charge-state fractions and file reading, off unless the
  `plasmacoll` tag is enabled (e.g. `scope_profiler.setup(enable_tags=["plasmacoll"])`). Adds the
  `scope-profiler>=0.7.5` dependency.

### Changed

- The combined coverage of the NumPy and the fake CuPy runs must be at least 95% (was 100%).

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
