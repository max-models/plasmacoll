# Changelog

All notable changes to this project are documented in this file and maintained manually. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `plasmacoll.units.unit_metadata`: a unit field without a default, as
  `dataclasses.field(metadata=unit_metadata("m"))`, which type checkers read as required (they
  read `unit_field("m")` as a default). `array=True` on both keeps an array an array.
- `to_array` converts mappings value by value and nested lists of quantity strings, the
  configuration-file form of a 2D array.
- The Townsend: `plasmacoll.units.Td` and `"80 Td"` in strings (astropy's own `Td` is a
  tera-day, so the unit is called `townsend` there). `solve_eedf` and `swarm_table` take it.
- `to_unit` and `to_array` take a bare unit as a scale (`u.keV`), complex values, and `to_unit`
  CuPy arrays.

### Changed

- The figures of the docs are interactive Plotly figures in the site's colours, drawn with
  plasma-plots (now in the `docs` extra), one figure per
  plot and with their formulas typeset as LaTeX.
- `convert_units` skips fields that are already floats and caches each class's unit fields:
  building a `NeutralBackground` of plain numbers takes 2.6 times less time.

## [0.5.0] - 2026-11-03

### Added

- `plasmacoll.units`: astropy quantities in the setup of a model. A setting given as a Quantity
  (`0.1 * u.ns`), a string (`"0.1 ns"`, as in a configuration file) or a number in its documented
  unit is stored as a plain number in that unit; a wrong dimension raises, and K, eV and J convert
  into each other. `unit_field` and `convert_units` for dataclasses, `to_unit`, `to_array`,
  `with_unit` and `rate_to_si`. astropy is imported only when a Quantity or a string is given
  (the extra `plasmacoll[units]`). Moved from picnic.
- The setup of plasmacoll takes quantities: `NeutralBackground`, `DensityProfile`,
  `CollisionProcess`, `CrossSection` and its constructors, `ZeroDReactor`, the beam-gas
  reactions, the process builders (argon, metals, caesium, hydrogen), `rate_coefficient`,
  `RateTable`, `solve_eedf` and `swarm_table`, `VibrationalDistribution`, the charge states and
  Rudd's cross sections. A tutorial, "Units".
- Superelastic collisions in `solve_eedf`: an excitation with `upper_state` (and
  `statistical_weight_ratio`, new `CollisionProcess` fields) also has its inverse, by detailed
  balance; its rate is `"<name> (superelastic)"` and its power `power["superelastic"]`.
  `MonteCarloCollisions` rejects such processes.
- `states` of `solve_eedf` and `swarm_table`: vibrational or rotational levels of a gas with their
  populations, which processes may name as their background or upper state.
- The two-term solver checked against LoKI-B++ (`test_boltzmann_lokib.py`, reference in
  `tests/unit/data/lokib_reference.json`, made by `tests/unit/_lokib_models.py`): temporal and
  spatial growth, equal and Opal-Peterson-Beaty sharing, an H2-like gas with rotational and
  vibrational levels and superelastic collisions; within LoKI-B's own discretisation error.
- A Numerics section in the docs (header and sidebar): an overview of the model's equations
  (`docs/src/content/docs/numerics/index.mdx`, with the main cards) and one page per module in
  the order of the model (starlight-theme's `methodcard: { pages: true }`, theme 0.7.4).
- Method cards (methodcard) on the equations of the model: the null-collision step (frequencies,
  bound, candidates, forced collisions, attenuation, scattering angles), the collision
  kinematics, the backgrounds and density profiles, the cross-section tables, the rate
  coefficients, the two-term Boltzmann solver and its moments, the charge-state fractions, the
  beam-gas reactions, the cross-section fits (Janev, Johnson, Vriens-Smeets, Park, Rudd,
  Phelps, Lotz, Langevin), the vibrational populations, merging and splitting and the reactor,
  with their assumptions and references. `test_method_cards.py` checks them: SymPy rules for 35
  closed forms and fits, the observed order of seven discretisations (table and profile
  interpolation, the Maxwellian and EEDF rate integrals, the rate table, the Boltzmann solver
  and Rudd's moments), and the cards of the docs against those of the code. Adds the
  `methodcard` dependency.
- The docs show the cards: a "Numerics" page and a section on each method's API page.
- `vriens_smeets_excitation_rate` and `vriens_smeets_ionization_rate` in
  `plasmacoll.atomic_hydrogen`: the Maxwellian rate coefficients of Vriens and Smeets (1980) for
  H(n) at any `n`, beyond the n = 24 where Johnson's excitation formula turns negative.
- The reactions of a fast hydrogen or deuterium beam with H₂ gas (`hydrogen_beam_on_h2`,
  `BeamGasReactions`): H⁻ stripping and H⁺ capture from Janev, Reiter and Samm (2003), eqs. (159)
  and (144)–(145) (also in `JANEV_FITS`), gas ionization by protons from Rudd's model, and labelled
  placeholders for the rest; as charge-state cross sections at one energy and as MCC processes.
- A process kind `beam_ionization`: a fast projectile ionizes the target and keeps its velocity,
  the target ion starts at rest and the electron with a few times the binding energy (the
  backstreaming ions of an accelerator).
- `collide_now` and `attenuation` for event-driven collisions.
- Caesium: electron-impact ionization and Cs/Cs+ collisions on H2.
- The neutrals each collision step consumes, per background.

### Fixed

- The docstrings render on the API pages: code spans instead of reST roles such as `:func:`.
- `merge_markers`: the docstring gives the group size the code uses, `max(3, ceil(2 N / target))`.
- `solve_eedf`: spatial growth converges with strong ionization (a secant step on `alpha/N`) and
  on its own grid search (done with temporal growth, then spatial growth on the chosen grid);
  the exponent of `F0` in a cell is van Leer's mean of the one-sided slopes, so that a first
  iterate with empty cells no longer blows up the collision integrals (1000 Td on 500 cells
  did not converge).

## [0.4.0] - 2026-10-28

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
