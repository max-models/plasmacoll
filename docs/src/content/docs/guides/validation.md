---
title: Validation
description: The closed-form results the test suite checks the operator against.
sidebar:
  order: 7
---

Besides unit tests of every kind's kinematics (exact energy loss, energy sharing, isotropy,
products), the test suite checks the operator in a 0D reactor against closed-form solutions. With
Maxwell-molecule cross sections the frequency $\nu = nK$ is constant. The method then collides a
marker within one step with probability $p = (1 - e^{-s\nu\Delta t})/s$, where $s$ is the bound
safety factor, and the expectations are exact functions of the step count:

| Test                                         | Expectation                                                                          | Tolerance |
| -------------------------------------------- | ------------------------------------------------------------------------------------ | --------- |
| collided fraction in one step                | $(1 - e^{-\nu_{\max}\Delta t})\,\nu/\nu_{\max}$                                       | 5σ        |
| attachment, electron survival                | $(1 - p)^n$                                                                          | 2 %       |
| ionization, electron growth                  | $(1 + p)^n$, and $n_e - n_i$ constant                                                 | 2 %       |
| equal-mass elastic, mean energy              | $\tfrac32kT_g + (E_0 - \tfrac32kT_g)\,e^{-\nu_{\rm eff}t/2}$                           | 2 %       |
| resonant charge exchange of a drifting beam  | $T_g + (T_0-T_g)f + \frac{Mu_0^2}{3k}f(1-f)$, $f = (1-p)^n$                            | 2 %       |
| thermal ions on a gas at the same temperature | mean energy unchanged (detailed balance)                                            | 1 %       |
| cold ions in a 300 K gas                     | relax to 300 K and stay there                                                        | 2–3 %     |
| density profile                              | collision probability proportional to the local density, zero where it is zero       | 5–8 %     |
| ions in a flowing, heated gas                | relax to the gas's flow velocity and local temperature                               | 5 %       |
| ions in a uniform field $E$                  | drift $v_d = qE/(\mu\nu)$ and Wannier's mean energy $\tfrac32kT_g + \tfrac12(m+M)v_d^2$ | 3 %       |
| marker merging                               | weight, momentum and kinetic energy unchanged                                        | exact     |
| ion-impact ionization                        | energy and momentum conserved up to the electron's $O(\sqrt{m_e/M})$ terms             | exact     |
| Opal–Peterson–Beaty sharing                  | mean secondary energy $w\ln(1+(R/2w)^2)/(2\arctan(R/2w))$                            | 2 %       |

The drift velocity and the Wannier energy hold exactly for a Maxwell-molecule cross section with
isotropic scattering, which makes the field test a check of the field push and the collision
kinematics together: a swarm benchmark with a closed-form answer. For real gases, compare the
same reactor diagnostics (`mean_velocity`, `mean_energy_ev` and the collision counts) with
`solve_eedf` for the same processes.

## The two-term Boltzmann solver

`solve_eedf` is checked against closed forms, against published two-term results for model
gases, against [LoKI-B](https://github.com/LoKI-Suite/LoKI-B-cpp) and, where it is installed,
against [BOLOS](https://github.com/aluque/bolos), two independent implementations of the method:

| Test                                          | Expectation                                                                   | Agreement   |
| --------------------------------------------- | ----------------------------------------------------------------------------- | ----------- |
| constant collision frequency                  | Maxwellian at $kT + M(E/N)^2\gamma^2/(6m_eK^2)$, $\mu N = e/(m_eK)$            | 0.2 %       |
| no field                                      | Maxwellian at the gas temperature                                             | 0.1 %       |
| constant cross section, cold gas              | Druyvesteyn distribution and its mean energy                                  | $10^{-8}$   |
| Reid ramp, 1–40 Td                            | mean energy, drift velocity and $DN$ of Reid (1979) and Flynn et al. (2024)   | 0.1 %       |
| Lucas–Saelee ionization gas, 30 Td            | the same and $k_{iz}$, Flynn et al. (2024), two-term                          | 0.2 %       |
| synthetic gas with every process, 30–300 Td   | mean energy, $\mu N$, $DN$ and the rates of BOLOS                             | 0.2 %       |
| argon-like gas, 100–1000 Td, 12 cases         | LoKI-B: temporal and spatial growth, equal and Opal–Peterson–Beaty sharing    | 0.01–0.1 %  |
| H2-like gas with levels, 1–100 Td             | LoKI-B: rotational and vibrational levels, superelastic collisions            | 0.01–0.03 % |
| levels at the gas temperature, no field       | Maxwellian at $kT$; $x_l k = x_u k^\mathrm{sup}$ for every level pair         | $10^{-6}$   |
| electron number                               | every column of the converged operator sums to zero                           | $10^{-9}$   |
| energy                                        | power from the field equals the collisional losses, second order in the cell  | $10^{-4}$   |

The Reid ramp gas has a constant elastic cross section of $6\,\text{Å}^2$ and an excitation of
$10(\varepsilon - 0.2)\,\text{Å}^2/\mathrm{eV}$, with $M = 4$ u in a cold gas
(Reid, Aust. J. Phys. 32, 231 (1979)). The Lucas–Saelee gas has
$\sigma_m = 4\varepsilon^{-1/2}\,\text{Å}^2$ and an ionization of
$0.1(\varepsilon - 15.6)\,\text{Å}^2/\mathrm{eV}$ with $m_e/M = 10^{-3}$, as restated by Flynn
et al., J. Phys. D 57, 255204 (2024). At 1000 Td BOLOS on the same grid differs by 1 %, and
converges to the result of `solve_eedf` as its grid is refined.

The LoKI-B reference was computed with LoKI-B++ (the C++ version of the LisbOn KInetics
Boltzmann solver, Tejero-del-Caz et al., Plasma Sources Sci. Technol. 28, 043001 (2019); version
1.0, commit `add5518`) on synthetic cross sections, since LXCat's data may not be redistributed:

- an argon-like gas (elastic, an excitation at 11.5 eV, ionization at 15.75 eV, attachment
  around 6 eV) at 100, 300 and 1000 Td, with LoKI-B's temporal and spatial growth models and its
  `equalSharing` and `usingSDCS` (Opal–Peterson–Beaty, $w = 10$ eV) ionization operators;
- an H2-like gas at 300 K: rotational levels $J = 0$–9 of $v = 0$ with H2's rotational constant
  ($2^{-7}$ eV) and nuclear-spin weights, Gerjuoy–Stein $J \to J + 2$ excitations, a vibrational
  excitation $v = 0 \to 1$ (0.5 eV) with $v = 1$ populated at 2000 K, an electronic excitation
  and ionization, at 1, 10, 30 and 100 Td. Every rotational and the vibrational excitation is
  reversible; the superelastic collisions raise the mean energy at 1 Td by 40 %.

LoKI-B is first order in its energy step, and its results on 4096 and 2048 cells differ by
$10^{-5}$ to $10^{-3}$. On LoKI-B's grid with 1000 cells, `solve_eedf` agrees with it to within
twice that difference (and with LoKI-B extrapolated to a zero step to about $10^{-4}$), for the
mean energy, $\mu N$, $DN$, the effective Townsend coefficient of spatial growth, the net
ionization of temporal growth, and every inelastic and superelastic rate above
$10^{-25}\,\mathrm{m^3/s}$; the distributions agree to 0.5 % over six decades. On its default
grid (200 cells) it is within 0.2 %, 0.6 % at 1000 Td. LoKI-B's Townsend coefficient is
$k_{iz}/w$ in temporal growth, so the comparison there is of the net ionization rate. The
[example](/plasmacoll/examples/07-two-term-boltzmann/) prints the Reid comparison.

The whole suite runs on NumPy and on cunumpy's stand-in for CuPy in CI, and the combined
coverage of the two runs is 100 %. The [examples](/plasmacoll/examples/) plot several of these
comparisons.
