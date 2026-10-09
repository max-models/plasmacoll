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
| gas transfer                                 | plasma + gas momentum and energy unchanged; consumed gas = ionizations + attachments  | exact     |
| closed volume of hot, drifting ions          | gas and ions reach a common temperature and flow                                     | 5 %       |
| attachment depleting the gas                 | $n_g - n_e$ constant                                                                  | exact     |
| speed classes                                | real collisions unchanged, far fewer null candidates                                 | 5σ        |
| crossed fields                               | $\mathbf E\times\mathbf B$ drift $E/B$                                                | 1 %       |
| rate and Townsend coefficients from counts   | $K(1-e^{-x})/x$ for a Maxwell molecule, $x = s\nu\Delta t$; $\alpha = kN/v_d$           | 2–5 %     |
| Coulomb scattering                           | momentum exact, energy exact without reused markers; drag on a fast beam              | 3 %       |
| Coulomb temperature relaxation               | Spitzer rate, see [its accuracy](/plasmacoll/guides/charged-collisions/#accuracy-and-the-time-step) | 12 % |
| mutual neutralization, unequal weights       | $n_- = \Delta / ((n_{+0}/n_{-0})\,e^{K\Delta t} - 1)$, $n_+ - n_-$ constant          | 2 %       |
| recombination                                | momentum exact                                                                       | exact     |

The drift velocity and the Wannier energy hold exactly for a Maxwell-molecule cross section with
isotropic scattering, which makes the field test a check of the field push and the collision
kinematics together: a swarm benchmark with a closed-form answer. For real gases, compare the
same reactor diagnostics (`mean_velocity`, `mean_energy_ev` and the collision counts) with a
Boltzmann solver such as BOLSIG+ for the same cross-section set.

The capstone check of the whole collision step is the Turner et al. (2013) capacitive discharge
benchmark. `examples/turner_ccp/turner_case1.py` is a minimal 1D3V electrostatic
particle-in-cell code for its case 1 that uses plasmacoll for every collision. It needs the
benchmark's cross-section sets from LXCat, and a full run takes hours, so it is not part of the
test suite; compare its time-averaged ion density with the paper's figures.

The whole suite runs on NumPy and on cunumpy's stand-in for CuPy in CI, and the combined
coverage of the two runs is 100 %. The [tutorials](/plasmacoll/tutorials/) plot several of these
comparisons.
