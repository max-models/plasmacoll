---
title: Validation
description: The closed-form results the test suite checks the operator against.
sidebar:
  order: 6
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

The whole suite runs on NumPy and on cunumpy's stand-in for CuPy in CI, and the combined
coverage of the two runs is 100 %. The [tutorials](/pymcc/tutorials/) plot several of these
comparisons.
