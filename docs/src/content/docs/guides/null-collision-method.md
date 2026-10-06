---
title: The null-collision method
description: What one collision step does, how the frequency bound is built, and how to choose the time step.
sidebar:
  order: 1
---

`MonteCarloCollisions` implements the null-collision method of Vahedi and Surendra
(Comput. Phys. Commun. **87**, 179 (1995)). A marker of species $s$ with velocity $\mathbf v$ meets
a neutral of background $b$ with velocity $\mathbf w$, drawn from the background's Maxwellian, at
relative speed $g = |\mathbf v - \mathbf w|$. Process $k$ has the collision frequency

$$\nu_k(g) = n_b\,\sigma_k(E)\,g,$$

where $n_b$ is the neutral density and $E$ the energy the cross section is tabulated against
(see [energy frames](/plasmacoll/guides/collision-processes/#energy-frames)).

## One step

1. **Bound.** The operator takes $g_{\max}$ as the fastest marker's speed plus eight thermal
   speeds of the neutrals and looks up a bound $\nu_{\max} \ge \sum_k \nu_k(g)$ for all
   $g \le g_{\max}$.
2. **Candidates.** Every marker becomes a candidate with probability
   $P = 1 - e^{-\nu_{\max}\Delta t}$. On NumPy the number of candidates is drawn from a binomial
   distribution and then the candidates themselves, so the cost scales with the candidates, not
   with all markers. Other generators (CuPy's) draw one number per marker.
3. **Selection.** Each candidate draws a neutral velocity and a uniform number
   $R\,\nu_{\max}$ with $R \in [0, 1)$. It undergoes process $k$ if
   $\sum_{j<k}\nu_j < R\,\nu_{\max} \le \sum_{j\le k}\nu_j$, and a *null* collision (nothing
   happens) if $R\,\nu_{\max}$ exceeds the total.
4. **Kinematics.** The chosen process changes the velocity, removes the marker or creates
   products. See [collision processes](/plasmacoll/guides/collision-processes/).

Real collisions of process $k$ thus happen with probability $P\,\nu_k/\nu_{\max}$ per step. That
is $\nu_k\Delta t$ to first order, independent of the bound.

## The frequency bound

At construction the operator tabulates $\nu(g) = \sum_k\nu_k(g)$ on a logarithmic grid of
`num_bound_samples` speeds, merged with the speeds of every table point and threshold, and stores
its running maximum. Because the cross sections are piecewise linear in energy, the frequency's
local maxima lie at or near those points. `bound_safety` (default 1.02) covers maxima between
grid points. Beyond the last table point the cross sections are constant and $\nu$ grows
linearly with $g$, which the direct evaluation at $g_{\max}$ covers.

The bound must hold. If a candidate's real total frequency exceeds $\nu_{\max}$, the collision rate
is underestimated. The operator counts such candidates in `MCCDiagnostics.bound_violations` and
warns once.

## Density profiles and thinning

A background with a profile has density $n_b\,f(\mathbf x)$. The bound uses the largest
factor $\max f$, and each candidate's real frequency uses the factor at its position. Markers
where the gas is thin are therefore mostly null collisions. This is thinning of a Poisson process,
and it is exact.

A profile is a `DensityProfile` (piecewise linear along one axis) or a
`FunctionProfile(function, max_factor)` of all position components; any object with a
`max_factor` and a call on the positions works. A `FunctionProfile` must not exceed its
`max_factor`; where it does, the candidates are counted as bound violations.

## Flowing and heated gas

`NeutralBackground(..., drift=(ux, uy, uz))` is a gas flowing with that mean velocity, and
`temperature_profile=` scales its temperature in space like a density profile. Neutral partners
are drawn from the local drifting Maxwellian, and the bound covers relative speeds up to the
marker speed plus the flow speed plus eight of the largest thermal speeds.

A background is fixed during a step. To follow gas heating or depletion, build the new
background between steps and pass it to `MonteCarloCollisions.set_background`, which
retabulates the bound.

## Choosing the time step

A marker collides at most once per step, and products do not collide in the step that created
them. Rates are therefore correct to first order in $\nu_{\max}\Delta t$. A population that
ionizes at constant $\nu$ grows by $(1 + P)$ per step instead of $e^{\nu\Delta t}$, which
[tutorial 3](/plasmacoll/tutorials/03-ionization-attachment/) measures. Keep
$\nu_{\max}\Delta t \lesssim 0.1$ (a few percent error on the rate), or collide every $m$ steps
with $m\Delta t$ only if $\nu_{\max} m\Delta t$ stays that small.

The operator checks this: `MCCDiagnostics.collision_probability` reports
$1 - e^{-\nu_{\max}\Delta t}$ of every call, and the operator warns once when it exceeds
`max_collision_probability` (default 0.1).

## Random numbers

Every operator owns a generator. `seed` and `rank` select the stream
`numpy.random.SeedSequence(seed, spawn_key=(rank,))`, so MPI ranks that pass their rank draw
independent numbers from one seed, and a run with the same seed and rank count is reproducible.
On the CuPy backend the stream seeds a CuPy generator. A generator can also be passed directly
with `rng=`.
