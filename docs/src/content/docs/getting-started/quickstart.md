---
title: Quickstart
description: Collide electrons with a neutral gas in a few lines.
---

Define the cross sections, the neutral gas and the processes, build a `MonteCarloCollisions`
operator, and collide markers with it:

```python
import plasmacoll
from plasmacoll.constants import ATOMIC_MASS, ELECTRON_MASS, ev_to_kelvin

argon_mass = 39.95 * ATOMIC_MASS
gas = plasmacoll.NeutralBackground(
    "Ar", density=1e21, temperature=300.0, mass=argon_mass
)

elastic = plasmacoll.CrossSection.constant(1e-19)  # m^2
ionization = plasmacoll.CrossSection.from_table("ionization.dat", threshold=15.76)

mcc = plasmacoll.MonteCarloCollisions(
    species_masses={"e": ELECTRON_MASS, "Ar+": argon_mass},
    backgrounds=[gas],
    processes={
        "e": [
            plasmacoll.CollisionProcess("elastic", "Ar", elastic, energy_frame="lab"),
            plasmacoll.CollisionProcess(
                "ionization",
                "Ar",
                ionization,
                energy_frame="lab",
                products={"electron": "e", "ion": "Ar+"},
            ),
        ]
    },
    seed=1,
)
```

## Collide arrays

`collide` takes the arrays of one species, positions `(N, ndim)`, velocities `(N, 3)` in m/s
and weights `(N,)`, and a time step in seconds:

```python
result = mcc.collide("e", positions, velocities, weights, dt=1e-11)
# the velocities after the step
result.velocities
# markers consumed by attachment, detachment and charge transfer
result.removed
# new markers: a list of NewMarkers(positions, velocities, weights)
result.created["Ar+"]
# real collisions per process
result.diagnostics.counts
```

## Collide marker sets

`collide_species` updates every species in place. It removes consumed markers and adds the
products once all species have collided:

```python
electrons = plasmacoll.ParticleArrays.maxwellian(
    100_000, ELECTRON_MASS, ev_to_kelvin(5.0), seed=2
)
ions = plasmacoll.ParticleArrays.empty()
diagnostics = mcc.collide_species({"e": electrons, "Ar+": ions}, dt=1e-11)
```

## A 0D reactor

`ZeroDReactor` repeats that step and records densities, mean energies, temperatures and
collision counts as arrays over time:

```python
reactor = plasmacoll.ZeroDReactor(mcc, {"e": electrons}, volume=1e-6)
history = reactor.run(dt=1e-11, num_steps=1000, every=10)
history.density["Ar+"]  # m^-3, one value per recorded step
```

With `electric_field=(Ex, Ey, Ez)` (V/m) and `charges={"e": -1, "Ar+": 1}` the reactor is a swarm
experiment: every step accelerates the charged species before colliding them, and
`history.mean_velocity` holds the drift velocities. `max_markers=N` merges a species back to
$N/2$ markers whenever ionization grows it beyond $N$.

Next: [the null-collision method](/plasmacoll/guides/null-collision-method/), or the
[tutorials](/plasmacoll/tutorials/).
