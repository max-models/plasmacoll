"""Draw the animated figure of the docs home page: electron-H2 cross sections.

The cross sections of electron collisions with H2 (vibrational and electronic excitation,
ionization and the dissociative attachment of the vibrational levels) from plasmacoll's tables
of the Janev, Reiter and Samm (2003) fits, on log-log axes, with dots running along the curves.
Hovering over a curve names its reaction. Rerun with ``make docs-hero`` (needs
``plasma_plots.hero``); it writes ``docs/src/assets/hero.svg``.
"""

from pathlib import Path

import numpy as np
from plasma_plots.hero import HeroFigure

from plasmacoll.janev import JANEV_FITS, dissociative_attachment_levels

OUTPUT = Path(__file__).parent / "src" / "assets" / "hero.svg"
MAIN = {"e_h2_ionization", "e_h2_vibrational_1", "e_h2_triplet_b", "e_h2_singlet_B"}
KEYS = [
    "e_h2_vibrational_1",
    "e_h2_vibrational_2",
    "e_h2_triplet_b",
    "e_h2_triplet_a",
    "e_h2_triplet_c",
    "e_h2_singlet_B",
    "e_h2_singlet_C",
    "e_h2_singlet_EF",
    "e_h2_ionization",
    "e_h2_dissociative_ionization",
]


def curve(table, num=70):
    """Return log10 energy and log10 sigma of a table, subsampled and above 1e-24 m^2."""
    energy = np.geomspace(max(table.energy[0], 0.3), 2.0e3, num)
    sigma = np.interp(energy, table.energy, table.sigma)
    keep = sigma > 1.0e-24
    return np.log10(energy[keep]), np.log10(sigma[keep])


fig = HeroFigure(xlim=(-2.8, 3.4), ylim=(-24.3, -18.6))
rng = np.random.default_rng(3)
for key in KEYS:
    fit = JANEV_FITS[key]
    x, y = curve(fit.cross_section(max_energy=2.0e3))
    main = key in MAIN
    line = fig.line(
        x,
        y,
        tone="accent" if main else "muted",
        width=2.2 if main else 1.2,
        opacity=0.9 if main else 0.55,
        title=fit.reaction,
    )
    duration = rng.uniform(8.0, 16.0)
    for k in range(2 if main else 1):
        fig.tracer(
            line, duration=duration, offset=duration * (k / 2 + rng.random() / 4)
        )
for table in dissociative_attachment_levels()[::2]:
    x, y = curve(table, num=30)
    if len(x) > 1:
        fig.line(x, y, tone="high", width=1.0, opacity=0.35, title=table.label)
fig.save(OUTPUT)
