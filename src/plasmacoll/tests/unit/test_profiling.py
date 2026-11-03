"""The opt-in scope-profiler regions: off unless the application enables the tag.

The cross sections are synthetic, not physical data.
"""

from __future__ import annotations

import cunumpy as xp
import pytest
import scope_profiler as sp

from plasmacoll import CollisionProcess, ParticleArrays, ZeroDReactor
from plasmacoll.tests.unit._helpers import M_E, beam, constant, operator

REACTOR_REGIONS = {
    "plasmacoll.tabulate_bounds",
    "plasmacoll.reactor_run",
    "plasmacoll.reactor_step",
    "plasmacoll.collide_species",
    "plasmacoll.collide",
    "plasmacoll.collide_process",
}


@pytest.fixture(autouse=True)
def _fresh_profiler(monkeypatch):
    """Keep the profiling state of these tests out of the others."""
    monkeypatch.delenv("SCOPE_PROFILER_ENABLE_TAGS", raising=False)
    sp.reset()
    yield
    sp.reset()


def _region_names_of_a_reactor_run(**setup_kwargs) -> set[str]:
    sp.setup(output=None, **setup_kwargs)
    # built after setup, so the bound tabulation is profiled too
    mcc = operator([CollisionProcess("elastic", "gas", constant(1.0e-18))], {"e": M_E})
    electrons = ParticleArrays(
        xp.zeros((1000, 1)), beam(1000, 10.0, M_E), xp.ones(1000)
    )
    # an application region, recorded either way
    with sp.region("user.run"):
        ZeroDReactor(mcc, {"e": electrons}).run(dt=1.0e-10, num_steps=3)
    results = sp.finalize(verbose=False, return_results=True)
    return set(results.region_names)


def test_regions_are_off_by_default() -> None:
    assert _region_names_of_a_reactor_run() == {"user.run"}


@pytest.mark.parametrize("tags", [["plasmacoll"], ["*"]])
def test_enabling_the_tag_records_the_regions(tags: list[str]) -> None:
    names = _region_names_of_a_reactor_run(enable_tags=tags)
    assert names == REACTOR_REGIONS | {"user.run"}


def test_the_opt_in_tag_has_an_entry_point():
    """``scope-profiler tags`` finds the tag through the package's entry point."""
    from importlib.metadata import entry_points

    from plasmacoll._profiling import PROFILE_TAGS

    found = {
        entry_point.name: entry_point.value
        for entry_point in entry_points(group="scope_profiler.opt_in_tags")
    }
    assert found.get("plasmacoll") == "plasmacoll"
    assert "plasmacoll" in PROFILE_TAGS
