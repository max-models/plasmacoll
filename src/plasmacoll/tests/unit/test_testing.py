"""The fake LXCat sets of `plasmacoll.testing` read back as LXCat files."""

from __future__ import annotations

from pathlib import Path

import pytest

from plasmacoll import find_lxcat_process, read_lxcat
from plasmacoll.testing import fake_argon_set, lxcat_block


def test_fake_argon_set_reads_back(tmp_path):
    """The fake set has its three processes, thresholds and flat cross sections."""
    path = fake_argon_set(tmp_path)
    assert path == tmp_path / "data" / "Ar_siglo.txt"
    processes = read_lxcat(path)
    assert [p.kind for p in processes] == ["EFFECTIVE", "EXCITATION", "IONIZATION"]
    ionization = find_lxcat_process(processes, "IONIZATION")
    assert ionization.parameter == pytest.approx(15.8)
    assert float(ionization.cross_section(500.0)) == pytest.approx(2e-20)


def test_lxcat_block_and_name(tmp_path):
    """A single block, written under another name."""
    block = lxcat_block("ELASTIC", "H2", 2.7e-4, "E + H2 -> E + H2", [(0.0, 1e-19)])
    assert block.startswith("ELASTIC\nH2\n 0.00027\n")
    path = fake_argon_set(tmp_path, Path("other.txt"))
    assert path.exists()
