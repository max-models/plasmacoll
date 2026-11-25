"""Fake cross-section sets in LXCat's text format, for tests of the wiring only.

LXCat data may not be redistributed, so tests that read LXCat files write fake ones:
flat cross sections with the right process kinds and thresholds. Not physical data.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

__all__ = ["fake_argon_set", "lxcat_block"]


def lxcat_block(
    kind: str,
    target: str,
    parameter: float,
    process: str,
    table: Iterable[tuple[float, float]],
) -> str:
    """Return one process of an LXCat file.

    Args:
        kind: The process kind, e.g. ``"EXCITATION"``.
        target: The target line, e.g. ``"Ar -> Ar*(11.5eV)"``.
        parameter: The threshold (eV) or the mass ratio.
        process: The ``PROCESS:`` line.
        table: Pairs of energy (eV) and cross section (m^2).
    """
    lines = [
        kind,
        target,
        f" {parameter:.6g}",
        f"PROCESS: {process}",
        "COLUMNS: Energy | sigma",
        "-----------------------------",
    ]
    lines += [f" {energy:.6g} {sigma:.6g}" for energy, sigma in table]
    lines.append("-----------------------------")
    return "\n".join(lines) + "\n\n"


def fake_argon_set(
    directory: Path, name: Path | str = Path("data") / "Ar_siglo.txt"
) -> Path:
    """Write a FAKE electron-argon set (effective, one excitation, ionization).

    Not physical data: flat cross sections of ``1e-19``, ``1e-20`` and ``2e-20`` m^2
    above the thresholds 0, 11.5 and 15.8 eV. The default ``name`` is that of LXCat's
    SIGLO argon set, for code that reads the real one.

    Args:
        directory: The directory to write into.
        name: The file name, relative to ``directory`` (parents are created).

    Returns:
        The file written, ``directory / name``.
    """
    text = (
        lxcat_block(
            "EFFECTIVE",
            "Ar",
            1.36e-5,
            "E + Ar -> E + Ar, Effective (FAKE)",
            [(0.0, 1.0e-19), (100.0, 1.0e-19), (1000.0, 1.0e-19)],
        )
        + lxcat_block(
            "EXCITATION",
            "Ar -> Ar*(11.5eV)",
            11.5,
            "E + Ar -> E + Ar*(11.5eV), Excitation (FAKE)",
            [(11.5, 0.0), (20.0, 1.0e-20), (1000.0, 1.0e-20)],
        )
        + lxcat_block(
            "IONIZATION",
            "Ar -> Ar^+",
            15.8,
            "E + Ar -> E + E + Ar+, Ionization (FAKE)",
            [(15.8, 0.0), (100.0, 2.0e-20), (1000.0, 2.0e-20)],
        )
    )
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path
