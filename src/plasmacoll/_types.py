"""Typing aliases shared by the modules of plasmacoll."""

from __future__ import annotations

from os import PathLike
from typing import Any, TypeAlias

#: A NumPy or CuPy array, as returned by :mod:`cunumpy` for the active backend.
Array: TypeAlias = Any
#: A file path given as a string or a path object.
PathLikeStr: TypeAlias = str | PathLike[str]
