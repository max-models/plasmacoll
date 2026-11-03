"""Shared test setup."""

import sys

import cunumpy  # noqa: F401  (installs the fake CuPy when CUNUMPY_FAKE_CUPY is set)

# the fake cupy.random answers every attribute with a function, __file__ too,
# which breaks the module introspection of astropy's import: give it none
_random = sys.modules.get("cupy.random")
if _random is not None and not isinstance(getattr(_random, "__file__", None), str):
    _random.__file__ = None
