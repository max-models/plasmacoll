"""Physical quantities in the setup of a model: astropy Quantities in, plain numbers out.

Every physical setting declares the unit the code keeps it in (SI, and eV for
energies and temperatures as kT). It may be given as

- an astropy Quantity, ``0.1 * u.ns`` (also an array, or a list of them);
- a string an astropy Quantity parses, ``"0.1 ns"``, as in a configuration file;
- a bare number, which is already in the declared unit.

and is stored as a float (or a tuple of floats) in the declared unit, so the
models and their arrays never see a Quantity. A value of the wrong dimension
raises. Temperatures and energies convert between K, eV and J.

astropy is imported only when a Quantity or a string is given: a bare number
needs no astropy. Install it with ``pip install plasmacoll[units]``.

A dataclass declares the unit of a field with :func:`unit_field` and calls
:func:`convert_units` in its ``__post_init__``; a function converts an argument
with :func:`to_unit` (or :func:`to_array` for arrays); a configuration file
writes a value back with :func:`with_unit`, e.g. ``dt: 1e-10 s``.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import re
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = [
    "convert_units",
    "dataclass_to_dict",
    "describe",
    "is_quantity",
    "parse_quantity",
    "rate_to_si",
    "to_array",
    "to_unit",
    "unit_field",
    "unit_metadata",
    "with_unit",
]


def _astropy_units() -> Any:
    """Return ``astropy.units``, importing it on first use."""
    try:
        import astropy.units as u  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - astropy is a test dependency
        raise ImportError(
            "quantities need astropy: pip install plasmacoll[units]"
        ) from error
    return u


def is_quantity(value: Any) -> bool:
    """Return whether ``value`` is an astropy Quantity (without importing astropy)."""
    units = sys.modules.get("astropy.units")
    return units is not None and isinstance(value, units.Quantity)


def unit_metadata(unit: str, *, array: bool = False) -> dict[str, Any]:
    """Return the field metadata that declares ``unit``, for ``dataclasses.field``.

    The form for a field without a default, which type checkers read as a
    required field (they read any other call as a default)::

        density: float = dataclasses.field(metadata=unit_metadata("m^-3"))

    ``array=True`` keeps arrays as arrays (:func:`to_array`) instead of
    turning them into tuples (:func:`to_unit`).
    """
    return {"unit": unit, "array": array}


def unit_field(
    unit: str, default: Any = dataclasses.MISSING, *, array: bool = False, **kwargs: Any
) -> Any:
    """Return a dataclass field kept in ``unit``, with a default.

    :func:`convert_units` turns a Quantity or a string given for it into
    numbers in ``unit``; ``array=True`` keeps arrays as arrays. A field without
    a default is better declared with
    ``dataclasses.field(metadata=unit_metadata(unit))``, which type checkers
    read as required.
    """
    return dataclasses.field(
        default=default, metadata=unit_metadata(unit, array=array), **kwargs
    )


def parse_quantity(value: str) -> Any:
    """Return the Quantity of a string such as ``"0.1 ns"`` or ``"[[0, 1], [2, 3]] mm"``."""
    u = _astropy_units()
    try:
        return u.Quantity(value)
    except (TypeError, ValueError):
        pass
    # astropy reads flat lists only; a nested one is read as a literal
    match = re.fullmatch(r"\s*(\[.*\])\s*(.*?)\s*", value)
    try:
        if match is None:
            raise ValueError
        return np.array(ast.literal_eval(match[1]), dtype=float) * u.Unit(match[2])
    except (TypeError, ValueError, SyntaxError) as error:
        raise ValueError(f"cannot read {value!r} as a quantity") from error


def _magnitude(value: Any, unit: str, name: str) -> Any:
    """Return the numbers of a Quantity in ``unit`` (a dimensionless one as it is)."""
    u = _astropy_units()
    if value.unit == u.dimensionless_unscaled:
        return value.value
    try:
        return value.to_value(unit, equivalencies=u.temperature_energy())
    except u.UnitConversionError as error:
        wanted = f"in units of {unit}" if unit else "a plain number"
        raise ValueError(f"{name} must be {wanted}, got {value}") from error


def to_unit(value: Any, unit: str, name: str = "value") -> Any:
    """Return ``value`` as plain numbers in ``unit``.

    A Quantity, or a string that parses as one, is converted (a dimensionless
    one is a number in ``unit``, as YAML reads ``1e-10`` as a string); a number
    is taken as already in ``unit``. Sequences and arrays become tuples, and
    mappings keep their keys. ``None`` stays ``None``.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = parse_quantity(value)
    if is_quantity(value):
        value = _magnitude(value, unit, name)
    if isinstance(value, Mapping):
        if "unit" in value:
            raise ValueError(
                f"{name}: write a quantity as a string such as '2.5 mm', not "
                f"{dict(value)}"
            )
        return {
            key: to_unit(item, unit, f"{name}[{key!r}]") for key, item in value.items()
        }
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return float(value)
        return tuple(to_unit(item, unit, name) for item in value)
    if isinstance(value, (list, tuple)):
        return tuple(to_unit(item, unit, name) for item in value)
    if isinstance(value, (int, float, np.number)):
        return float(value)
    raise TypeError(f"{name} must be a number or a quantity, got {value!r}")


def to_array(value: Any, unit: str, name: str = "value") -> Any:
    """Return ``value`` in ``unit``, keeping arrays as arrays.

    For arguments that may be arrays (a profile, a field): a Quantity or a
    string is converted to a float or a NumPy array, a list of quantities to a
    NumPy array, and anything else, a number or an array of any backend (NumPy,
    CuPy), is returned unchanged.
    """
    if isinstance(value, str):
        value = parse_quantity(value)
    if is_quantity(value):
        magnitude = _magnitude(value, unit, name)
        return float(magnitude) if np.ndim(magnitude) == 0 else np.asarray(magnitude)
    if isinstance(value, (list, tuple)) and any(
        isinstance(item, str) or is_quantity(item) for item in value
    ):
        return np.asarray(to_unit(value, unit, name), dtype=float)
    return value


def rate_to_si(value: Any, name: str = "rate") -> float | None:
    """Return a particle rate in SI units.

    A rate of particles per second per metre of every axis a source does not
    span has the unit ``1/s``, ``1/(m s)`` or ``1/(m2 s)``, depending on the
    grid; a Quantity or a string of any of them is converted to SI.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = parse_quantity(value)
    if is_quantity(value):
        u = _astropy_units()
        per_second = (value * u.s).si
        if per_second.unit not in (u.one, u.m**-1, u.m**-2):
            raise ValueError(
                f"{name} must be particles per second (per m or m^2), got {value}"
            )
        return float(per_second.value)
    return float(value)


def convert_units(obj: Any) -> None:
    """Convert every field of the dataclass ``obj`` that declares a unit.

    The fields declared with :func:`unit_field` or :func:`unit_metadata`; an
    array field (``array=True``) goes through :func:`to_array`, the others
    through :func:`to_unit`. Works on frozen dataclasses too.
    """
    for entry in dataclasses.fields(obj):
        unit = entry.metadata.get("unit")
        if unit is not None:
            convert = to_array if entry.metadata.get("array") else to_unit
            value = convert(getattr(obj, entry.name), unit, entry.name)
            object.__setattr__(obj, entry.name, value)


def _number(value: float) -> str:
    """Return the shortest text that reads back as exactly ``value``."""
    value = float(value)
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    short = format(value, ".15g")
    return short if float(short) == value else repr(value)


def with_unit(value: Any, unit: str) -> Any:
    """Return the configuration-file form of ``value`` in ``unit``.

    A number becomes a string such as ``"1e-10 s"``; a sequence a list of them
    (YAML reads ``[5 mm, 3 mm]``, not ``[5, 3] mm``). ``None`` stays ``None``.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {key: with_unit(item, unit) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [with_unit(item, unit) for item in value]
    return f"{_number(value)} {unit}".rstrip()


def describe(value: Any, unit: str) -> str:
    """Return ``value`` in ``unit`` for printing, e.g. ``1e-10 s``."""
    if value is None:
        return "None"
    if isinstance(value, (list, tuple, np.ndarray)):
        return "[" + ", ".join(_number(v) for v in np.ravel(value)) + f"] {unit}"
    return f"{_number(value)} {unit}"


def _plain(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    return value


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Return the configuration-file form of a flat dataclass.

    Fields with a unit are written with :func:`with_unit`, tuples as lists;
    the class reads it back with ``cls(**params)``.
    """
    params = {}
    for entry in dataclasses.fields(obj):
        value = getattr(obj, entry.name)
        unit = entry.metadata.get("unit")
        params[entry.name] = _plain(value) if unit is None else with_unit(value, unit)
    return params
