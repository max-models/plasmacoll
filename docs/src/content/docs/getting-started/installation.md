---
title: Installation
description: Install plasmacoll and its optional extras.
---

plasmacoll needs Python 3.10 or newer. It depends on NumPy and
[cunumpy](https://pypi.org/project/cunumpy/), which selects NumPy or CuPy as the array backend.

```bash
pip install plasmacoll
```

For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/max-models/plasmacoll
cd plasmacoll
pip install -e ".[dev]"
```

or with [uv](https://docs.astral.sh/uv/):

```bash
make install    # uv sync --extra dev, plus the pre-commit hooks
```

## GPU

On a machine with CuPy installed, select the CuPy backend before the first array is created:

```bash
CUNUMPY_BACKEND=cupy python my_simulation.py
```

or `cunumpy.set_backend("cupy")` at the start of the program. plasmacoll then creates its arrays and
random numbers on the GPU. The test suite also runs on cunumpy's host-memory stand-in for CuPy,
so the CuPy code paths are checked without a GPU:

```bash
make test-cupy  # CUNUMPY_FAKE_CUPY=1 CUNUMPY_BACKEND=cupy pytest
```

## Optional extras

| Extra  | Installs                                                           |
| ------ | ------------------------------------------------------------------ |
| `test` | `pytest` and `pytest-cov`                                          |
| `docs` | the notebook runner, matplotlib and griffe for the API reference   |
| `dev`  | ruff, pyright, ty and pre-commit, plus the `test` and `docs` extras |
