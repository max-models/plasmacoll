"""The Coulomb collision kernels: Pyccel against NumPy, and the CUDA arithmetic on the CPU.

The NumPy versions run the Pyccel source as Python, so all versions must agree to
round-off for the same random numbers. The CUDA versions are compiled as C++ and run
one thread (cell) after the other, with stubs for the CUDA built-ins.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cunumpy as xp
import numpy as np
import pytest

import plasmacoll
from plasmacoll.kernels.coulomb_collide_inter import kernel as INTER
from plasmacoll.kernels.coulomb_collide_intra import kernel as INTRA

COMPILER = shutil.which("c++")
#: The directory containing plasmacoll, for ``#include "plasmacoll/..."``.
SOURCE_ROOT = Path(plasmacoll.__file__).resolve().parents[1]

STUBS = r"""
#include <cmath>
#include <cstdio>
#define __host__
#define __device__
#define __global__
#define __forceinline__ inline
struct dim3_ { unsigned x, y, z; };
static dim3_ threadIdx, blockDim, blockIdx, gridDim;
"""

#: The C type of each array dtype.
C_TYPES = {"f": "double", "i": "long long"}


def _c_array(name, values):
    values = np.asarray(values)
    if values.dtype.kind == "f":
        flat = ", ".join(repr(float(v)) for v in np.ravel(values))
    else:
        flat = ", ".join(str(int(v)) for v in np.ravel(values))
    return f"static {C_TYPES[values.dtype.kind]} {name}[] = {{{flat}}};\n"


def _view(name, array, ndim):
    shape = ", ".join(str(n) for n in array.shape)
    strides = ", ".join(str(s // array.itemsize) for s in array.strides)
    c_type = C_TYPES[array.dtype.kind]
    return (
        f"Array{ndim}D<{c_type}> {name}_view{{{name}, {{{shape}}}, {{{strides}}}}};\n"
    )


def _run(tmp_path, kernel, arrays, views, call, outputs):
    """Compile ``kernel`` with a main that calls it for each thread and prints ``outputs``."""
    n_threads, call = call
    body = "".join(_c_array(name, a) for name, a in arrays.items())
    body += f"int main() {{ blockDim.x = 1; gridDim.x = {n_threads};\n"
    body += "".join(_view(name, arrays[name], ndim) for name, ndim in views.items())
    body += (
        f"for (long long t = 0; t < {n_threads}; ++t) "
        f"{{ blockIdx.x = (unsigned)t; {call}; }}\n"
    )
    body += "".join(
        f"for (long long i = 0; i < {arrays[name].size}; ++i) "
        f'printf("%.17g\\n", (double){name}[i]);\n'
        for name in outputs
    )
    body += "return 0; }\n"
    source = tmp_path / "kernel.cpp"
    exe = tmp_path / "kernel"
    source.write_text(STUBS + kernel.source.replace('extern "C"', "") + body)
    subprocess.run(
        [
            str(COMPILER),
            "-std=c++17",
            "-O1",
            f"-I{xp.cuda.cuda_include_dir()}",
            f"-I{SOURCE_ROOT}",
            str(source),
            "-o",
            str(exe),
        ],
        check=True,
    )
    printed = np.array(
        subprocess.run(
            [str(exe)], check=True, capture_output=True, text=True
        ).stdout.split(),
        dtype=float,
    )
    results, offset = [], 0
    for name in outputs:
        array = arrays[name]
        results.append(
            printed[offset : offset + array.size]
            .reshape(array.shape)
            .astype(array.dtype)
        )
        offset += array.size
    return results


#: Markers per cell: empty, single, a pair, a triplet, odd and even counts.
COUNTS_A = np.array([0, 1, 2, 3, 6, 9, 4])
COUNTS_B = np.array([2, 0, 1, 5, 3, 9, 0])


def _starts(counts):
    starts = np.zeros(len(counts) + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)
    return starts


def _case(seed):
    rng = np.random.default_rng(seed)
    na, nb = int(COUNTS_A.sum()), int(COUNTS_B.sum())
    uniforms_intra = rng.random((5, na))
    uniforms_intra[1] = 1.0 - uniforms_intra[1]
    uniforms_inter = rng.random((5, na + nb))
    uniforms_inter[1] = 1.0 - uniforms_inter[1]
    return {
        # scattering parameters from small to beyond isotropic
        "coef": 10.0 ** rng.uniform(-2.0, 3.0, len(COUNTS_A)),
        "start_a": _starts(COUNTS_A),
        "start_b": _starts(COUNTS_B),
        "va": rng.normal(size=(3, na)),
        "vb": rng.normal(size=(3, nb)),
        "wa": rng.uniform(0.5, 1.5, na),
        "wb": rng.uniform(0.5, 1.5, nb),
        "u_intra": uniforms_intra,
        "u_inter": uniforms_inter,
    }


def _intra(case, method, implementation):
    velocities = case["va"].copy()
    order = np.empty(velocities.shape[1], dtype=np.int64)
    with xp.kernels.use_host_kernel_implementation(implementation):
        INTRA(
            case["coef"],
            case["start_a"],
            velocities,
            case["wa"],
            order,
            case["u_intra"],
            1.5,
            method,
        )
    return velocities, order


def _inter(case, method, implementation):
    va, vb = case["va"].copy(), case["vb"].copy()
    order_a = np.empty(va.shape[1], dtype=np.int64)
    order_b = np.empty(vb.shape[1], dtype=np.int64)
    with xp.kernels.use_host_kernel_implementation(implementation):
        INTER(
            case["coef"],
            case["start_a"],
            case["start_b"],
            va,
            vb,
            case["wa"],
            case["wb"],
            order_a,
            order_b,
            case["u_inter"],
            1.0,
            3.0,
            method,
        )
    return va, vb, order_a, order_b


@pytest.mark.parametrize("method", [0, 1])
def test_compiled_coulomb_kernels_match_numpy(method):
    """The Pyccel and NumPy versions give the same velocities and permutations."""
    case = _case(1)
    for compiled, plain in zip(
        _intra(case, method, "pyccel"), _intra(case, method, "numpy"), strict=True
    ):
        np.testing.assert_allclose(compiled, plain, rtol=1e-13, atol=1e-13)
    for compiled, plain in zip(
        _inter(case, method, "pyccel"), _inter(case, method, "numpy"), strict=True
    ):
        np.testing.assert_allclose(compiled, plain, rtol=1e-13, atol=1e-13)


def test_intra_kernel_permutes_each_cell_and_keeps_its_momentum():
    """``order`` is a permutation within each cell; equal weights keep each cell's momentum."""
    case = _case(2)
    case["wa"] = np.ones_like(case["wa"])
    velocities, order = _intra(case, 1, "pyccel")
    starts = case["start_a"]
    for cell in range(len(COUNTS_A)):
        part = slice(starts[cell], starts[cell + 1])
        assert sorted(order[part]) == list(range(starts[cell], starts[cell + 1]))
        np.testing.assert_allclose(
            velocities[:, part].sum(axis=1), case["va"][:, part].sum(axis=1), atol=1e-12
        )
        np.testing.assert_allclose(
            np.sum(velocities[:, part] ** 2),
            np.sum(case["va"][:, part] ** 2),
            rtol=1e-12,
            atol=1e-12,
        )
    # cells of fewer than two markers are untouched, the others changed
    assert np.array_equal(velocities[:, :1], case["va"][:, :1])
    assert not np.array_equal(velocities[:, 1:3], case["va"][:, 1:3])


@pytest.mark.skipif(COMPILER is None, reason="no C++ compiler")
@pytest.mark.parametrize("method", [0, 1])
def test_coulomb_kernels_cuda_arithmetic_matches_cpu(tmp_path, method):
    """The CUDA versions, run one cell after the other, match the Pyccel versions."""
    case = _case(3)
    na, nb, cells = case["va"].shape[1], case["vb"].shape[1], len(COUNTS_A)
    arrays = {
        "coef": case["coef"],
        "start_a": case["start_a"],
        "start_b": case["start_b"],
        "va": case["va"].copy(),
        "vb": case["vb"].copy(),
        "wa": case["wa"],
        "wb": case["wb"],
        "order_a": np.zeros(na, dtype=np.int64),
        "order_b": np.zeros(nb, dtype=np.int64),
        "u_intra": case["u_intra"],
        "u_inter": case["u_inter"],
    }
    views = {
        "coef": 1,
        "start_a": 1,
        "start_b": 1,
        "va": 2,
        "vb": 2,
        "wa": 1,
        "wb": 1,
        "order_a": 1,
        "order_b": 1,
        "u_intra": 2,
        "u_inter": 2,
    }
    call = (
        cells,
        "coulomb_collide_intra(coef_view, start_a_view, va_view, wa_view, "
        f"order_a_view, u_intra_view, 1.5, {method})",
    )
    va, order = _run(
        tmp_path,
        INTRA.cuda_kernel,
        arrays,
        views,
        call,
        ["va", "order_a"],
    )
    expected_va, expected_order = _intra(case, method, "pyccel")
    np.testing.assert_allclose(va, expected_va, rtol=1e-12, atol=1e-12)
    assert np.array_equal(order, expected_order)

    arrays["va"] = case["va"].copy()
    call = (
        cells,
        "coulomb_collide_inter(coef_view, start_a_view, start_b_view, va_view, "
        "vb_view, wa_view, wb_view, order_a_view, order_b_view, u_inter_view, "
        f"1.0, 3.0, {method})",
    )
    (tmp_path / "inter").mkdir()
    va, vb = _run(
        tmp_path / "inter",
        INTER.cuda_kernel,
        arrays,
        views,
        call,
        ["va", "vb"],
    )
    expected = _inter(case, method, "pyccel")
    np.testing.assert_allclose(va, expected[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(vb, expected[1], rtol=1e-12, atol=1e-12)
