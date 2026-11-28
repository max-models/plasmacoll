"""Binary Coulomb collisions of the markers of one species with each other, per cell.

The Pyccel version (CPU) of the ``coulomb_collide_intra`` kernel;
``coulomb_collide_intra_cuda.cu`` next to this file is the CUDA version (GPU) and
``coulomb_collide_intra_numpy.py`` runs this file as plain Python, all with the same
arguments. :func:`plasmacoll.coulomb.collide_intra` calls it.
"""

# mypy: ignore-errors

from math import cos, exp, log, sin, sinh, sqrt

TWO_PI = 6.283185307179586


def _cos_chi(s: float, method: int, r1: float, r2: float) -> float:
    """The cosine of the scattering angle of a pair with the scattering parameter ``s``.

    ``method`` 0 (Takizuka and Abe): ``tan(chi / 2)`` is normal with the variance
    ``s / 2``, drawn by Box-Muller from ``r1`` in (0, 1] and ``r2``. ``method`` 1
    (Nanbu): the cumulative angle of all small-angle collisions in the step, with
    ``coth A - 1/A = exp(-s)`` from the fit of Perez et al. (2012), from ``r1``.
    """
    if method == 0:
        delta = sqrt(-s * log(r1)) * cos(TWO_PI * r2)
        delta2 = delta * delta
        return (1.0 - delta2) / (1.0 + delta2)
    if s < 0.1:
        cos_chi = 1.0 + s * log(r1)
    elif s < 6.0:
        if s < 3.0:
            inverse = 0.0056958 + s * (
                0.9560202
                + s
                * (-0.508139 + s * (0.47913906 + s * (-0.12788975 + s * 0.02389567)))
            )
            a = 1.0 / inverse
        else:
            a = 3.0 * exp(-s)
            inverse = 1.0 / a
        cos_chi = inverse * log(exp(-a) + 2.0 * r1 * sinh(a))
    else:
        cos_chi = 2.0 * r1 - 1.0
    if cos_chi < -1.0:
        cos_chi = -1.0
    if cos_chi > 1.0:
        cos_chi = 1.0
    return cos_chi


def _scatter_pair(
    va: "float[:,:]",
    ia: int,
    vb: "float[:,:]",
    ib: int,
    mass_a: float,
    mass_b: float,
    coef: float,
    method: int,
    r1: float,
    r2: float,
    r3: float,
    update_a: bool,
    update_b: bool,
) -> None:
    """Turn the relative velocity of markers ``ia`` of ``va`` and ``ib`` of ``vb``.

    The scattering parameter is ``s = coef / u**3`` with ``u`` the relative speed;
    the change of the relative velocity is shared in the ratio of the masses
    (Takizuka and Abe 1977, eq. 4-5), which keeps the momentum and the energy of the
    pair, and applied to a marker only if its ``update_`` flag is set.
    """
    ux = va[0, ia] - vb[0, ib]
    uy = va[1, ia] - vb[1, ib]
    uz = va[2, ia] - vb[2, ib]
    u2 = ux * ux + uy * uy + uz * uz
    if u2 <= 0.0:
        return
    u = sqrt(u2)
    s = coef / (u2 * u)
    cos_chi = _cos_chi(s, method, r1, r2)
    one_minus = 1.0 - cos_chi
    sin_chi = sqrt(max(0.0, (1.0 - cos_chi) * (1.0 + cos_chi)))
    phi = TWO_PI * r3
    cos_phi = cos(phi)
    sin_phi = sin(phi)
    u_perp = sqrt(ux * ux + uy * uy)
    if u_perp > 1.0e-12 * u:
        dux = (
            (ux / u_perp) * uz * sin_chi * cos_phi
            - (uy / u_perp) * u * sin_chi * sin_phi
            - ux * one_minus
        )
        duy = (
            (uy / u_perp) * uz * sin_chi * cos_phi
            + (ux / u_perp) * u * sin_chi * sin_phi
            - uy * one_minus
        )
        duz = -u_perp * sin_chi * cos_phi - uz * one_minus
    else:
        dux = u * sin_chi * cos_phi
        duy = u * sin_chi * sin_phi
        duz = -uz * one_minus
    total = mass_a + mass_b
    if update_a:
        fa = mass_b / total
        va[0, ia] += fa * dux
        va[1, ia] += fa * duy
        va[2, ia] += fa * duz
    if update_b:
        fb = mass_a / total
        vb[0, ib] -= fb * dux
        vb[1, ib] -= fb * duy
        vb[2, ib] -= fb * duz


def _collide_cell(
    velocities: "float[:,:]",
    weights: "float[:]",
    order: "int[:]",
    uniforms: "float[:,:]",
    start: int,
    count: int,
    coef: float,
    mass: float,
    method: int,
) -> None:
    """Shuffle the ``count`` markers from ``start`` and collide them in pairs."""
    for k in range(count):
        order[start + k] = start + k
    for k in range(count - 1, 0, -1):
        j = int(uniforms[0, start + k] * (k + 1))
        if j > k:
            j = k
        swap = order[start + k]
        order[start + k] = order[start + j]
        order[start + j] = swap
    if count < 2 or coef <= 0.0:
        return
    # an odd count: the first three markers collide as a triplet, three pairs
    # of half a step each (Takizuka and Abe 1977)
    first = 3 if count % 2 == 1 else 0
    # the weight of the pairs, sum of min(w_i, w_j) (half for the triplet)
    paired = 0.0
    if first == 3:
        for k in range(3):
            wa = weights[order[start + k]]
            wb = weights[order[start + (k + 1) % 3]]
            paired += 0.5 * min(wa, wb)
    for k in range(first, count - 1, 2):
        paired += min(weights[order[start + k]], weights[order[start + k + 1]])
    if paired <= 0.0:
        return
    # n^2 / n_12 with n_12 = 2 * paired / V: coef holds W^2 / V
    scale = coef / (2.0 * paired)
    if first == 3:
        for k in range(3):
            ia = order[start + k]
            ib = order[start + (k + 1) % 3]
            slot = start + k
            wa = weights[ia]
            wb = weights[ib]
            wmax = max(wa, wb)
            accept = uniforms[4, slot] * wmax
            _scatter_pair(
                velocities,
                ia,
                velocities,
                ib,
                mass,
                mass,
                0.5 * scale,
                method,
                uniforms[1, slot],
                uniforms[2, slot],
                uniforms[3, slot],
                accept < wb,
                accept < wa,
            )
    for k in range(first, count - 1, 2):
        ia = order[start + k]
        ib = order[start + k + 1]
        slot = start + k
        wa = weights[ia]
        wb = weights[ib]
        wmax = max(wa, wb)
        accept = uniforms[4, slot] * wmax
        _scatter_pair(
            velocities,
            ia,
            velocities,
            ib,
            mass,
            mass,
            scale,
            method,
            uniforms[1, slot],
            uniforms[2, slot],
            uniforms[3, slot],
            accept < wb,
            accept < wa,
        )


def coulomb_collide_intra(
    coef: "float[:]",
    cell_start: "int[:]",
    velocities: "float[:,:]",
    weights: "float[:]",
    order: "int[:]",
    uniforms: "float[:,:]",
    mass: float,
    method: int,
) -> None:
    """Collide the markers of one species with each other, in random pairs per cell.

    The markers are sorted by cell: cell ``c`` holds markers ``cell_start[c]`` to
    ``cell_start[c + 1] - 1``. Each cell's markers are shuffled (Fisher-Yates with
    ``uniforms[0]``, the permutation written to ``order``) and paired; an odd count
    starts with a triplet. A pair of relative speed ``u`` scatters with ``s =
    coef[c] / (2 sum_pairs min(w_i, w_j)) / u**3`` (``coef`` is ``q^4 lnL dt W^2 /
    (4 pi eps0^2 mu^2 V)``, ``W`` the cell's weight, ``V`` its volume). The heavier
    marker of a pair changes with the probability ``w_light / w_heavy`` (Perez et al.
    2012), so with equal weights each pair keeps its momentum and energy exactly.
    Pair ``k`` of a cell uses ``uniforms[1:5, cell_start[c] + k]``; ``velocities`` is
    ``(3, N)``, ``uniforms`` ``(5, N)``; ``method`` 0 is Takizuka-Abe, 1 Nanbu.
    """
    # fmt: off
    #$ omp parallel for schedule(dynamic, 16)
    # fmt: on
    for cell in range(coef.shape[0]):
        _collide_cell(
            velocities,
            weights,
            order,
            uniforms,
            cell_start[cell],
            cell_start[cell + 1] - cell_start[cell],
            coef[cell],
            mass,
            method,
        )
