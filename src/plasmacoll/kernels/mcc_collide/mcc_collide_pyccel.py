"""The null-collision Monte Carlo step of the candidates of one species.

The Pyccel version (CPU) of the ``mcc_collide`` kernel; ``mcc_collide_numpy.py`` runs
this file as plain Python, with the same arguments. `MonteCarloCollisions.collide`
calls it for the common case (`MonteCarloCollisions.kernel_plan`): backgrounds without
profiles, tabulated cross sections, and processes of the kinds ``elastic``,
``excitation``, ``backscatter`` and electron-impact ``ionization``. The random numbers
are drawn outside, so the kernel is deterministic.
"""

# mypy: ignore-errors

from math import atan, cos, pi, sin, sqrt, tan

ELEMENTARY_CHARGE = 1.602176634e-19
HARTREE_EV = 27.211386245988

# process kinds
SCATTER = 0
BACKSCATTER = 1
IONIZATION = 2

# scattering models
ISOTROPIC = 0
VAHEDI_SURENDRA = 1
OKHRIMOVSKYY = 2

# energy sharing of ionization
EQUAL = 0
UNIFORM = 1
OPAL = 2


def _sigma(
    energy: float,
    table_energy: "float[:]",
    table_sigma: "float[:]",
    start: int,
    length: int,
    threshold: float,
) -> float:
    """The cross section at ``energy`` (eV): linear between points, held outside."""
    if energy < threshold:
        return 0.0
    last = start + length - 1
    if energy <= table_energy[start]:
        return table_sigma[start]
    if energy >= table_energy[last]:
        return table_sigma[last]
    low = start
    high = last
    while high - low > 1:
        middle = (low + high) // 2
        if table_energy[middle] <= energy:
            low = middle
        else:
            high = middle
    width = table_energy[high] - table_energy[low]
    if width <= 0.0:
        return table_sigma[high]
    t = (energy - table_energy[low]) / width
    return table_sigma[low] + t * (table_sigma[high] - table_sigma[low])


def _cos_chi(model: int, energy_ev: float, uniform: float) -> float:
    """The cosine of the scattering angle of `plasmacoll.mcc.scattering_cosine`."""
    if energy_ev < 0.0:
        energy_ev = 0.0
    if model == VAHEDI_SURENDRA:
        if energy_ev < 1.0e-6:
            return 1.0 - 2.0 * uniform
        cosine = (2.0 + energy_ev - 2.0 * (1.0 + energy_ev) ** uniform) / energy_ev
    elif model == OKHRIMOVSKYY:
        scaled = 4.0 * energy_ev / HARTREE_EV
        xi = scaled / (1.0 + scaled)
        cosine = 1.0 - 2.0 * uniform * (1.0 - xi) / (1.0 + xi * (1.0 - 2.0 * uniform))
    else:
        return 1.0 - 2.0 * uniform
    if cosine < -1.0:
        cosine = -1.0
    if cosine > 1.0:
        cosine = 1.0
    return cosine


def _direction(
    model: int,
    energy_ev: float,
    u1: float,
    u2: float,
    gx: float,
    gy: float,
    gz: float,
) -> tuple[float, float, float]:
    """A unit vector at the model's angle about ``(gx, gy, gz)`` (isotropic: any)."""
    cos_chi = _cos_chi(model, energy_ev, u1)
    sin_chi = sqrt(max(1.0 - cos_chi * cos_chi, 0.0))
    phi = 2.0 * pi * u2
    if model == ISOTROPIC:
        return sin_chi * cos(phi), sin_chi * sin(phi), cos_chi
    # first = axis x g / |axis x g|, axis the coordinate axis least aligned with g
    if abs(gx) > 0.9:
        ax = 0.0
        ay = 1.0
    else:
        ax = 1.0
        ay = 0.0
    fx = ay * gz
    fy = -ax * gz
    fz = ax * gy - ay * gx
    norm = sqrt(fx * fx + fy * fy + fz * fz)
    fx = fx / norm
    fy = fy / norm
    fz = fz / norm
    sx = gy * fz - gz * fy
    sy = gz * fx - gx * fz
    sz = gx * fy - gy * fx
    a = sin_chi * cos(phi)
    b = sin_chi * sin(phi)
    return (
        cos_chi * gx + a * fx + b * sx,
        cos_chi * gy + a * fy + b * sy,
        cos_chi * gz + a * fz + b * sz,
    )


def mcc_collide(
    vx_all: "float[:]",
    vy_all: "float[:]",
    vz_all: "float[:]",
    weights: "float[:]",
    candidates: "int64[:]",
    normals: "float[:,:]",
    uniforms: "float[:,:]",
    nu_max: float,
    mass: float,
    background_mass: "float[:]",
    background_thermal: "float[:]",
    background_drift: "float[:,:]",
    kind: "int64[:]",
    scattering: "int64[:]",
    background: "int64[:]",
    density: "float[:]",
    lookup_mass: "float[:]",
    loss: "float[:]",
    threshold: "float[:]",
    sharing: "int64[:]",
    sharing_width: "float[:]",
    table_start: "int64[:]",
    table_length: "int64[:]",
    table_energy: "float[:]",
    table_sigma: "float[:]",
    counts: "int64[:]",
    product_parent: "int64[:]",
    product_electron: "float[:,:]",
    product_ion: "float[:,:]",
    energy_change: "float[:]",
) -> int:
    """Collide the ``candidates`` of one species with a real or the null process.

    ``vx_all``, ``vy_all``, ``vz_all`` are the velocity components of all ``N``
    markers, updated in place; ``normals`` ``(3 B, C)`` the
    standard normal draws of the neutral velocities (``B`` backgrounds, ``C``
    candidates); ``uniforms`` ``(6, C)``: the process draw, then two pairs for the
    scattering directions and one for the energy sharing. Per process: its ``kind``
    (``SCATTER``, ``BACKSCATTER``, ``IONIZATION``), ``scattering`` model, background,
    density (1/m^3), the mass of its lookup energy, its energy ``loss`` (J), threshold
    (eV), energy ``sharing`` and its width (J), and its table in ``table_energy`` and
    ``table_sigma``. ``counts`` gets the collisions per process and, at the end, the
    null collisions and the bound violations; an ionization writes its parent, the new
    electron's and the ion's velocity to the ``product_*`` arrays. ``energy_change[0]``
    gets the change of the incident markers' kinetic energy (J, with their weights).
    Returns the number of products.
    """
    num_candidates = candidates.shape[0]
    num_processes = kind.shape[0]
    num_products = 0
    for c in range(num_candidates):
        i = candidates[c]
        vx = vx_all[i]
        vy = vy_all[i]
        vz = vz_all[i]
        selection = uniforms[0, c] * nu_max
        cumulative = 0.0
        chosen = -1
        gx = 0.0
        gy = 0.0
        gz = 0.0
        g = 0.0
        wx = 0.0
        wy = 0.0
        wz = 0.0
        for p in range(num_processes):
            b = background[p]
            pwx = background_drift[b, 0] + background_thermal[b] * normals[3 * b, c]
            pwy = background_drift[b, 1] + background_thermal[b] * normals[3 * b + 1, c]
            pwz = background_drift[b, 2] + background_thermal[b] * normals[3 * b + 2, c]
            pgx = vx - pwx
            pgy = vy - pwy
            pgz = vz - pwz
            pg = sqrt(pgx * pgx + pgy * pgy + pgz * pgz)
            energy = 0.5 * lookup_mass[p] * pg * pg / ELEMENTARY_CHARGE
            sigma = _sigma(
                energy,
                table_energy,
                table_sigma,
                table_start[p],
                table_length[p],
                threshold[p],
            )
            cumulative = cumulative + density[p] * sigma * pg
            if chosen < 0 and selection < cumulative:
                chosen = p
                gx = pgx
                gy = pgy
                gz = pgz
                g = pg
                wx = pwx
                wy = pwy
                wz = pwz
        if cumulative > nu_max * (1.0 + 1.0e-9):
            counts[num_processes + 1] += 1
        if chosen < 0:
            counts[num_processes] += 1
            continue
        counts[chosen] += 1
        p = chosen
        neutral_mass = background_mass[background[p]]
        total_mass = mass + neutral_mass
        reduced_mass = mass * neutral_mass / total_mass
        cx = (mass * vx + neutral_mass * wx) / total_mass
        cy = (mass * vy + neutral_mass * wy) / total_mass
        cz = (mass * vz + neutral_mass * wz) / total_mass
        if g > 0.0:
            ux = gx / g
            uy = gy / g
            uz = gz / g
        else:
            ux = 0.0
            uy = 0.0
            uz = 1.0
        if kind[p] == SCATTER:
            lookup_energy = 0.5 * lookup_mass[p] * g * g / ELEMENTARY_CHARGE
            speed_squared = g * g - 2.0 * loss[p] / reduced_mass
            new_speed = sqrt(max(speed_squared, 0.0))
            dx, dy, dz = _direction(
                scattering[p],
                lookup_energy,
                uniforms[1, c],
                uniforms[2, c],
                ux,
                uy,
                uz,
            )
            factor = neutral_mass / total_mass * new_speed
            nx = cx + factor * dx
            ny = cy + factor * dy
            nz = cz + factor * dz
        elif kind[p] == BACKSCATTER:
            factor = neutral_mass / total_mass
            nx = cx - factor * gx
            ny = cy - factor * gy
            nz = cz - factor * gz
        else:
            energy = 0.5 * mass * g * g
            residual = max(energy - loss[p], 0.0)
            if sharing[p] == EQUAL:
                fraction = 0.5
            elif sharing[p] == UNIFORM:
                fraction = uniforms[5, c]
            elif residual > 0.0:
                width = sharing_width[p]
                fraction = (
                    width * tan(uniforms[5, c] * atan(residual / (2.0 * width)))
                ) / residual
            else:
                fraction = 0.0
            primary_energy = (1.0 - fraction) * residual
            secondary_energy = fraction * residual
            primary_speed = sqrt(2.0 * primary_energy / mass)
            secondary_speed = sqrt(2.0 * secondary_energy / mass)
            dx, dy, dz = _direction(
                scattering[p],
                primary_energy / ELEMENTARY_CHARGE,
                uniforms[1, c],
                uniforms[2, c],
                ux,
                uy,
                uz,
            )
            nx = wx + primary_speed * dx
            ny = wy + primary_speed * dy
            nz = wz + primary_speed * dz
            dx, dy, dz = _direction(
                scattering[p],
                secondary_energy / ELEMENTARY_CHARGE,
                uniforms[3, c],
                uniforms[4, c],
                ux,
                uy,
                uz,
            )
            product_parent[num_products] = i
            product_electron[0, num_products] = wx + secondary_speed * dx
            product_electron[1, num_products] = wy + secondary_speed * dy
            product_electron[2, num_products] = wz + secondary_speed * dz
            product_ion[0, num_products] = wx
            product_ion[1, num_products] = wy
            product_ion[2, num_products] = wz
            num_products += 1
        energy_change[0] += (
            0.5
            * mass
            * weights[i]
            * (nx * nx + ny * ny + nz * nz - vx * vx - vy * vy - vz * vz)
        )
        vx_all[i] = nx
        vy_all[i] = ny
        vz_all[i] = nz
    return num_products
