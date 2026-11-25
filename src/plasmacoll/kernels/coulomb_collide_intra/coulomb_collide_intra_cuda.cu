// The CUDA version (GPU) of the coulomb_collide_intra kernel;
// coulomb_collide_intra_pyccel.py next to this file is the Pyccel version (CPU),
// with the same arguments. One thread per cell: the markers of a cell are
// shuffled and collided in pairs (a triplet first for an odd count).
#include "cunumpy/array_view.cuh"
#include "plasmacoll/kernels/coulomb_pair.cuh"

extern "C" __global__
void coulomb_collide_intra(
    Array1D<double> coef,
    Array1D<long long> cell_start,
    Array2D<double> velocities,
    Array1D<double> weights,
    Array1D<long long> order,
    Array2D<double> uniforms,
    double mass,
    long long method
) {
    const long long cell = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (cell >= coef.shape[0]) return;
    const long long start = cell_start(cell);
    const long long count = cell_start(cell + 1) - start;
    coulomb_shuffle(order, uniforms, start, start, count);
    if (count < 2 || coef(cell) <= 0.0) return;
    const long long first = count % 2 == 1 ? 3 : 0;
    double paired = 0.0;
    if (first == 3) {
        for (long long k = 0; k < 3; ++k) {
            paired += 0.5 * fmin(weights(order(start + k)),
                                 weights(order(start + (k + 1) % 3)));
        }
    }
    for (long long k = first; k < count - 1; k += 2) {
        paired += fmin(weights(order(start + k)), weights(order(start + k + 1)));
    }
    if (paired <= 0.0) return;
    const double scale = coef(cell) / (2.0 * paired);
    if (first == 3) {
        for (long long k = 0; k < 3; ++k) {
            const long long ia = order(start + k);
            const long long ib = order(start + (k + 1) % 3);
            const long long slot = start + k;
            const double wa = weights(ia), wb = weights(ib);
            const double accept = uniforms(4, slot) * fmax(wa, wb);
            coulomb_scatter_pair(velocities, ia, velocities, ib, mass, mass,
                                 0.5 * scale, method, uniforms(1, slot),
                                 uniforms(2, slot), uniforms(3, slot),
                                 accept < wb, accept < wa);
        }
    }
    for (long long k = first; k < count - 1; k += 2) {
        const long long ia = order(start + k);
        const long long ib = order(start + k + 1);
        const long long slot = start + k;
        const double wa = weights(ia), wb = weights(ib);
        const double accept = uniforms(4, slot) * fmax(wa, wb);
        coulomb_scatter_pair(velocities, ia, velocities, ib, mass, mass, scale,
                             method, uniforms(1, slot), uniforms(2, slot),
                             uniforms(3, slot), accept < wb, accept < wa);
    }
}
