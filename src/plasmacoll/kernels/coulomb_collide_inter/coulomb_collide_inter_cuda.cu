// The CUDA version (GPU) of the coulomb_collide_inter kernel;
// coulomb_collide_inter_pyccel.py next to this file is the Pyccel version (CPU),
// with the same arguments. One thread per cell: the markers of both species in
// the cell are shuffled, and each marker of the species with more markers there
// collides with one of the other, whose markers are reused in turn.
#include "cunumpy/array_view.cuh"
#include "plasmacoll/kernels/coulomb_pair.cuh"

extern "C" __global__
void coulomb_collide_inter(
    Array1D<double> coef,
    Array1D<long long> start_a,
    Array1D<long long> start_b,
    Array2D<double> velocities_a,
    Array2D<double> velocities_b,
    Array1D<double> weights_a,
    Array1D<double> weights_b,
    Array1D<long long> order_a,
    Array1D<long long> order_b,
    Array2D<double> uniforms,
    double mass_a,
    double mass_b,
    long long method
) {
    const long long cell = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (cell >= coef.shape[0]) return;
    const long long first_a = start_a(cell), count_a = start_a(cell + 1) - first_a;
    const long long first_b = start_b(cell), count_b = start_b(cell + 1) - first_b;
    const long long slot = first_a + first_b;
    coulomb_shuffle(order_a, uniforms, first_a, slot, count_a);
    coulomb_shuffle(order_b, uniforms, first_b, slot + count_a, count_b);
    if (count_a == 0 || count_b == 0 || coef(cell) <= 0.0) return;
    const long long pairs = count_a > count_b ? count_a : count_b;
    double paired = 0.0;
    for (long long k = 0; k < pairs; ++k) {
        paired += fmin(weights_a(order_a(first_a + k % count_a)),
                       weights_b(order_b(first_b + k % count_b)));
    }
    if (paired <= 0.0) return;
    const double scale = coef(cell) / paired;
    for (long long k = 0; k < pairs; ++k) {
        const long long ia = order_a(first_a + k % count_a);
        const long long ib = order_b(first_b + k % count_b);
        const double wa = weights_a(ia), wb = weights_b(ib);
        const double accept = uniforms(4, slot + k) * fmax(wa, wb);
        coulomb_scatter_pair(velocities_a, ia, velocities_b, ib, mass_a, mass_b,
                             scale, method, uniforms(1, slot + k),
                             uniforms(2, slot + k), uniforms(3, slot + k),
                             accept < wb, accept < wa);
    }
}
