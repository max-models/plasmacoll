// The scattering of one pair of the binary Coulomb collisions, shared by the
// CUDA versions of coulomb_collide_intra and coulomb_collide_inter (the same
// arithmetic as their Pyccel versions' _cos_chi and _scatter_pair).
#ifndef PLASMACOLL_COULOMB_PAIR_CUH
#define PLASMACOLL_COULOMB_PAIR_CUH

#include "cunumpy/array_view.cuh"

#define PLASMACOLL_TWO_PI 6.283185307179586

// cos(chi) for the scattering parameter s: method 0 Takizuka-Abe (tan(chi/2)
// normal with variance s/2), 1 Nanbu with the fit of Perez et al. (2012).
__device__ __forceinline__ double coulomb_cos_chi(double s, long long method,
                                                  double r1, double r2) {
    if (method == 0) {
        const double delta = sqrt(-s * log(r1)) * cos(PLASMACOLL_TWO_PI * r2);
        const double delta2 = delta * delta;
        return (1.0 - delta2) / (1.0 + delta2);
    }
    double cos_chi;
    if (s < 0.1) {
        cos_chi = 1.0 + s * log(r1);
    } else if (s < 6.0) {
        double a, inverse;
        if (s < 3.0) {
            inverse = 0.0056958 + s * (0.9560202 + s * (-0.508139 + s * (0.47913906
                      + s * (-0.12788975 + s * 0.02389567))));
            a = 1.0 / inverse;
        } else {
            a = 3.0 * exp(-s);
            inverse = 1.0 / a;
        }
        cos_chi = inverse * log(exp(-a) + 2.0 * r1 * sinh(a));
    } else {
        cos_chi = 2.0 * r1 - 1.0;
    }
    if (cos_chi < -1.0) cos_chi = -1.0;
    if (cos_chi > 1.0) cos_chi = 1.0;
    return cos_chi;
}

__device__ __forceinline__ void coulomb_scatter_pair(
    Array2D<double> va, long long ia, Array2D<double> vb, long long ib,
    double mass_a, double mass_b, double coef, long long method,
    double r1, double r2, double r3, bool update_a, bool update_b) {
    const double ux = va(0, ia) - vb(0, ib);
    const double uy = va(1, ia) - vb(1, ib);
    const double uz = va(2, ia) - vb(2, ib);
    const double u2 = ux * ux + uy * uy + uz * uz;
    if (u2 <= 0.0) return;
    const double u = sqrt(u2);
    const double s = coef / (u2 * u);
    const double cos_chi = coulomb_cos_chi(s, method, r1, r2);
    const double one_minus = 1.0 - cos_chi;
    const double sin_chi = sqrt(fmax(0.0, (1.0 - cos_chi) * (1.0 + cos_chi)));
    const double phi = PLASMACOLL_TWO_PI * r3;
    const double cos_phi = cos(phi);
    const double sin_phi = sin(phi);
    const double u_perp = sqrt(ux * ux + uy * uy);
    double dux, duy, duz;
    if (u_perp > 1.0e-12 * u) {
        dux = (ux / u_perp) * uz * sin_chi * cos_phi
              - (uy / u_perp) * u * sin_chi * sin_phi - ux * one_minus;
        duy = (uy / u_perp) * uz * sin_chi * cos_phi
              + (ux / u_perp) * u * sin_chi * sin_phi - uy * one_minus;
        duz = -u_perp * sin_chi * cos_phi - uz * one_minus;
    } else {
        dux = u * sin_chi * cos_phi;
        duy = u * sin_chi * sin_phi;
        duz = -uz * one_minus;
    }
    const double total = mass_a + mass_b;
    if (update_a) {
        const double fa = mass_b / total;
        va(0, ia) += fa * dux;
        va(1, ia) += fa * duy;
        va(2, ia) += fa * duz;
    }
    if (update_b) {
        const double fb = mass_a / total;
        vb(0, ib) -= fb * dux;
        vb(1, ib) -= fb * duy;
        vb(2, ib) -= fb * duz;
    }
}

// A random permutation of start .. start + count - 1 (Fisher-Yates with the
// uniforms of row 0 from column slot).
__device__ __forceinline__ void coulomb_shuffle(Array1D<long long> order,
                                                Array2D<double> uniforms,
                                                long long start, long long slot,
                                                long long count) {
    for (long long k = 0; k < count; ++k) order(start + k) = start + k;
    for (long long k = count - 1; k > 0; --k) {
        long long j = (long long)(uniforms(0, slot + k) * (double)(k + 1));
        if (j > k) j = k;
        const long long swap = order(start + k);
        order(start + k) = order(start + j);
        order(start + j) = swap;
    }
}

#endif
