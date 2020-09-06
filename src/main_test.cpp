//
// Created by xetql on 6/29/20.
//
#include <yalbb/ljpotential.hpp>
#include <iomanip>
#include "../zoltan/zoltan_fn.hpp"
#include "spatial_elements.hpp"
#include <string>
#include <random>
#include <yalbb/params.hpp>

template<int N, class T, class GetPosPtrFunc>
std::array<Real, N> test_lj_compute_force(const T* receiver, const T* source, Real eps, Real sig2, Real rc, GetPosPtrFunc getPosPtr) {
    Real r2 = 0.0;

    std::array<Real, N> delta_dim;
    std::array<Real, N> force;

    const auto rec_pos = getPosPtr(const_cast<T*>(receiver));
    const auto sou_pos = getPosPtr(const_cast<T*>(source));

    for (int dim = 0; dim < N; ++dim) delta_dim[dim] = rec_pos->at(dim) - sou_pos->at(dim);
    for (int dim = 0; dim < N; ++dim) r2 += (delta_dim[dim] * delta_dim[dim]);

    const Real min_r2 = (rc*rc) / 10000.0;
    auto sig6 = sig2*sig2*sig2;
    //delta = std::max(delta, min_r2);

    //const Real C_LJ = -compute_LJ_scalar(delta, eps, sig2, rc*rc);
    auto C_LJ = ((24.0*eps*(sig6)) / (r2*r2*r2*r2)) * (1.0 - (2.0*sig6)*r2/(r2*r2*r2*r2));

    for (int dim = 0; dim < N; ++dim) {
        force[dim] = (C_LJ);
    }

    return force;
}

Real my_compute_LJ_scalar(Real r2, Real eps, Real sig, Real rc2) {
    Real lj1 = 48.0 * eps * std::pow(sig,12.0);
    Real lj2 = 24.0 * eps * std::pow(sig,6.0);
    Real r2i = 1.0 / r2;
    Real r6i = r2i*r2i*r2i;
    return r6i * (lj1 * r6i - lj2) * r2i;
}

int main(int argc, char** argv){
    constexpr int N = 2;

    std::cout << std::fixed << std::setprecision(12);

    auto option = get_params(argc, argv);
    if (!option.has_value()) {
        exit(EXIT_FAILURE);
    }
    auto params = option.value();


    params.simsize = std::ceil(params.simsize / params.rc) * params.rc;

    print_params(params);

    // Data getter function (position and velocity) *required*
    auto getPositionPtrFunc = [](auto* e) -> std::array<Real, N>* { return &e->position; };
    auto getVelocityPtrFunc = [](auto* e) -> std::array<Real, N>* { return &e->velocity; };
    // Short range force function computation
    auto getForceFunc = [eps=params.eps_lj, sig=params.sig_lj, rc=params.rc, getPositionPtrFunc](const auto* receiver, const auto* source){
        return lj_compute_force<N>(receiver, source, eps, sig*sig, rc, getPositionPtrFunc);
    };

    constexpr int NUMBER_OF_SAMPLES = 1e4;
    constexpr auto min = 0.0;
    const Real max = params.simsize;
    BoundingBox<N> bbox;
    for(int i = 0; i < N; ++i) {
        bbox[2*i]   = min;
        bbox[2*i+1] = params.simsize;
    }
    std::cout << get_total_cell_number<N>(bbox, params.rc) << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Real> dist(max/2.0f, max/2.0f + 2.5*params.sig_lj);


    std::array<Real, N> ref_vel;
    std::array<Real, N> ref_pos;
    std::generate(ref_vel.begin(), ref_vel.end(), [](){return 0.0f;});
    std::generate(ref_pos.begin(), ref_pos.end(), [max](){return max/2.0f;});

    elements::Element<N> ref(ref_pos, ref_vel, 0, 0);
    std::vector<elements::Element<N>> local = {ref};
    std::vector<elements::Element<N>> remote;

    std::vector<Index> lscl(2), head;
    std::vector<Real> flocal(N);
    std::ofstream fforce;
    fforce.open("force.txt");
    std::ofstream fpoints;
    fpoints.open("points.txt");
    std::array<Real, N> delta_dim;
    fpoints << ref_pos << std::endl;
    Real sig2 = params.sig_lj*params.sig_lj;
    for(int i = 0; i < NUMBER_OF_SAMPLES; ++i){
        Real r2 = 0.0;
        std::array<Real, N> pos;

        std::generate(pos.begin(), pos.end(), [&gen, &d=dist](){return d(gen);});

        fpoints << pos << std::endl;

        elements::Element<N> p(pos, ref_vel, 1, 1);

        remote = {p};

        for (int dim = 0; dim < N; ++dim) delta_dim[dim] = ref_pos.at(dim) - pos.at(dim);
        for (int dim = 0; dim < N; ++dim) r2 += (delta_dim[dim] * delta_dim[dim]);

        //r2 = 1e-4 + i*(0.0075)/NUMBER_OF_SAMPLES;
        CLL_init<N, elements::Element<N>> ({{local.data(), 1}, {remote.data(), 1}}, getPositionPtrFunc, bbox, params.rc, &head, &lscl);

        std::fill(flocal.begin(), flocal.end(), (Real) 0.0);

        CLL_compute_forces<N, elements::Element<N>>(&flocal, local, remote, getPositionPtrFunc, bbox, params.rc, &head, &lscl, getForceFunc);

        //fforce << my_compute_LJ_scalar(r2, 1.0, 1e-2, 10) << " " << r2 << std::endl;
        fforce << (r2 < (6.25*sig2) ? -my_compute_LJ_scalar(r2, 1.0, 1e-2, 10) : 0.0) << " " << r2 << std::endl;

    }

    fforce.close();
    fpoints.close();
}