#include "run.hpp"
#include "parser.hpp"
#include <yalbb/yalbb.hpp>
int main(int argc, char** argv) {
    constexpr unsigned N = YALBB_DIMENSION;
    static_assert(N==2);
    YALBB yalbb(argc, argv);

    Parser parser;
    std::unique_ptr<param_t> params = parser.get_params(argc, argv);

    if (!params) { exit(EXIT_FAILURE); }

    const std::array<Real, 2*N> simbox = get_simbox<N>(params->simsize);

    const auto center = get_box_center<N>(simbox);

    Boundary<N> boundary = SphericalBoundary<N>{center, params->simsize /2};

    auto unaryForce = [G=params->G, center] (const auto& element, auto fbegin) {
        using namespace vec::generic;
        // auto f = normalize(center - element.position) * G;
        std::array<Real, 2> f = {0, -G};
        std::copy(f.begin(), f.end(), fbegin);
    };

    auto binaryForce = [eps=params->eps_lj, sig=params->sig_lj, rc=params->rc](const auto* receiver, const auto* source)->std::array<Real, N>{
        auto getPosFunc = [](auto* e) { return &(e->position); } ;
        return lj_compute_force<N>(receiver, source, eps, sig*sig, rc, getPosFunc);
    };

    experiment::GravityCircle<N, param_t> exp(simbox, params, elements::register_datatype<N>(), MPI_COMM_WORLD, "GravityCircle");
    run<N, norcb::NoRCB>(yalbb, params.get(), exp, boundary, binaryForce, unaryForce, [APP_COMM=MPI_COMM_WORLD, &params](){
        return new norcb::NoRCB(init_domain<Real>(
                -params->simsize, -params->simsize, 2*params->simsize, 2*params->simsize), APP_COMM);
    });

    run<N, rcb::RCB>(yalbb, params.get(), exp, boundary, binaryForce, unaryForce, [APP_COMM=MPI_COMM_WORLD, &params](){
        return new rcb::RCB(init_domain<Real>(
                -params->simsize, -params->simsize, 2*params->simsize, 2*params->simsize), APP_COMM);
    });

    return 0;
}


