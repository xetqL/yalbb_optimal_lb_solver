#include "run.hpp"
#include "parser.hpp"
#include "rcb.hpp"

int main(int argc, char** argv) {
    constexpr unsigned N = YALBB_DIMENSION;

    static_assert(N==2);

    YALBB yalbb(argc, argv);

    Parser parser;
    std::unique_ptr<param_t> params = parser.get_params(argc, argv);

    // benchmark params
    params->nframes = 1;
    params->npframe = 1;
//    params->record  = false;
    params->monitor = true;
    params->burn    = true;

    params->nb_best_path = 0;

    if (!params) { exit(EXIT_FAILURE); }

    const std::array<Real, 2*N> simbox = get_simbox<N>(params->simsize);

    const auto center = get_box_center<N>(simbox);

    Boundary<N> boundary = CubicalBoundary<N>{simbox, params->bounce};

    auto unaryForce = [G=params->G](const auto& element, auto fbegin) {
        *(fbegin + 1) += -G;
    };

    auto binaryForce = [eps=params->eps_lj, sig=params->sig_lj, rc=params->rc](const auto* receiver, const auto* source)->std::array<Real, N>{
        return {0.0,0.0};
    };

    experiment::UniformCube<N, param_t> exp(simbox, params, elements::register_datatype<N>(), MPI_COMM_WORLD, "");

    run<N, rcb::RCB>(yalbb, params.get(), exp, boundary, binaryForce, unaryForce, [APP_COMM=MPI_COMM_WORLD, &params](){
        return new rcb::RCB(init_domain<Real>(
                -params->simsize, -params->simsize, 2*params->simsize, 2*params->simsize), APP_COMM);
    });

    run<N, norcb::NoRCB>(yalbb, params.get(), exp, boundary, binaryForce, unaryForce, [APP_COMM=MPI_COMM_WORLD, &params](){
        return new norcb::NoRCB(init_domain<Real>(
                -params->simsize, -params->simsize, 2*params->simsize, 2*params->simsize), APP_COMM);
    });

    return 0;
}






