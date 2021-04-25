#include "run.hpp"
#include "parser.hpp"
int main(int argc, char** argv) {
    constexpr unsigned N = 3;
    YALBB yalbb(argc, argv);

    Parser parser;
    std::unique_ptr<param_t> params = parser.get_params(argc, argv);

    if (!params) { exit(EXIT_FAILURE); }

    const std::array<Real, 2*N> simbox = get_simbox<N>(params->simsize);

    const auto center = get_box_center<N>(simbox);

    Boundary<N> boundary = CubicalBoundary<N>{simbox, params->bounce};

    auto unaryForce = [G=params->G](const auto& element, auto fbegin) {
        using namespace vec::generic;
        *(fbegin + (N-1)) = G;
    };

    auto binaryForce = [eps=params->eps_lj, sig=params->sig_lj, rc=params->rc](const auto* receiver, const auto* source)->std::array<Real, N>{
        auto getPosFunc = [](auto* e) { return &(e->position); } ;
        return lj_compute_force<N>(receiver, source, eps, sig*sig, rc, getPosFunc);
    };

    experiment::UniformCube<N, param_t> exp(simbox, params, elements::register_datatype<N>(), MPI_COMM_WORLD, "ExpansionCompresion");

    if constexpr (N<3){
        run<N, norcb::NoRCB>(yalbb, params.get(), exp, boundary, binaryForce, unaryForce, [APP_COMM=MPI_COMM_WORLD, &params](){
            auto lb_ptr = new norcb::NoRCB(norcb::init_domain<Real>(
                    -200, -200,200, 200), APP_COMM);
            return lb_ptr;
        });
    }

    run<N, Zoltan_Struct>(yalbb, params.get(), exp, boundary, binaryForce, unaryForce, [APP_COMM=MPI_COMM_WORLD](){
        float ver;
        if(Zoltan_Initialize(0, nullptr, &ver) != ZOLTAN_OK) {
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
        return zoltan_create_wrapper(APP_COMM);
    });

    return 0;
}






