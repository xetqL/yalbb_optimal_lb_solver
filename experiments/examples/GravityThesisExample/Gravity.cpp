// YALBB entry point
#include <yalbb/run.hpp>
// add custom parsing capabilities
#include "parser.hpp"
// define your custom LB functions
#include "lb_custom.hpp"
// define the initial state of your elements
#include "experience.hpp"
// Syntactic sugar
template<unsigned N_DIM>
using MyElement = BaseElement<N_DIM>;

int main(int argc, char** argv) {
    constexpr unsigned N = YALBB_DIMENSION;
    YALBB yalbb(argc, argv);
    Parser parser {};
    auto params = parser.get_params(argc, argv);
    if (!params) { exit(EXIT_FAILURE); }
    const std::array<Real, 2*N> simbox = get_simbox<N>(params->simsize);
    const auto center = get_box_center<N>(simbox);
    auto boundary     = CubicalBoundary<N>{simbox, 1.0};

    auto unaryForce   = [G=params->G] (const auto& element, auto fbegin) { *(fbegin + 1) += -G; };

    auto binaryForce  = [eps=params->eps_lj, sig=params->sig_lj, rc=params->rc] (const auto* ei, const auto* ej) {
        return lj_compute_force<N>(ei, ej, eps, sig*sig, rc, MyElement<N>::getElementPositionPtr);
    };

    experiment::UniformCube<N, param_t> exp(simbox, params,
                                            MyElement<N>::register_datatype(), yalbb.comm,
                                            "Gravity");
    run<N, Zoltan_Struct>(yalbb, params.get(), exp, boundary, "HSFC",
      MyElement<N>::register_datatype(), MyElement<N>::getElementPositionPtr, MyElement<N>::getElementVelocityPtr,
      binaryForce, unaryForce, [] () {
        // ...
        /* Create and return a Zoltan_Struct* */
    });
    return 0;
}






