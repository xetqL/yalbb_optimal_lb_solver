#include "run.hpp"
int main(int argc, char** argv) {
    constexpr unsigned N = 3;
    using LoadBalancer = Zoltan_Struct;

    auto option = get_params(argc, argv);
    if (!option.has_value()) { exit(EXIT_FAILURE); }
    auto params = option.value();

    const auto simbox = get_simbox<N>(params.simsize);

    Boundary<N> boundary = CubicalBoundary<N>{simbox, params.bounce};

    auto unaryForceFunc = [params](const auto& element, auto fbegin) {};

    experiment::ContractSphere<N> exp(simbox, params, elements::register_datatype<N>(), MPI_COMM_WORLD, "contract");

    run<N, LoadBalancer>(argc, argv, exp, boundary, unaryForceFunc);

    return 0;
}
