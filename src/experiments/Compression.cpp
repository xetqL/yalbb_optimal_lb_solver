#include "run.hpp"
int main(int argc, char** argv) {
    constexpr unsigned N = 3;
    using LoadBalancer = Zoltan_Struct;

    auto option = get_params(argc, argv);
    if (!option.has_value()) { exit(EXIT_FAILURE); }
    auto params = option.value();
    const std::array<Real, 2*N> simbox = {0, params.simsize, 0,params.simsize, 0,params.simsize};

    Boundary<N> boundary = CubicalBoundary<N>{simbox, params.bounce};

    run<N, LoadBalancer>(argc, argv, experiment::ContractSphere{}, boundary);

    return 0;
}
