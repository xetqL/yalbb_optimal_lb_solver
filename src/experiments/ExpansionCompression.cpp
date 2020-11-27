
#include "run.hpp"
int main(int argc, char** argv) {
    constexpr unsigned N = 3;
    using LoadBalancer = Zoltan_Struct;

    auto option = get_params(argc, argv);
    if (!option.has_value()) { exit(EXIT_FAILURE); }
    auto params = option.value();
    const std::array<Real, N> box_center ={params.simsize / 2.0f, params.simsize / 2.0f, params.simsize / 2.0f};

    Boundary<N> boundary     = SphericalBoundary<N>{box_center, params.simsize / 2.0f};

    run<N, LoadBalancer>(argc, argv, experiment::ContractSphere{}, boundary);

    return 0;
}


