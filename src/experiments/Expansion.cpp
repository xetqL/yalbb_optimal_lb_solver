#include "run.hpp"
#include <yalbb/boundary.hpp>
int main(int argc, char** argv) {
    constexpr unsigned N = 3;
    using LoadBalancer = Zoltan_Struct;

    auto option = get_params(argc, argv);
    if (!option.has_value()) { exit(EXIT_FAILURE); }
    auto params = option.value();
    const std::array<Real, 2*N> simbox = {0, params.simsize, 0,params.simsize, 0,params.simsize};

    std::cout << simbox << std::endl;

    Boundary<N> boundary = CubicalBoundary<N>(simbox, params.bounce);

    auto unaryForceFunc = [params](const auto& element, auto fbegin) {};



    run<N, LoadBalancer>(argc, argv, experiment::ExpandSphere{}, boundary, unaryForceFunc);

    return 0;
}