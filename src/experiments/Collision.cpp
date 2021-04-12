#include "run.hpp"
int main(int argc, char** argv) {
    constexpr unsigned N = 2;
    using LoadBalancer = Zoltan_Struct;

    auto option = get_params(argc, argv);
    if (!option.has_value()) { exit(EXIT_FAILURE); }
    auto params = option.value();

    const std::array<Real, 2*N> simbox = get_simbox<N>(params.simsize);

    Boundary<N> boundary = CubicalBoundary<N>{simbox, params.bounce};

    auto unaryForceFunc = [params](const auto& element, auto fbegin) {};

    run<N, LoadBalancer>(argc, argv, experiment::CollidingSphere{}, boundary, unaryForceFunc);

    return 0;
}