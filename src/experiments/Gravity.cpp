#include "run.hpp"
int main(int argc, char** argv) {
    constexpr unsigned N = 2;
    using LoadBalancer = Zoltan_Struct;

    auto option = get_params(argc, argv);

    if (!option.has_value()) { exit(EXIT_FAILURE); }

    auto params = option.value();

    const std::array<Real, 2*N> simbox = get_simbox<N>(params.simsize);

    Boundary<N> boundary = CubicalBoundary<N>{simbox, params.bounce};

    auto unaryForceFunc = [params](const auto& element, auto fbegin) {
        using namespace vec::generic;
        std::array<Real, N> f = {0., params.G};
        std::copy(f.begin(), f.end(), fbegin);
    };

    experiment::UniformCube<N> exp(simbox, params, elements::register_datatype<N>(), MPI_COMM_WORLD, "ExpansionCompresion");

    run<N, LoadBalancer>(argc, argv, exp, boundary, unaryForceFunc);

    return 0;
}