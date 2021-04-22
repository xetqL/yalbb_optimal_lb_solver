
#include "run.hpp"
int main(int argc, char** argv) {
    constexpr unsigned N = 3;
    using LoadBalancer = Zoltan_Struct;

    auto option = get_params(argc, argv);

    if (!option.has_value()) { exit(EXIT_FAILURE); }

    auto params = option.value();

    const std::array<Real, 2*N> simbox = {0, params.simsize, 0,params.simsize, 0,params.simsize};

    const std::array<Real, N> box_center ={params.simsize / 2.0f, params.simsize / 2.0f, params.simsize / 2.0f};

    Boundary<N> boundary = SphericalBoundary<N> {box_center, params.simsize / 2.0f};

    auto unaryForceFunc = [params, box_center] (const auto& element, auto fbegin) {
        using namespace vec::generic;
        auto f = normalize(box_center - element.position) * params.G;
        std::copy(f.begin(), f.end(), fbegin);
    };

    experiment::ExpandSphere<N> exp(simbox, params, elements::register_datatype<N>(), MPI_COMM_WORLD, "ExpansionCompresion");

    run<N, LoadBalancer>(argc, argv, exp, boundary, unaryForceFunc);

    return 0;
}


