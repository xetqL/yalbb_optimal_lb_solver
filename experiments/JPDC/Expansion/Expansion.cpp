#include <yalbb/run.hpp>
#include "parser.hpp"
#include "lb_zoltan.hpp"
#include "experience.hpp"
int main(int argc, char** argv) {
    constexpr unsigned N = YALBB_DIMENSION;

    YALBB yalbb(argc, argv);

    Parser parser;
    std::unique_ptr<param_t> params = parser.get_params(argc, argv);

    if (!params) { exit(EXIT_FAILURE); }

    const std::array<Real, 2*N> simbox = get_simbox<N>(params->simsize);

    const auto center = get_box_center<N>(simbox);

    Boundary<N> boundary = SphericalBoundary<N>{center, params->simsize /2};

    auto unaryForce = [](const auto& element, auto fbegin) {};

    auto binaryForce = [eps=params->eps_lj, sig=params->sig_lj, rc=params->rc](const auto* receiver, const auto* source)->std::array<Real, N>{
        auto getPosFunc = [](auto* e) { return &(e->position); } ;
        return lj_compute_force<N>(receiver, source, eps, sig*sig, rc, getPosFunc);
    };

    auto datatype           = elements::Element<N>::register_datatype();
    experiment::ExpandSphere<N, param_t> exp(simbox, params, datatype, MPI_COMM_WORLD, "Expansion");

    auto getPositionPtrFunc = elements::Element<N>::getElementPositionPtr;
    auto getVelocityPtrFunc = elements::Element<N>::getElementVelocityPtr;

    run<N, Zoltan_Struct>(yalbb, params.get(), exp, boundary, "HSFC", datatype, getPositionPtrFunc, getVelocityPtrFunc,binaryForce, unaryForce, [APP_COMM=yalbb.comm](){
        float ver;

        const char* LB_METHOD = "HSFC";

        if(Zoltan_Initialize(0, nullptr, &ver) != ZOLTAN_OK) {
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        auto zz = Zoltan_Create(APP_COMM);

        Zoltan_Set_Param(zz, "DEBUG_LEVEL", "0");
        Zoltan_Set_Param(zz, "LB_METHOD", LB_METHOD);
        Zoltan_Set_Param(zz, "DETERMINISTIC", "1");
        Zoltan_Set_Param(zz, "NUM_GID_ENTRIES", "1");

        Zoltan_Set_Param(zz, "NUM_LID_ENTRIES", "1");
        Zoltan_Set_Param(zz, "OBJ_WEIGHT_DIM", "0");
        Zoltan_Set_Param(zz, "RCB_REUSE", "1");
        Zoltan_Set_Param(zz, "RETURN_LISTS", "ALL");

        Zoltan_Set_Param(zz, "RCB_OUTPUT_LEVEL", "0");
        Zoltan_Set_Param(zz, "KEEP_CUTS", "1");

        Zoltan_Set_Param(zz, "AUTO_MIGRATE", "FALSE");

        return zz;
    });

    return 0;
}
