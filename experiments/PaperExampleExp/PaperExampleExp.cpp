#include "run.hpp"
#include "parser.hpp"
#include <yalbb/yalbb.hpp>
int main(int argc, char** argv) {
    constexpr unsigned N = YALBB_DIMENSION;

    YALBB yalbb(argc, argv);

    Parser parser;
    std::unique_ptr<param_t> params = parser.get_params(argc, argv);

    if (!params) { exit(EXIT_FAILURE); }

    const std::array<Real, 2*N> simbox = get_simbox<N>(params->simsize);
    Boundary<N> boundary = CubicalBoundary<N>{simbox, params->bounce};

    auto unaryForce = [](const auto& element, auto fbegin) {};

    auto binaryForce = [eps=params->eps_lj, sig=params->sig_lj, rc=params->rc](const auto* receiver, const auto* source)->std::array<Real, N>{
        auto getPosFunc = [](auto* e) { return &(e->position); } ;
        return lj_compute_force<N>(receiver, source, eps, sig*sig, rc, getPosFunc);
    };

    experiment::HalfUniformHalfOrthogonal<N, param_t> exp(simbox, params, elements::register_datatype<N>(), MPI_COMM_WORLD, "PaperExample");

    run<N, norcb::NoRCB>(yalbb, params.get(), exp, boundary, "NoRCB", binaryForce, unaryForce, [APP_COMM=yalbb.comm, &params](){
        return new norcb::NoRCB(init_domain<Real>(
                -params->simsize, -params->simsize, 2*params->simsize, 2*params->simsize), APP_COMM);
    });

    run<N, Zoltan_Struct>(yalbb, params.get(), exp, boundary, "HSFC", binaryForce, unaryForce, [APP_COMM=yalbb.comm](){
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

    run<N, Zoltan_Struct>(yalbb, params.get(), exp, boundary, "RCB",  binaryForce, unaryForce, [APP_COMM=yalbb.comm](){
        float ver;

        const char* LB_METHOD = "RCB";

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

    run<N, Zoltan_Struct>(yalbb, params.get(), exp, boundary, "RIB",  binaryForce, unaryForce, [APP_COMM=yalbb.comm](){
        float ver;

        const char* LB_METHOD = "RIB";

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
