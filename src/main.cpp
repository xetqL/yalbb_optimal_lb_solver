#include <string>
#include <mpi.h>
#include <random>

#include <yalbb/simulator.hpp>
#include <yalbb/shortest_path.hpp>
#include <yalbb/node.hpp>

#include <yalbb/probe.hpp>
#include <yalbb/ljpotential.hpp>
#include <iomanip>
#include <search.h>

#include "initial_conditions.hpp"
#include "loadbalancing.hpp"
#include "spatial_elements.hpp"
#include "utils.hpp"
#include "experience.hpp"
#include "../../yalbb/includes/yalbb/policy.hpp"
template<typename ... Args>
std::string fmt( const std::string& format, Args ... args )
{
    size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    std::unique_ptr<char[]> buf( new char[ size ] );
    snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

using Config = std::tuple<std::string, std::string, sim_param_t, lb::Criterion>;

int main(int argc, char** argv) {
    constexpr int N = 3;
    int _rank, _nproc;
    float ver;

    std::cout << std::fixed << std::setprecision(6);

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &_nproc);
    const int rank = _rank, nproc = _nproc;
    MPI_Comm APP_COMM;
    MPI_Comm_dup(MPI_COMM_WORLD, &APP_COMM);

    auto option = get_params(argc, argv);
    if (!option.has_value()) {
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    auto params = option.value();
    auto burn_params = option.value();

    params.rc = 2.5f * params.sig_lj;
    params.simsize = std::ceil(params.simsize / params.rc) * params.rc;
    burn_params.npart   = params.npart * 0.1f;
    burn_params.nframes = 40;
    burn_params.npframe = 50;

    const std::array<Real, 2*N> simbox      = {0, params.simsize, 0,params.simsize, 0,params.simsize};
    const std::array<Real, N>   simlength   = {params.simsize, params.simsize, params.simsize};
    const std::array<Real, N>   box_center  = {params.simsize / (Real) 2.0,params.simsize / (Real)2.0,params.simsize / (Real)2.0};
    const std::array<Real, N>   singularity = {params.simsize / (Real) 2.0,params.simsize / (Real)2.0,params.simsize / (Real)2.0};

    MPI_Bcast(&params.seed, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        print_params(params);
        std::cout << "Computating with " << nproc << " PEs"<<std::endl;
    }

    if(Zoltan_Initialize(argc, argv, &ver) != ZOLTAN_OK) {
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////START PARITCLE INITIALIZATION///////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    auto datatype = elements::register_datatype<N>();
    // Data getter function (position and velocity) *required*
    auto getPositionPtrFunc = [](auto* e) -> std::array<Real, N>* { return &e->position; };
    auto getVelocityPtrFunc = [](auto* e) -> std::array<Real, N>* { return &e->velocity; };
    // Short range force function computation
    auto getForceFunc = [eps=params.eps_lj, sig=params.sig_lj, rc=params.rc, getPositionPtrFunc](const auto* receiver, const auto* source)->std::array<Real, N>{
        return lj_compute_force<N>(receiver, source, eps, sig*sig, rc, getPositionPtrFunc);
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////FINISHED PARITCLE INITIALIZATION///////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    using LoadBalancer       = Zoltan_Struct;

    using DoLoadBalancing    = lb::DoLB<LoadBalancer, decltype(getPositionPtrFunc)>;
    // Domain-box intersection function *required*
    // Solve interactions
    auto boxIntersectFunc    = lb::IntersectDomain<LoadBalancer> {params.rc};
    // Point-in-domain callback *required*
    // Solve belongings
    auto pointAssignFunc     = lb::AssignPoint<LoadBalancer> {};

    auto doPartition         = lb::DoPartition<LoadBalancer> {};
    // Partitioning + migration function *required*
    // Solve partitioning
    auto doLoadBalancingFunc = DoLoadBalancing(params.rc, datatype, APP_COMM, getPositionPtrFunc);

    auto destroyLB           = lb::Destroyer<LoadBalancer>{};
    auto createLB            = lb::Creator<LoadBalancer>{};
    using Particle           = elements::Element<N>;
    using Experiment         = experiment::experiment_t<N, LoadBalancer, decltype(doPartition), decltype(getPositionPtrFunc), decltype(pointAssignFunc)>;
    Experiment initExperiment= experiment::Expand2DSphere;
    Boundary<N> boundary     = SphericalBoundary<N>{box_center, params.simsize / 10.0f};
    FunctionWrapper fWrapper(getPositionPtrFunc, getVelocityPtrFunc, getForceFunc, boxIntersectFunc, pointAssignFunc, doLoadBalancingFunc);

    double load_balancing_cost = 0;
    double load_balancing_parallel_efficiency = 0;

    std::vector<Config> configs;

    /** Burn CPU cycle */
    {
        MPI_Comm APP_COMM;
        MPI_Comm_dup(MPI_COMM_WORLD, &APP_COMM);

        auto zlb = createLB(APP_COMM);
        auto[mesh_data, probe, lbtime, exp_name] = initExperiment(zlb, simbox, burn_params, datatype, APP_COMM, getPositionPtrFunc, pointAssignFunc, doPartition, "Burn CPU Cycle:");
        simulate<N>(zlb, &mesh_data, lb::Static {}, boundary, fWrapper, &burn_params, &probe, datatype, APP_COMM, "BURN");
        destroyLB(zlb);
    }

    std::vector<int> opt_scenario;
    Probe solution_stats(nproc);
    if(params.nb_best_path) {
        auto zlb = createLB(APP_COMM);
        auto[mesh_data, probe, lbtime, exp_name] = initExperiment(zlb, simbox, params, datatype, APP_COMM, getPositionPtrFunc, pointAssignFunc, doPartition, "A*\n");
        const auto simulation_name = params.prefix + "/" + exp_name + "/Astar";
        std::tie(solution_stats,opt_scenario) = simulate_shortest_path<N>(zlb, &mesh_data, boundary, fWrapper, &params, datatype,
                      lb::Copier<LoadBalancer>{},
                      lb::Destroyer<LoadBalancer>{}, APP_COMM, simulation_name);
        load_balancing_cost = solution_stats.compute_avg_lb_time();
        configs.emplace_back("AstarReproduce\n", "AstarReproduce",  params, lb::Reproduce{opt_scenario});
    }

    configs.emplace_back("Static",              "Static",           params, lb::Static{});
    // Periodic
    configs.emplace_back("Periodic 1000",       "Periodic_1000",    params, lb::Periodic{1000});
    configs.emplace_back("Periodic 500",        "Periodic_500",     params, lb::Periodic{500});
    configs.emplace_back("Periodic 250",        "Periodic_250",     params, lb::Periodic{250});
    configs.emplace_back("Periodic 100",        "Periodic_100",     params, lb::Periodic{100});
    configs.emplace_back("Periodic 50",         "Periodic_50",      params, lb::Periodic{50});
    // Menon-like criterion
    configs.emplace_back("VanillaMenon",        "VMenon",           params, lb::VanillaMenon{});
    configs.emplace_back("OfflineMenon",        "OMenon",           params, lb::OfflineMenon{});
    configs.emplace_back("ImprovedMenon",       "IMenon",           params, lb::ImprovedMenon{});
    configs.emplace_back("PositivMenon",        "PMenon",           params, lb::ImprovedMenonNoMax{});
    configs.emplace_back("ZhaiMenon",           "ZMenon",           params, lb::ZhaiMenon{});
    // Procassini
    configs.emplace_back("Procassini 145",      "Procassini_145p",  params, lb::Procassini{1.45});
    configs.emplace_back("Procassini 120",      "Procassini_120p",  params, lb::Procassini{1.20});
    configs.emplace_back("Procassini 115",      "Procassini_115p",  params, lb::Procassini{1.15});
    configs.emplace_back("Procassini 100",      "Procassini_100p",  params, lb::Procassini{1.00});
    configs.emplace_back("Procassini 90",       "Procassini_90p",   params, lb::Procassini{0.95});
    // Marquez
    configs.emplace_back("Marquez 145",         "Marquez_145",      params, lb::Marquez{1.45});
    configs.emplace_back("Marquez 125",         "Marquez_125",      params, lb::Marquez{1.25});
    configs.emplace_back("Marquez 85",          "Marquez_85",       params, lb::Marquez{0.85});
    configs.emplace_back("Marquez 65",          "Marquez_65",       params, lb::Marquez{0.65});

    for(auto& cfg : configs){
        auto& [preamble, config_name, params, criterion] = cfg;
        auto zlb = createLB(APP_COMM);
        auto[mesh_data, probe, lbtime, exp_name] = initExperiment(zlb, simbox, params, datatype, APP_COMM, getPositionPtrFunc, pointAssignFunc, doPartition, preamble);
        const std::string simulation_name = params.prefix + "/" + exp_name + "/" + config_name;
        probe.push_load_balancing_time(load_balancing_cost);
        probe.push_load_balancing_parallel_efficiency(1.0);
        simulate<N>(zlb, &mesh_data, criterion, boundary, fWrapper, &params, &probe, datatype, APP_COMM, simulation_name);
        destroyLB(zlb);
    }

    MPI_Finalize();
    return 0;

}
