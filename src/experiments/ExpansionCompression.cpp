#include <string>
#include <mpi.h>
#include <random>
#include <iomanip>

#include <yalbb/simulator.hpp>
#include <yalbb/shortest_path.hpp>
#include <yalbb/policy.hpp>
#include <yalbb/probe.hpp>
#include <yalbb/ljpotential.hpp>

#include "initial_conditions.hpp"
#include "loadbalancing.hpp"
#include "experience.hpp"

using Config = std::tuple<std::string, std::string, sim_param_t, lb::Criterion>;

int main(int argc, char** argv) {
    constexpr int N = 3;
    int nproc;
    float ver;

    std::cout << std::fixed << std::setprecision(6);

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
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

    burn_params.npart   = (int) (params.npart * 0.1);
    burn_params.nframes = 40;
    burn_params.npframe = 50;
    burn_params.monitor = false;

    const std::array<Real, 2*N> simbox      = {0, params.simsize, 0,params.simsize, 0,params.simsize};
    const std::array<Real,   N> simlength   = {params.simsize, params.simsize, params.simsize};
    const std::array<Real,   N> box_center  = {params.simsize / (Real) 2.0,params.simsize / (Real)2.0,params.simsize / (Real)2.0};
    const std::array<Real,   N> singularity = {params.simsize / (Real) 2.0,params.simsize / (Real)2.0,params.simsize / (Real)2.0};

    MPI_Bcast(&params.seed, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(Zoltan_Initialize(argc, argv, &ver) != ZOLTAN_OK) {
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Particles function definition
    auto datatype = elements::register_datatype<N>();
    // Getter (position and velocity)
    auto getPositionPtrFunc = [](auto* e) -> std::array<Real, N>* { return &e->position; };
    auto getVelocityPtrFunc = [](auto* e) -> std::array<Real, N>* { return &e->velocity; };
    // Short range force function computation
    auto getForceFunc = [eps=params.eps_lj, sig=params.sig_lj, rc=params.rc, getPositionPtrFunc](const auto* receiver, const auto* source)->std::array<Real, N>{
        return lj_compute_force<N>(receiver, source, eps, sig*sig, rc, getPositionPtrFunc);
    };
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Specify the load balancer
    using LoadBalancer       = Zoltan_Struct;
    // Who are my neighbors ?
    auto boxIntersectFunc    = lb::IntersectDomain<LoadBalancer> {params.rc};
    // Which domain this point belongs to?
    auto pointAssignFunc     = lb::AssignPoint<LoadBalancer> {};
    // Partition the domain without migration
    auto doPartition         = lb::DoPartition<LoadBalancer> {};
    // Load balance workload among CPUs
    auto doLoadBalancingFunc = lb::DoLB<LoadBalancer, decltype(getPositionPtrFunc)>(params.rc, datatype, APP_COMM, getPositionPtrFunc);
    // Destroy this LB struct
    auto destroyLB           = lb::Destroyer<LoadBalancer>{};
    // Create a LB struct
    auto createLB            = lb::Creator<LoadBalancer>{};

    // Specify the experiment you want
    using Experiment         = experiment::experiment_t<N, LoadBalancer, decltype(doPartition), decltype(getPositionPtrFunc), decltype(pointAssignFunc)>;
    Experiment initExperiment= experiment::ExpandSphere;
    // Specify the type of boundary
    Boundary<N> boundary     = SphericalBoundary<N>{box_center, params.simsize / 2.0f};
    // Wrap everything
    FunctionWrapper fWrapper(getPositionPtrFunc, getVelocityPtrFunc, getForceFunc, boxIntersectFunc, pointAssignFunc, doLoadBalancingFunc);

    double load_balancing_cost = 0;
    double load_balancing_parallel_efficiency = 0;

    std::vector<Config> configs{};

    /** Burn CPU cycle */
    {
        MPI_Comm APP_COMM;
        MPI_Comm_dup(MPI_COMM_WORLD, &APP_COMM);
        auto zlb = createLB(APP_COMM);
        auto[mesh_data, probe, lbtime, exp_name] = initExperiment(zlb, simbox, burn_params, datatype, APP_COMM, getPositionPtrFunc, pointAssignFunc, doPartition, "Burn CPU Cycle:");
        simulate<N>(zlb, &mesh_data, lb::Static{}, boundary, fWrapper, &burn_params, &probe, datatype, APP_COMM, "BURN");
        destroyLB(zlb);
    }

    if(params.nb_best_path) {
        Probe solution_stats(nproc);
        std::vector<int> opt_scenario{};
        auto zlb = createLB(APP_COMM);
        auto[mesh_data, probe, lbtime, exp_name] = initExperiment(zlb, simbox, params, datatype, APP_COMM, getPositionPtrFunc, pointAssignFunc, doPartition, "A*\n");
        const auto simulation_name = fmt("%s/%s/%s",params.prefix,exp_name,"/Astar");
        std::tie(solution_stats,opt_scenario) = simulate_shortest_path<N>(zlb, &mesh_data, boundary, fWrapper, &params, datatype,
                                                                          lb::Copier<LoadBalancer>{},
                                                                          lb::Destroyer<LoadBalancer>{}, APP_COMM, simulation_name);
        load_balancing_cost = solution_stats.compute_avg_lb_time();
        configs.emplace_back("AstarReproduce\n", "AstarReproduce",  params, lb::Reproduce{opt_scenario});
    }

    experiment::load_configs(configs, params);

    for(auto& cfg : configs){
        auto& [preamble, config_name, params, criterion] = cfg;
        auto zlb = createLB(APP_COMM);
        auto[mesh_data, probe, lbtime, exp_name] = initExperiment(zlb, simbox, params, datatype, APP_COMM, getPositionPtrFunc, pointAssignFunc, doPartition, preamble);
        const std::string simulation_name = fmt("%s/%s/%s",params.prefix,exp_name,config_name);
        probe.push_load_balancing_time(load_balancing_cost);
        probe.push_load_balancing_parallel_efficiency(1.0);
        simulate<N>(zlb, &mesh_data, criterion, boundary, fWrapper, &params, &probe, datatype, APP_COMM, simulation_name);
        destroyLB(zlb);
    }

    MPI_Finalize();
    return 0;

}
