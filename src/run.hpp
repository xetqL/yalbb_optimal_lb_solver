//
// Created by xetql on 11/27/20.
//

#pragma once
#include <string>
#include <mpi.h>
#include <random>
#include <iomanip>

#include <yalbb/simulator.hpp>
#include <yalbb/shortest_path.hpp>
#include <yalbb/policy.hpp>
#include <yalbb/probe.hpp>
#include <yalbb/ljpotential.hpp>
#include <yalbb/functionwrapper.hpp>

#include "initial_conditions.hpp"
#include "loadbalancing.hpp"
#include "experience.hpp"

template<int N, class LoadBalancer, class Experiment, class UnaryForceFunc>
void run(int argc, char** argv, Experiment experimentGenerator, Boundary<N> boundary, UnaryForceFunc unaryFF) {
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

    burn_params.npart   = (int) (params.npart * 0.1);
    burn_params.nframes = 40;
    burn_params.npframe = 5;
    burn_params.monitor = false;
    burn_params.record  = false;

    const std::array<Real, 2*N> simbox      = get_simbox<N>(params.simsize);
    const std::array<Real,   N> simlength   = get_box_width<N>(simbox);
    const std::array<Real,   N> box_center  = get_box_center<N>(simbox);
    const std::array<Real,   N> singularity = get_box_center<N>(simbox);

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
    // Wrap everything
    FunctionWrapper fWrapper(getPositionPtrFunc, getVelocityPtrFunc, unaryFF, getForceFunc, boxIntersectFunc, pointAssignFunc, doLoadBalancingFunc);

    double load_balancing_cost = 0;
    double load_balancing_parallel_efficiency = 0;
    std::vector<experiment::Config> configs {};

    /** Burn CPU cycle */
    {
        MPI_Comm APP_COMM;
        MPI_Comm_dup(MPI_COMM_WORLD, &APP_COMM);
        auto zlb = createLB(APP_COMM);
        auto[mesh_data, probe, lbtime, exp_name] = experimentGenerator.template init<N>(zlb, simbox, burn_params, datatype, APP_COMM, getPositionPtrFunc, pointAssignFunc, doPartition, "Burn CPU Cycle:");
        simulate<N>(zlb, mesh_data, lb::Static{}, boundary, fWrapper, &burn_params, &probe, datatype, APP_COMM, "BURN");
        destroyLB(zlb);
        delete mesh_data;
    }

    if(params.nb_best_path) {
        Probe solution_stats(nproc);
        std::vector<int> opt_scenario{};
        auto zlb = createLB(APP_COMM);
        auto[mesh_data, probe, lbtime, exp_name] = experimentGenerator.template init<N>(zlb, simbox, params, datatype, APP_COMM, getPositionPtrFunc, pointAssignFunc, doPartition, "A*\n");
        const std::string simulation_name = fmt("%s%i/%i/%i/%s/Astar",params.simulation_name,params.npart,params.seed, params.id, exp_name);
        std::tie(solution_stats,opt_scenario) = simulate_shortest_path<N>(zlb, mesh_data, boundary, fWrapper, &params, datatype,
                                                                          lb::Copier<LoadBalancer>{},
                                                                          lb::Destroyer<LoadBalancer>{}, APP_COMM, simulation_name);
        load_balancing_cost = solution_stats.compute_avg_lb_time();
        configs.emplace_back("AstarReproduce\n", "AstarReproduce",  params, lb::Reproduce{opt_scenario});
        delete mesh_data;

    }

    experiment::load_configs(configs, params);

    for(auto& cfg : configs){
        auto& [preamble, config_name, params, criterion] = cfg;
        auto zlb = createLB(APP_COMM);
        auto[mesh_data, probe, lbtime, exp_name] = experimentGenerator.template init<N>(zlb, simbox, params, datatype, APP_COMM, getPositionPtrFunc, pointAssignFunc, doPartition, preamble);
        const std::string simulation_name = fmt("%s%i/%i/%i/%s/%s",params.simulation_name,params.npart,params.seed, params.id, exp_name,config_name);
        probe.push_load_balancing_time(load_balancing_cost);
        probe.push_load_balancing_parallel_efficiency(1.0);
        simulate<N>(zlb, mesh_data, criterion, boundary, fWrapper, &params, &probe, datatype, APP_COMM, simulation_name);
        destroyLB(zlb);
        delete mesh_data;
    }

    MPI_Finalize();
}