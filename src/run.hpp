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
#include <yalbb/yalbb.hpp>

#include "initial_conditions.hpp"
#include "loadbalancing.hpp"
#include "experience.hpp"

template<int N, class LoadBalancer, class Experiment, class BinaryForceFunc, class UnaryForceFunc, class LBCreatorFunc>
void run(const YALBB& yalbb, sim_param_t* params, Experiment experimentGenerator, Boundary<N> boundary, BinaryForceFunc binaryFunc, UnaryForceFunc unaryFF, LBCreatorFunc createLB) {
    int nproc;
    float ver;

    std::cout << std::fixed << std::setprecision(6);

    auto APP_COMM = yalbb.comm;

    sim_param_t burn_params = *params;

    burn_params.npart   = static_cast<int>(burn_params.npart * 0.1);
    burn_params.nframes = 1;
    burn_params.npframe = 5;
    burn_params.monitor = false;
    burn_params.record  = false;

    const std::array<Real, 2*N> simbox      = get_simbox<N>(params->simsize);
    const std::array<Real,   N> simlength   = get_box_width<N>(simbox);
    const std::array<Real,   N> box_center  = get_box_center<N>(simbox);
    const std::array<Real,   N> singularity = get_box_center<N>(simbox);

    MPI_Bcast(&params->seed, 1, MPI_INT, 0, MPI_COMM_WORLD);

    auto datatype           = elements::register_datatype<N>();
    auto getPositionPtrFunc = elements::getElementPositionPtr<N>;
    auto getVelocityPtrFunc = elements::getElementVelocityPtr<N>;

    // Who are my neighbors ?
    auto boxIntersectFunc    = lb::IntersectDomain<LoadBalancer> {params->rc};
    // Which domain this point belongs to?
    auto pointAssignFunc     = lb::AssignPoint<LoadBalancer> {};
    // Partition the domain without migration
    auto doPartition         = lb::DoPartition<LoadBalancer> {};
    // Load balance workload among CPUs
    auto doLoadBalancingFunc = lb::DoLB<LoadBalancer, decltype(getPositionPtrFunc)>(params->rc, datatype, APP_COMM, getPositionPtrFunc);
    // Destroy this LB struct
    auto destroyLB           = lb::Destroyer<LoadBalancer>{};
    // Get name of LB
    auto getLBName           = lb::NameGetter<LoadBalancer>{};

    // Wrap everything
    FunctionWrapper fWrapper(getPositionPtrFunc, getVelocityPtrFunc,
                                unaryFF, binaryFunc, boxIntersectFunc, pointAssignFunc, doLoadBalancingFunc);

    std::vector<experiment::Config> configs {};

    /** Burn CPU cycle */
    if(false) {
        MPI_Comm APP_COMM;
        MPI_Comm_dup(MPI_COMM_WORLD, &APP_COMM);
        auto zlb = createLB();
        auto[mesh_data, probe, exp_name] = experimentGenerator.init(zlb, getPositionPtrFunc, "Burn CPU Cycle:");
        simulate<N>(zlb, mesh_data.get(), lb::Static{}, boundary, fWrapper, &burn_params, &probe, datatype, APP_COMM, "BURN");
        destroyLB(zlb);
    }

    if(params->nb_best_path) {
        Probe solution_stats(nproc);
        std::vector<int> opt_scenario{};
        auto zlb = createLB();
        auto[mesh_data, probe, exp_name] = experimentGenerator.template init(zlb, getPositionPtrFunc, "A*\n");
        const std::string simulation_name = fmt("%s_%s_%i/%i/%i/%s/Astar", getLBName(), params->simulation_name, params->npart, params->seed, params->id, exp_name);
        std::tie(solution_stats,opt_scenario) = simulate_shortest_path<N>(zlb, mesh_data.get(), boundary, fWrapper, params, datatype,
                                                                          lb::Copier<LoadBalancer>{},
                                                                          lb::Destroyer<LoadBalancer>{}, APP_COMM, simulation_name);
        configs.emplace_back("AstarReproduce\n", "AstarReproduce",  *params, lb::Reproduce{opt_scenario});
    }

    experiment::load_configs(configs, *params);

    for(auto& cfg : configs) {

        auto& [preamble, config_name, params, criterion] = cfg;
        auto zlb = createLB();
        auto[mesh_data, probe, exp_name] = experimentGenerator.template init(zlb, getPositionPtrFunc, preamble);
        const std::string simulation_name = fmt("%s_%s_%i/%i/%i/%s/%s",getLBName(),params.simulation_name,params.npart,params.seed, params.id, exp_name,config_name);

        simulate<N>(zlb, mesh_data.get(), criterion, boundary, fWrapper, &params, &probe, datatype, APP_COMM, simulation_name);
        destroyLB(zlb);
    }
}