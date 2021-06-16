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
void run(const YALBB& yalbb, sim_param_t* params, Experiment experimentGenerator, Boundary<N> boundary, std::string lb_name,
         BinaryForceFunc binaryFunc, UnaryForceFunc unaryFF, LBCreatorFunc createLB) {
    std::cout << std::fixed << std::setprecision(6);

    auto APP_COMM = yalbb.comm;
    auto nproc = yalbb.comm_size;

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

    // Wrap everything
    FunctionWrapper fWrapper(getPositionPtrFunc, getVelocityPtrFunc,
                             unaryFF, binaryFunc, boxIntersectFunc, pointAssignFunc, doLoadBalancingFunc);

    std::vector<experiment::Config> configs {};

    std::string directory = fmt("%s_%s_%i/%i/%i/%i/", lb_name, params->simulation_name, params->npart, params->seed, nproc, params->id);
    const auto exp_name = experimentGenerator.get_exp_name();

    /** Burn CPU cycle */
     if(params->burn) {
         sim_param_t burn_params = *params;

         const std::string simulation_name = fmt("%s/%s/%s", directory, exp_name, "burn_cpu");
         std::string folder_prefix = fmt("%s/%s", "logs", simulation_name);

         simulation::MonitoringSession report_session {!yalbb.my_rank, burn_params.record, folder_prefix, "", burn_params.monitor};

         burn_params.npart   = static_cast<int>(burn_params.npart);
         burn_params.nframes = 5;
         burn_params.npframe = 5;
         burn_params.monitor = false;
         burn_params.record  = false;

         MPI_Comm APP_COMM;
         MPI_Comm_dup(MPI_COMM_WORLD, &APP_COMM);
         auto zlb = createLB();
         auto[mesh_data, probe] = experimentGenerator.init(zlb, getPositionPtrFunc, "Burn CPU Cycle:", report_session);
         simulate<N>(zlb, mesh_data.get(), lb::Static{}, boundary, fWrapper, &burn_params, &probe, datatype, report_session, APP_COMM, simulation_name);
         destroyLB(zlb);
    }

    if(params->nb_best_path) {
        const std::string simulation_name = fmt("%s/%s/Astar", directory, exp_name);
        std::string folder_prefix = fmt("%s/%s", "logs", simulation_name);
        simulation::MonitoringSession report_session {!yalbb.my_rank, params->record, folder_prefix, "", params->monitor};

        Probe solution_stats(nproc);
        std::vector<int> opt_scenario{};
        auto zlb = createLB();
        auto[mesh_data, probe] = experimentGenerator.template init(zlb, getPositionPtrFunc, "A*\n", report_session);
        std::tie(solution_stats,opt_scenario) = simulate_shortest_path<N>(zlb, mesh_data.get(), boundary, fWrapper, params, datatype,
                                                                          lb::Copier<LoadBalancer>{},
                                                                          lb::Destroyer<LoadBalancer>{}, APP_COMM, simulation_name);
        configs.emplace_back("AstarReproduce\n", "AstarReproduce",  *params, lb::Reproduce{opt_scenario});
    }

    experiment::load_configs(configs, *params);

    for(auto& cfg : configs) {
        auto& [preamble, config_name, params, criterion] = cfg;
        const std::string simulation_name = fmt("%s/%s/%s", directory, exp_name, config_name);
        std::string folder_prefix = fmt("%s/%s", "logs", simulation_name);
        simulation::MonitoringSession report_session {!yalbb.my_rank, params.record, folder_prefix, "", params.monitor};

        auto zlb = createLB();

        auto[mesh_data, probe]  = experimentGenerator.init(zlb, getPositionPtrFunc, preamble, report_session);

        simulate<N>(zlb, mesh_data.get(), criterion, boundary, fWrapper, &params, &probe, datatype, report_session, APP_COMM, simulation_name);

        destroyLB(zlb);
    }
}