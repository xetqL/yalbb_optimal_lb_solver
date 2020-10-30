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
#include "zoltan_fn.hpp"
#include "spatial_elements.hpp"
#include "utils.hpp"
#include "experience.hpp"
#include "../../yalbb/includes/yalbb/policy.hpp"

template<int N>
MESH_DATA<elements::Element<N>> generate_random_particles_with_rejection(int rank, sim_param_t params) {
    MESH_DATA<elements::Element<N>> mesh;

    if (!rank) {
        std::cout << "Generating data ..." << std::endl;
        std::shared_ptr<initial_condition::lj::RejectionCondition<N>> condition;
        const int MAX_TRIAL = 1000000;
        condition = std::make_shared<initial_condition::lj::RejectionCondition<N>>(
                &(mesh.els), params.sig_lj, (params.sig_lj * params.sig_lj), params.T0, 0, 0, 0,
                params.simsize, params.simsize, params.simsize, &params
        );
        statistic::UniformSphericalDistribution<N, Real> sphere(params.simsize / 3.0, params.simsize / 2.0, params.simsize / 2.0, 2.0 * params.simsize / 3.0);
        std::uniform_real_distribution<Real> udist(0, 2.0*params.T0*params.T0);
        initial_condition::lj::RandomElementsGen<N>(params.seed, MAX_TRIAL, condition)
                .generate_elements(mesh.els, params.npart,
                        [&sphere](auto& my_gen) -> std::array<Real, N> { return sphere(my_gen); },
                        [&udist] (auto& my_gen) -> std::array<Real, N> { return {udist(my_gen), udist(my_gen), udist(my_gen)};});
        std::cout << mesh.els.size() << " Done !" << std::endl;
    }

    return mesh;
}
constexpr unsigned NumConfig = 5;
using Config = std::tuple<std::string, std::string, sim_param_t, lb::Criterion>;
#define __STR_FUNCNAME__(x) std::string(#x)

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
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////FINISHED PARITCLE INITIALIZATION///////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Domain-box intersection function *required*
    // Solve interactions
    auto boxIntersectFunc   = [rank, rc=params.rc](auto* zlb, double x1, double y1, double z1, double x2, double y2, double z2, int* PEs, int* num_found){
        auto neighbors = zlb->get_neighbors(rank, rc);
        std::copy(neighbors.begin(), neighbors.end(), PEs);
        *num_found = neighbors.size();
    };

    // Point-in-domain callback *required*
    // Solve belongings
    auto pointAssignFunc    = [](auto* zlb, const auto* e, int* PE) {
        zlb->lookup_domain(e->position, PE);
    };

    // Partitioning + migration function *required*
    // Solve partitioning
    auto doLoadBalancingFunc = [rc = params.rc, getPositionPtrFunc, boxIntersectFunc, datatype, APP_COMM](auto* zlb, MESH_DATA<elements::Element<N>>* mesh_data) {
        // Get mesh from LB struct
        // ...

        // Update mesh weights using particles
        // ...
        std::vector<elements::Element<N>> sampled;

        for(int i = 0; i < 5; ++i)
            std::copy(mesh_data->els.begin(), mesh_data->els.end(), std::back_inserter(sampled));

        MESH_DATA<elements::Element<N>> interactions;

        auto bbox      = get_bounding_box<N>(rc, getPositionPtrFunc, sampled);
        auto remote_el = retrieve_ghosts<N>(zlb, sampled, bbox, boxIntersectFunc, rc, datatype, APP_COMM);
        std::vector<Index> lscl, head;
        const auto nlocal  = sampled.size();
        apply_resize_strategy(&lscl,   nlocal + remote_el.size() );

        CLL_init<N, elements::Element<N>>({{sampled.data(), nlocal}, {remote_el.data(), remote_el.size()}}, getPositionPtrFunc, bbox, rc, &head, &lscl);

        CLL_foreach_interaction(sampled.data(), nlocal, remote_el.data(), getPositionPtrFunc, bbox, rc, &head, &lscl,
                                [&interactions](const auto *r, const auto *s) {
                                    interactions.els.push_back(midpoint<N>(*r, *s));
                                });
        zlb->partition(mesh_data->els, getPositionPtrFunc);
    };

    // Short range force function computation
    auto getForceFunc = [eps=params.eps_lj, sig=params.sig_lj, rc=params.rc, getPositionPtrFunc](const auto* receiver, const auto* source)->std::array<Real, N>{
        return lj_compute_force<N>(receiver, source, eps, sig*sig, rc, getPositionPtrFunc);
    };

    auto doPartition = [](auto* zlb, auto* mesh_data, auto getPos){
        zlb->partition(mesh_data->els, getPos);
    };

    using Particle     = elements::Element<N>;
    using LoadBalancer = StripeLB<Particle, N, N-1>;
    using Experiment   = experiment::experiment_t<N, LoadBalancer, decltype(doPartition), decltype(getPositionPtrFunc), decltype(pointAssignFunc)>;

    Experiment initExperiment = experiment::UniformCube;

    FunctionWrapper fWrapper(getPositionPtrFunc, getVelocityPtrFunc, getForceFunc, boxIntersectFunc, pointAssignFunc, doLoadBalancingFunc);

    double load_balancing_cost = 0;
    double load_balancing_parallel_efficiency = 0;

    std::vector<Config> configs;

    /** Burn CPU cycle */
    {
        MPI_Comm APP_COMM;
        MPI_Comm_dup(MPI_COMM_WORLD, &APP_COMM);

        auto zlb = new LoadBalancer(APP_COMM);
        auto[mesh_data, probe, lbtime, exp_name] = initExperiment(zlb, simbox, burn_params, datatype, APP_COMM, getPositionPtrFunc, pointAssignFunc, doPartition, "Burn CPU Cycle:");
        simulate<N>(zlb, &mesh_data, lb::Static {}, fWrapper, &burn_params, &probe, datatype, APP_COMM, "BURN");
    }

    std::vector<int> opt_scenario;
    Probe solution_stats(nproc);
    if(params.nb_best_path) {
        auto zlb = new LoadBalancer(APP_COMM);
        auto[mesh_data, probe, lbtime, exp_name] = initExperiment(zlb, simbox, params, datatype, APP_COMM, getPositionPtrFunc, pointAssignFunc, doPartition, "A*\n");
        std::tie(solution_stats,opt_scenario) = simulate_shortest_path<N>(zlb, &mesh_data,  fWrapper, &params, datatype,
                      [](auto* lb){ return allocate_from<Particle, N, N-1>(*lb);},
                      [](auto* lb){ destroy(lb);}, APP_COMM, "Astar");
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
        auto zlb = new LoadBalancer(APP_COMM);
        auto[mesh_data, probe, lbtime, exp_name] = initExperiment(zlb, simbox, params, datatype, APP_COMM, getPositionPtrFunc, pointAssignFunc, doPartition, preamble);
        const auto simulation_name = params.prefix.append("/").append(exp_name).append("/").append(config_name);
        probe.push_load_balancing_time(load_balancing_cost);
        probe.push_load_balancing_parallel_efficiency(1.0);
        simulate<N>(zlb, &mesh_data, criterion, fWrapper, &params, &probe, datatype, APP_COMM, simulation_name);
        destroy(zlb);
    }

    MPI_Finalize();
    return 0;

}
