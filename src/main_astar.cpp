#include <string>
#include <mpi.h>
#include <random>

#include <yalbb/simulator.hpp>
#include <yalbb/shortest_path.hpp>
#include <yalbb/node.hpp>

#include <yalbb/probe.hpp>
#include <yalbb/ljpotential.hpp>
#include <iomanip>

#include "initial_conditions.hpp"
#include "zoltan_fn.hpp"
#include "spatial_elements.hpp"
#include "utils.hpp"
template<int N>
MESH_DATA<elements::Element<N>> generate_random_particles(int rank, sim_param_t params){
    MESH_DATA<elements::Element<N>> mesh;

    if (!rank)
    {
        std::cout << "Generating data ..." << std::endl;
        std::shared_ptr<initial_condition::lj::RejectionCondition<N>> condition;
        std::mt19937 my_gen(params.seed);
        statistic::UniformSphericalDistribution<N, Real> sphere(params.simsize / 3.0, params.simsize / 2.0, params.simsize / 2.0, 2*params.simsize / 3.0);
        std::uniform_real_distribution<Real> udist(0, 2*params.T0*params.T0);
        for(int i = 0;i < params.npart; ++i) {
            mesh.els.emplace_back(sphere(my_gen), (std::array<float,3>) {udist(my_gen),udist(my_gen),udist(my_gen)}, i, i);
        }
        std::cout << mesh.els.size() << " Done !" << std::endl;
    }

    return mesh;
}

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
        statistic::UniformSphericalDistribution<N, Real> sphere(params.simsize / 3.0, params.simsize / 2.0, params.simsize / 2.0, 2*params.simsize / 3.0);
        std::uniform_real_distribution<Real> udist(0, 2*params.T0*params.T0);
        initial_condition::lj::RandomElementsGen<N>(params.seed, MAX_TRIAL, condition)
                .generate_elements(mesh.els, params.npart,
                        [&sphere](auto& my_gen){ return sphere(my_gen); },
                        [&udist] (auto& my_gen)->std::array<Real, N>{ return {udist(my_gen),udist(my_gen),udist(my_gen)};});
        std::cout << mesh.els.size() << " Done !" << std::endl;
    }

    return mesh;
}

int main(int argc, char** argv) {
    constexpr int N = 3;
    int rank, nproc;
    float ver;

    std::cout << std::fixed << std::setprecision(6);

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    MPI_Comm APP_COMM;
    MPI_Comm_dup(MPI_COMM_WORLD, &APP_COMM);

    auto option = get_params(argc, argv);
    if (!option.has_value()) {
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    auto params = option.value();

    params.rc = 2.5f * params.sig_lj;
    params.simsize = std::ceil(params.simsize / params.rc) * params.rc;

    MPI_Bcast(&params.seed, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        print_params(params);
        std::cout << "Computating with " << nproc << " PEs"<<std::endl;
    }

    if(Zoltan_Initialize(argc, argv, &ver) != ZOLTAN_OK) {
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    auto zz = zoltan_create_wrapper(APP_COMM);

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
    auto boxIntersectFunc   = [](auto* zlb, double x1, double y1, double z1, double x2, double y2, double z2, int* PEs, int* num_found){
        Zoltan_LB_Box_Assign(zlb, x1, y1, z1, x2, y2, z2, PEs, num_found);
    };

    // Point-in-domain callback *required*
    // Solve belongings
    auto pointAssignFunc    = [](auto* zlb, const auto* e, int* PE) {
        auto pos_in_double = get_as_double_array<N>(e->position);
        Zoltan_LB_Point_Assign(zlb, &pos_in_double.front(), PE);
    };

    // Partitioning + migration function *required*
    // Solve partitioning
    auto doLoadBalancingFunc = [params, getPositionPtrFunc, boxIntersectFunc, datatype, APP_COMM](auto* zlb, MESH_DATA<elements::Element<N>>* mesh_data) {
        // Get mesh from LB struct
        // ...

        // Update mesh weights using particles
        // ...

        MESH_DATA<elements::Element<N>> interactions;
        interactions.els.reserve(mesh_data->els.size()*1.5);
        auto bbox      = get_bounding_box<N>(params.rc, getPositionPtrFunc, mesh_data->els);
        auto remote_el = retrieve_ghosts<N>(zlb, mesh_data->els, bbox, boxIntersectFunc, params.rc, datatype, APP_COMM);
        std::vector<Index> lscl, head;
        const auto nlocal  = mesh_data->els.size();
        apply_resize_strategy(&lscl,   nlocal + remote_el.size() );

        try{
            CLL_init<N, elements::Element<N>>({{mesh_data->els.data(), nlocal}, {remote_el.data(), remote_el.size()}}, getPositionPtrFunc, bbox, params.rc, &head, &lscl);
        } catch (const std::out_of_range& oor) {
            std::cout << "Out of Range error in: " << __FILE__ << ":" << __PRETTY_FUNCTION__ << oor.what() << std::endl;
            abort();
        }

        CLL_foreach_interaction(mesh_data->els.data(), nlocal, remote_el.data(), getPositionPtrFunc, bbox, params.rc, &head, &lscl,
            [&interactions](const auto *r, const auto *s) {
            interactions.els.push_back(midpoint<N>(*r, *s));
        });

        Zoltan_Do_LB<N>(&interactions, zlb);
    };

    // Short range force function computation
    auto getForceFunc = [eps=params.eps_lj, sig=params.sig_lj, rc=params.rc, getPositionPtrFunc](const auto* receiver, const auto* source){
        return lj_compute_force<N>(receiver, source, eps, sig*sig, rc, getPositionPtrFunc);
    };

    FunctionWrapper fWrapper(getPositionPtrFunc, getVelocityPtrFunc, getForceFunc, boxIntersectFunc, pointAssignFunc, doLoadBalancingFunc);

    MESH_DATA<elements::Element<N>> particles = generate_random_particles<N>(rank, params);
    Zoltan_Do_LB<N>(&particles, zz);
    migrate_data(zz, particles.els, pointAssignFunc, datatype, APP_COMM);

    double load_balancing_cost = 0;
    double load_balancing_parallel_efficiency = 0;

    std::vector<int> opt_scenario;
    Probe solution_stats(nproc);
    if(params.nb_best_path) {
        auto zlb = Zoltan_Copy(zz);
	    auto mesh_data = particles;
        if(!rank) std::cout << "SIM (A* optimized): Computation is starting" << std::endl;
        std::tie(solution_stats,opt_scenario) = simulate_shortest_path<N>(zlb, &mesh_data,  fWrapper, &params, datatype, [](Zoltan_Struct* lb){ return Zoltan_Copy(lb);}, [](Zoltan_Struct* lb){ Zoltan_Destroy(&lb);}, APP_COMM, "astar");
        load_balancing_cost = solution_stats.compute_avg_lb_time();
        load_balancing_parallel_efficiency = solution_stats.compute_avg_lb_parallel_efficiency();
        /** Experience Reproduce ASTAR **/
        {
            auto zlb = Zoltan_Copy(zz);
            if(!rank) std::cout << "SIM (ASTAR Criterion): Computation is starting" << std::endl;
            auto mesh_data = particles;
            Probe probe(nproc);
            probe.push_load_balancing_time(load_balancing_cost);
            PolicyExecutor menon_criterion_policy(&probe,[opt_scenario](Probe &probe) {
                return (bool) opt_scenario.at(probe.get_current_iteration());
            });
            simulate<N>(zlb, &mesh_data, &menon_criterion_policy, fWrapper, &params, &probe, datatype, APP_COMM, "astar_mimic");
        }
    }

    MPI_Finalize();
    return 0;

}