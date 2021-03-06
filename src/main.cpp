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

template<int N>
MESH_DATA<elements::Element<N>> generate_random_particles(int rank, sim_param_t params){
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

    const MESH_DATA<elements::Element<N>> particles = generate_random_particles<N>(rank, params);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////FINISHED PARITCLE INITIALIZATION///////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Domain-box intersection function *required*
    auto boxIntersectFunc   = [](auto* zlb, double x1, double y1, double z1, double x2, double y2, double z2, int* PEs, int* num_found){
        Zoltan_LB_Box_Assign(zlb, x1, y1, z1, x2, y2, z2, PEs, num_found);
    };
    // Point-in-domain callback *required*
    auto pointAssignFunc    = [](auto* zlb, const auto* e, int* PE) {
        auto pos_in_double = get_as_double_array<N>(e->position);
        Zoltan_LB_Point_Assign(zlb, &pos_in_double.front(), PE);
    };
    // Partitioning + migration function *required*
    auto doLoadBalancingFunc= [](auto* zlb, MESH_DATA<elements::Element<N>>* mesh_data){
        Zoltan_Do_LB<N>(mesh_data, zlb);
    };
    // Data getter function (position and velocity) *required*
    auto getPositionPtrFunc = [](auto* e) -> std::array<Real, N>* { return &e->position; };
    auto getVelocityPtrFunc = [](auto* e) -> std::array<Real, N>* { return &e->velocity; };

    // Short range force function computation
    auto getForceFunc = [eps=params.eps_lj, sig=params.sig_lj, rc=params.rc, getPositionPtrFunc](const auto* receiver, const auto* source){
        return lj_compute_force<N>(receiver, source, eps, sig*sig, rc, getPositionPtrFunc);
    };

    FunctionWrapper fWrapper(getPositionPtrFunc, getVelocityPtrFunc, getForceFunc, boxIntersectFunc, pointAssignFunc, doLoadBalancingFunc);
    auto datatype = elements::register_datatype<N>();
    double load_balancing_cost = 0;
    double load_balancing_parallel_efficiency = 0;

    /* Experience Without 
    {
        if(!rank) std::cout << "SIM (NoLB Criterion): Computation is starting" << std::endl;
        auto mesh_data = particles;
        Probe probe(nproc);
        PolicyExecutor menon_criterion_policy(&probe,[](Probe &probe) { return false; });
        simulate<N>(zlb, &mesh_data, &menon_criterion_policy, fWrapper, &params, &probe, datatype, APP_COMM, "nolb");
    }
    */
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

    /** Experience Menon **/
    {
        auto zlb = Zoltan_Copy(zz);
        if(!rank) std::cout << "SIM (Menon Criterion): Computation is starting" << std::endl;
        auto mesh_data = particles;
        Probe probe(nproc);
        probe.push_load_balancing_time(load_balancing_cost);
        PolicyExecutor menon_criterion_policy(&probe,[nframes=params.nframes, npframe = params.npframe](Probe &probe) {
            bool is_new_batch = (probe.get_current_iteration() % npframe == 0);
            return is_new_batch && (probe.get_cumulative_imbalance_time() >= probe.compute_avg_lb_time());
        });
        simulate<N>(zlb, &mesh_data, &menon_criterion_policy, fWrapper, &params, &probe, datatype, APP_COMM, "menon");
    }

    /** Experience Procassini **/
    {
        auto zlb = Zoltan_Copy(zz);
        auto mesh_data = particles;
        Probe probe(nproc);
        probe.push_load_balancing_time(load_balancing_cost);
        probe.push_load_balancing_parallel_efficiency(load_balancing_parallel_efficiency);

        if(!rank) {
            std::cout << "SIM (Procassini Criterion): Computation is starting." << std::endl;
            std::cout << "Average C = " << probe.compute_avg_lb_time() << std::endl;
        }

        PolicyExecutor procassini_criterion_policy(&probe,
        [npframe = params.npframe](Probe probe) {
                bool is_new_batch = (probe.get_current_iteration() % npframe == 0);
                Real epsilon_c = probe.get_efficiency();
                Real epsilon_lb= probe.compute_avg_lb_parallel_efficiency(); //estimation based on previous lb call
                Real S         = epsilon_c / epsilon_lb;
                Real tau_prime = probe.get_batch_time() *  S + probe.compute_avg_lb_time(); //estimation of next iteration time based on speed up + LB cost
                Real tau       = probe.get_batch_time();
                return is_new_batch && (tau_prime < tau);
            });

        simulate<N>(zlb, &mesh_data, &procassini_criterion_policy, fWrapper, &params, &probe, datatype, APP_COMM, "procassini");
    }

    /** Experience Marquez **/
    {
        auto zlb = Zoltan_Copy(zz);
        auto mesh_data = particles;

        if(!rank) {
            std::cout << "SIM (Marquez Criterion): Computation is starting." << std::endl;
        }
        Probe probe(nproc);
        PolicyExecutor marquez_criterion_policy(&probe,
            [rank, threshold = 0.05, npframe = params.npframe](Probe probe){
                bool is_new_batch = (probe.get_current_iteration() % npframe == 0);
                Real tolerance      = probe.get_avg_it() * threshold;
                Real tolerance_plus = probe.get_avg_it() + tolerance;
                Real tolerance_minus= probe.get_avg_it() - tolerance;
                return is_new_batch && (probe.get_min_it() < tolerance_minus || tolerance_plus < probe.get_max_it());
            });

        simulate<N>(zlb, &mesh_data, &marquez_criterion_policy, fWrapper, &params, &probe, datatype, APP_COMM, "marquez");
    }

    MPI_Finalize();
    return 0;

}
