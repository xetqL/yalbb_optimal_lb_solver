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
#include "spatial_elements.hpp"
#include "utils.hpp"
#include "StripeLB.hpp"
#include "experience.hpp"

int main(int argc, char** argv) {
    constexpr int N = 3;
    int _rank, _nproc;

    std::cout << std::fixed << std::setprecision(6);

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &_nproc);
    const int rank = _rank,nproc = _nproc;
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

    std::array<Real, 2*N> simbox      = {0, params.simsize, 0,params.simsize, 0,params.simsize};
    std::array<Real, N>   simlength   = {params.simsize, params.simsize, params.simsize};
    std::array<Real, N>   box_center  = {params.simsize / (Real) 2.0,params.simsize / (Real)2.0,params.simsize / (Real)2.0};
    std::array<Real, N>   singularity = {params.simsize / (Real) 2.0,params.simsize / (Real)2.0,params.simsize / (Real)2.0};

    MPI_Bcast(&params.seed, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        print_params(params);
        std::cout << "Computating with " << nproc << " PEs"<<std::endl;
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

    FunctionWrapper fWrapper(getPositionPtrFunc, getVelocityPtrFunc, getForceFunc, boxIntersectFunc, pointAssignFunc, doLoadBalancingFunc);

    double load_balancing_cost = 0;
    double load_balancing_parallel_efficiency = 0;

    std::vector<int> opt_scenario;
    Probe solution_stats(nproc);
    if(params.nb_best_path) {
        auto[zlb, mesh_data, probe, lbtime] = init_exp_uniform_cube_fixed_stripe_LB< N>(simbox, params, datatype, APP_COMM, getPositionPtrFunc, pointAssignFunc, "(A* optimized): ");
        std::tie(solution_stats,opt_scenario) =
                simulate_shortest_path<N>(zlb, &mesh_data,  fWrapper, &params, datatype,
                    [](auto* lb){
                        auto* ptr = new StripeLB<elements::Element<N>,N,2>(lb->comm);
                        std::copy(lb->stripes.begin(), lb->stripes.end(), ptr->stripes.begin());
                        return ptr;},
                    [](auto* lb){ destroy(lb);}, APP_COMM, "astar");

        load_balancing_cost = solution_stats.compute_avg_lb_time();
        load_balancing_parallel_efficiency = solution_stats.compute_avg_lb_parallel_efficiency();

        /** Experience Reproduce ASTAR **/
        destroy(zlb);

        {
            auto[zlb, mesh_data, probe, lbtime] = init_exp_uniform_cube_fixed_stripe_LB< N>(simbox, params, datatype, APP_COMM, getPositionPtrFunc, pointAssignFunc,  "(A* mimic): ");
            probe.push_load_balancing_time(load_balancing_cost);
            PolicyExecutor menon_criterion_policy(&probe,[opt_scenario](Probe &probe) {
                return (bool) opt_scenario.at(probe.get_current_iteration());
            });
            simulate<N>(zlb, &mesh_data, &menon_criterion_policy, fWrapper, &params, &probe, datatype, APP_COMM, "astar_mimic");
            destroy(zlb);
        }
    }

    {//burn cpu cycle

        auto burn_params = option.value();
	    burn_params.npart  = params.npart * 0.1f;
	    burn_params.nframes= 1;
	    burn_params.npframe= 1000;

        auto[zlb, mesh_data, probe, lbtime] = init_exp_uniform_cube_fixed_stripe_LB< N>(simbox, burn_params, datatype, APP_COMM, getPositionPtrFunc, pointAssignFunc,  "Burn CPU cycles: ");
	    PolicyExecutor menon_criterion_policy(&probe, [](Probe &probe) { return false; });
        simulate<N>(zlb, &mesh_data, &menon_criterion_policy, fWrapper, &burn_params, &probe, datatype, APP_COMM, "burn");
        destroy(zlb);
    }


    /** Experience Menon **/
    {

        auto[zlb, mesh_data, probe, lbtime] = init_exp_uniform_cube_fixed_stripe_LB< N>(simbox, params, datatype, APP_COMM, getPositionPtrFunc, pointAssignFunc,  "Menon (baseline): ");
        probe.push_load_balancing_time(lbtime / 2.0);
        PolicyExecutor menon_criterion_policy(&probe, [](Probe &probe) {
            return (probe.get_cumulative_imbalance_time() >= probe.compute_avg_lb_time());
        });
        simulate<N>(zlb, &mesh_data, &menon_criterion_policy, fWrapper, &params, &probe, datatype, APP_COMM, "menon");
        destroy(zlb);
    }

    /** Experience Procassini **/
    {
        auto[zlb, mesh_data, probe, lbtime] = init_exp_uniform_cube_fixed_stripe_LB< N>(simbox, params, datatype, APP_COMM, getPositionPtrFunc, pointAssignFunc,  "Procassini: ");

        probe.push_load_balancing_time(lbtime);
        probe.push_load_balancing_parallel_efficiency(load_balancing_parallel_efficiency);

        PolicyExecutor procassini_criterion_policy(&probe, [](Probe& probe) {
                Real epsilon_c = probe.get_efficiency();
                Real epsilon_lb= probe.compute_avg_lb_parallel_efficiency(); //estimation based on previous lb call
                Real S         = epsilon_c / epsilon_lb;
                Real tau_prime = probe.get_batch_time() *  S + probe.compute_avg_lb_time(); //estimation of next iteration time based on speed up + LB cost
                Real tau       = probe.get_batch_time();
                return (tau_prime < tau);
        });

        simulate<N>(zlb, &mesh_data, &procassini_criterion_policy, fWrapper, &params, &probe, datatype, APP_COMM, "procassini");
        destroy(zlb);
    }
    MPI_Finalize();
    return 0;

}
