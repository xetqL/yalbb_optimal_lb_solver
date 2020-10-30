//
// Created by xetql on 10/9/20.
//

#ifndef YALBB_EXAMPLE_EXPERIENCE_HPP
#define YALBB_EXAMPLE_EXPERIENCE_HPP
#include <yalbb/probe.hpp>
#include <yalbb/params.hpp>

#include <tuple>
#include <any>

#include "spatial_elements.hpp"
#include "StripeLB.hpp"
#include "zoltan_fn.hpp"
#include "initial_conditions.hpp"
template<int N>
using ExperimentRet = std::tuple<MESH_DATA<elements::Element<N>>, Probe, double, std::string>;

template<int N, class BalancerType, class DoPartition, class GetPosFunc, class GetPointFunc>
using experiment_t = ExperimentRet<N> (*) (BalancerType* zlb,
                                                         BoundingBox<N> simbox,
                                                         sim_param_t params,
                                                         MPI_Datatype datatype,
                                                         MPI_Comm APP_COMM,
                                                         GetPosFunc getPos,
                                                         GetPointFunc pointAssign,
                                                         DoPartition doPartition,
                                                         std::string preamble);

std::pair<int, int> pre_init_experiment(const std::string& preamble, MPI_Comm APP_COMM){
    int rank, nproc;
    MPI_Comm_rank(APP_COMM, &rank);
    MPI_Comm_size(APP_COMM, &nproc);
    par::pcout() << preamble << std::endl;
    return {rank, nproc};
}

template<int N>
ExperimentRet<N> post_init_experiment(const MESH_DATA<elements::Element<N>>& mesh_data, double lbtime, const std::string& func, int nproc, MPI_Comm APP_COMM) {
    MPI_Allreduce(MPI_IN_PLACE, &lbtime, 1, MPI_TIME, MPI_MAX, APP_COMM);
    par::pcout() << func << std::endl;
    Probe probe(nproc);
    return std::make_tuple(mesh_data, probe, lbtime, func);
}

template<int N, class BalancerType, class DoPartition, class GetPosFunc, class GetPointFunc>
ExperimentRet<N> init_exp_uniform_cube(
        BalancerType* zlb,
        BoundingBox<N> simbox,
        sim_param_t params,
        MPI_Datatype datatype,
        MPI_Comm APP_COMM,
        GetPosFunc getPos,
        GetPointFunc pointAssign,
        DoPartition doPartition,
        std::string preamble = "") {
    auto[rank, nproc] = pre_init_experiment(preamble, APP_COMM);

    // EXPERIMENTATION PART
    //auto zlb = new StripeLB<elements::Element<N>, N, N-1> (APP_COMM);
    auto mesh_data = generate_random_particles<N>(rank, params,
                                                  pos::UniformInCube<N>(simbox),
                                                  //vel::ParallelToAxis<N, N-1>( 2.0*params.T0*params.T0 ));
                                                  vel::GoToStripe<N, N-1>{params.T0 * params.T0, params.simsize - (params.simsize / (float) nproc)});
    PAR_START_TIMER(lbtime, APP_COMM);
    doPartition(zlb, &mesh_data, getPos);
    migrate_data(zlb, mesh_data.els, pointAssign, datatype, APP_COMM);
    END_TIMER(lbtime);
    // END

    return post_init_experiment<N>(mesh_data, lbtime, std::string(__FUNCTION__), nproc, APP_COMM);
}

template<int N, class BalancerType, class DoPartition, class GetPosFunc, class GetPointFunc>
ExperimentRet<N> init_exp_contracting_sphere_zoltan(
        BalancerType* zlb,
        BoundingBox<N> simbox,
        sim_param_t params,
        MPI_Datatype datatype,
        MPI_Comm APP_COMM,
        GetPosFunc getPos,
        GetPointFunc pointAssign,
        DoPartition doPartition,
        std::string preamble = "")
{
    auto[rank, nproc] = pre_init_experiment(preamble, APP_COMM);

    std::array<Real, N> box_center{};
    std::fill(box_center.begin(), box_center.end(), params.simsize / 2.0);
    auto mesh_data = generate_random_particles<N>(rank, params,
                                                  pos::UniformInSphere<N>(params.simsize / 2.0, box_center),
                                                  vel::ContractToPoint<N>(params.T0, box_center));
    PAR_START_TIMER(lbtime, APP_COMM);
    doPartition(zlb, &mesh_data, getPos);
    migrate_data(zlb, mesh_data.els, pointAssign, datatype, APP_COMM);
    END_TIMER(lbtime);

    return post_init_experiment<N>(mesh_data, lbtime, std::string(__FUNCTION__), nproc, APP_COMM);
}

template<int N, class BalancerType, class DoPartition, class GetPosFunc, class GetPointFunc>
ExperimentRet<N> init_exp_expanding_sphere_zoltan(
        BalancerType* zlb,
        BoundingBox<N> simbox,
        sim_param_t params,
        MPI_Datatype datatype,
        MPI_Comm APP_COMM,
        GetPosFunc getPos,
        GetPointFunc pointAssign,
        DoPartition doPartition,
        std::string preamble = "")
{
    auto[rank, nproc] = pre_init_experiment(preamble, APP_COMM);

    //auto zlb = zoltan_create_wrapper(APP_COMM);
    std::array<Real, N> box_center{};
    std::fill(box_center.begin(), box_center.end(), params.simsize / 2.0);
    auto mesh_data = generate_random_particles<N>(rank, params,
                                                  pos::UniformInSphere<N>(params.simsize / 20.0, box_center),
                                                  vel::ExpandFromPoint<N>(params.T0, box_center));
    PAR_START_TIMER(lbtime, APP_COMM);
    doPartition(zlb, &mesh_data, getPos);
    migrate_data(zlb, mesh_data.els, pointAssign, datatype, APP_COMM);
    END_TIMER(lbtime);

    return post_init_experiment<N>(mesh_data, lbtime, std::string(__FUNCTION__), nproc, APP_COMM);
}

#endif //YALBB_EXAMPLE_EXPERIENCE_HPP
