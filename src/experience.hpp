//
// Created by xetql on 10/9/20.
//

#ifndef YALBB_EXAMPLE_EXPERIENCE_HPP
#define YALBB_EXAMPLE_EXPERIENCE_HPP
#include <yalbb/probe.hpp>
#include <yalbb/params.hpp>

#include <tuple>

#include "spatial_elements.hpp"
#include "StripeLB.hpp"
#include "zoltan_fn.hpp"
#include "initial_conditions.hpp"
template<int N, class GetPosFunc, class GetPointFunc>
std::tuple<StripeLB<elements::Element<N>, N, N-1>*, MESH_DATA<elements::Element<N>>, Probe, double>
init_exp_uniform_cube_fixed_stripe_LB(
        BoundingBox<N> simbox,
        sim_param_t params,
        MPI_Datatype datatype,
        MPI_Comm APP_COMM,
        GetPosFunc getPos,
        GetPointFunc getPoint,
        std::string preamble = "")
{
    int rank, nproc;
    MPI_Comm_rank(APP_COMM, &rank);
    MPI_Comm_size(APP_COMM, &nproc);
    auto zlb = new StripeLB<elements::Element<N>, N, N-1> (APP_COMM);
    auto mesh_data = generate_random_particles<N>(rank, params,
                                                  CubePosition<N>(simbox),
                                                  ContractSphereVelocity<N>(0.0, {0.0, 0.0, 0.0}));
    PAR_START_TIMER(lbtime, APP_COMM);
    zlb->partition(mesh_data.els, getPos);
    migrate_data(zlb, mesh_data.els, getPoint, datatype, APP_COMM);
    END_TIMER(lbtime);
    MPI_Allreduce(MPI_IN_PLACE, &lbtime, 1, MPI_TIME, MPI_MAX, APP_COMM);

    par::pcout() << preamble << std::endl;

    Probe probe(nproc);

    return std::make_tuple(zlb, mesh_data, probe, lbtime);
}
#endif //YALBB_EXAMPLE_EXPERIENCE_HPP
