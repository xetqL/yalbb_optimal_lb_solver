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

namespace experiment{
using Config = std::tuple<std::string, std::string, sim_param_t, lb::Criterion>;
void load_configs(std::vector<Config>& configs, sim_param_t params){

    configs.emplace_back("Static",              "Static",           params, lb::Static{});
    // Periodic
    configs.emplace_back("Periodic 1000",       "Periodic_1000",    params, lb::Periodic{1000});
    configs.emplace_back("Periodic 500",        "Periodic_500",     params, lb::Periodic{500});
    configs.emplace_back("Periodic 250",        "Periodic_250",     params, lb::Periodic{250});
    configs.emplace_back("Periodic 100",        "Periodic_100",     params, lb::Periodic{100});
    configs.emplace_back("Periodic 50",         "Periodic_50",      params, lb::Periodic{50});
    configs.emplace_back("Periodic 25",         "Periodic_25",      params, lb::Periodic{25});
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
}

namespace {

std::pair<int, int> pre_init_experiment(const std::string& preamble, MPI_Comm APP_COMM){
    int rank, nproc;
    MPI_Comm_rank(APP_COMM, &rank);
    MPI_Comm_size(APP_COMM, &nproc);
    par::pcout() << preamble << std::endl;
    return {rank, nproc};
}

template<int N>
auto post_init_experiment(MESH_DATA<elements::Element<N>>* mesh_data, double lbtime, const std::string& func, int nproc, MPI_Comm APP_COMM) {
    MPI_Allreduce(MPI_IN_PLACE, &lbtime, 1, MPI_TIME, MPI_MAX, APP_COMM);
    par::pcout() << func << std::endl;
    Probe probe(nproc);
    return std::make_tuple(mesh_data, probe, lbtime, func);
}

}

struct UniformCube {
    template<int N, class BalancerType, class DoPartition, class GetPosFunc, class GetPointFunc>
    auto init(BalancerType* zlb,
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
        auto mesh_data = generate_random_particles<N>(rank, params,
                                                      pos::UniformInCube<N>(simbox),
                                                      vel::GoToStripe<N, N-1>{params.T0 * params.T0, params.simsize - (params.simsize / (float) nproc)});
        PAR_START_TIMER(lbtime, APP_COMM);
        doPartition(zlb, &mesh_data, getPos);
        migrate_data(zlb, mesh_data.els, pointAssign, datatype, APP_COMM);
        END_TIMER(lbtime);
        // END

        return post_init_experiment<N>(mesh_data, lbtime, std::string("UniformCube"), nproc, APP_COMM);
    }
};
struct ContractSphere{
template<int N, class BalancerType, class DoPartition, class GetPosFunc, class GetPointFunc>
auto init(
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

    return post_init_experiment<N>(mesh_data, lbtime, std::string("ContractSphere"), nproc, APP_COMM);
}
};
struct ExpandSphere {
template<int N, class BalancerType, class DoPartition, class GetPosFunc, class GetPointFunc>
auto init(
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

    return post_init_experiment<N>(mesh_data, lbtime, std::string("ExpandSphere"), nproc, APP_COMM);
}
};
struct Expand2DSphere {
    template<int N, class BalancerType, class DoPartition, class GetPosFunc, class GetPointFunc>
    auto init(
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
                                                      pos::UniformOnSphere<N>(params.simsize / 20.0, box_center),
                                                      vel::ExpandFromPoint<N>(params.T0, box_center));
        PAR_START_TIMER(lbtime, APP_COMM);
        doPartition(zlb, &mesh_data, getPos);
        migrate_data(zlb, mesh_data.els, pointAssign, datatype, APP_COMM);
        END_TIMER(lbtime);

        return post_init_experiment<N>(mesh_data, lbtime, std::string("Expand2DSphere"), nproc, APP_COMM);
    }
};
struct CollidingSphere {
template<int N, class BalancerType, class DoPartition, class GetPosFunc, class GetPointFunc>
auto init(
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
    using namespace vec::generic;
    auto[rank, nproc] = pre_init_experiment(preamble, APP_COMM);

    std::array<Real, N> box_length = get_box_width<N>(simbox);
    std::array<Real, N> shift = {box_length[0] / static_cast<Real>(9.0), 0};
    std::array<Real, N> box_center = get_box_center<N>(simbox);

    auto mesh_data = new MESH_DATA<elements::Element<N>>();
    mesh_data->els.push_back(elements::Element<N>({0.5, 0.5}, {0.0, 0.0}, 0, 0));
    mesh_data->els.push_back(elements::Element<N>({0.5+params.rc/2, 0.5}, {0.0, 0.0}, 0, 0));
    pos::UniformInSphere
    // generate_random_particles<N>(mesh_data, rank, params.seed, params.npart / 2,
    //                           pos::EquidistantOnDisk<N>(rank * params.npart / (2*nproc), 0.005, box_center - shift, 0.0),
    //                           vel::ParallelToAxis<N, 0>(params.T0), APP_COMM);
    // generate_random_particles<N>(mesh_data, rank, params.seed+1, params.npart / 2,
    //                              pos::Ordered<N>(params.sig_lj / 2, box_center + shift, params.sig_lj*params.sig_lj*0.1, 45.0),
    //                              vel::ParallelToAxis<N, 0>(-10.*params.T0));
    std::cout << mesh_data->els.size() << std::endl;
    lb::InitLB<BalancerType> init{};
    init(zlb, mesh_data);
    PAR_START_TIMER(lbtime, APP_COMM);
    doPartition(zlb, mesh_data, getPos);
    migrate_data(zlb, mesh_data->els, pointAssign, datatype, APP_COMM);
    END_TIMER(lbtime);

    return post_init_experiment<N>(mesh_data, lbtime, std::string("Collision"), nproc, APP_COMM);
}
};

}
#endif //YALBB_EXAMPLE_EXPERIENCE_HPP
