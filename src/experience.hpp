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

template<unsigned N, class TParam> class Experiment {
protected:
    BoundingBox<N> simbox;
    const std::unique_ptr<TParam>& params;
    MPI_Datatype datatype;
    MPI_Comm APP_COMM;
    std::string name;
    int rank, nproc;

    virtual void setup(MESH_DATA<elements::Element<N>>* mesh_data) = 0;
public:
    using param_type = TParam;
    Experiment(BoundingBox<N> simbox, const std::unique_ptr<TParam>& params,
               MPI_Datatype datatype, MPI_Comm APP_COMM,
               std::string name) :
            simbox(std::move(simbox)),
            params(params),
            datatype(datatype),
            APP_COMM(APP_COMM),
            name(std::move(name)){
        MPI_Comm_rank(APP_COMM, &rank);
        MPI_Comm_size(APP_COMM, &nproc);
    }

    template<class BalancerType, class GetPosFunc>
    auto init(BalancerType* zlb, GetPosFunc getPos, const std::string& preamble) {
        par::pcout() << preamble << std::endl;

        auto mesh_data = std::make_unique<MESH_DATA<elements::Element<N>>>();

        lb::InitLB<BalancerType>      init {};
        lb::DoPartition<BalancerType> doPartition {};
        lb::AssignPoint<BalancerType> pointAssign {};

        setup(mesh_data.get());
        init(zlb, mesh_data.get());

        Probe probe(nproc);
        PAR_START_TIMER(lbtime, APP_COMM);
        doPartition(zlb, mesh_data.get(), getPos);
        migrate_data(zlb, mesh_data->els, pointAssign, datatype, APP_COMM);
        END_TIMER(lbtime);
        size_t n_els = mesh_data->els.size();
        size_t sum   = mesh_data->els.size();
        MPI_Allreduce(MPI_IN_PLACE, &lbtime, 1, MPI_TIME, MPI_MAX, APP_COMM);
        MPI_Allreduce(MPI_IN_PLACE, &sum, 1,  par::get_mpi_type<size_t>(), MPI_SUM, APP_COMM);
        MPI_Allreduce(MPI_IN_PLACE, &n_els, 1,  par::get_mpi_type<size_t>(), MPI_MAX, APP_COMM);

        probe.push_load_balancing_time(lbtime);
        probe.push_load_balancing_parallel_efficiency((static_cast<Real>(sum) / static_cast<Real>(nproc)) / static_cast<Real>(n_els));
        probe.set_balanced(true);

        par::pcout() << name << std::endl;

        return std::make_tuple(std::move(mesh_data), probe, name);
    }
};

template<unsigned N, class TParam> class UniformCube      : public Experiment<N, TParam> {
protected:
    void setup(MESH_DATA<elements::Element<N>> *mesh_data) override {
        generate_random_particles<N>(mesh_data, this->rank, this->params->seed, this->params->npart,
                 pos::UniformInCube<N>(this->simbox),
                 vel::GoToStripe<N, N-1> {this->params->T0 * this->params->T0,
                                          this->params->simsize - (this->params->simsize / (float) this->nproc)});

    }
public:
    UniformCube(const BoundingBox<N> &simbox, const std::unique_ptr<TParam>& params, MPI_Datatype datatype,
                     MPI_Comm appComm, const std::string &name)
            : Experiment<N, TParam>(simbox, params, datatype, appComm, name) {}
};
template<unsigned N, class TParam> class ContractSphere   : public Experiment<N, TParam>{
protected:
    void setup(MESH_DATA<elements::Element<N>> *mesh_data) override {
        std::array<Real, N> box_center {};
        std::fill(box_center.begin(), box_center.end(), this->params->simsize / 2.0);
        generate_random_particles<N>(mesh_data, this->rank, this->params->seed, this->params->npart,
                                  pos::UniformInSphere<N>(this->params->simsize / 2.0, box_center),
                                  vel::ContractToPoint<N>(this->params->T0, box_center));

    }
public:
    ContractSphere(const BoundingBox<N> &simbox, const std::unique_ptr<TParam>& params, MPI_Datatype datatype,
                     MPI_Comm appComm, const std::string &name)
            : Experiment<N, TParam>(simbox, params, datatype, appComm, name) {}
};
template<unsigned N, class TParam> class ExpandSphere     : public Experiment<N, TParam> {
protected:
    void setup(MESH_DATA<elements::Element<N>> *mesh_data) override {
        std::array<Real, N> box_center{};
        std::fill(box_center.begin(), box_center.end(), this->params->simsize / 2.0);
        generate_random_particles<N>(mesh_data, this->rank, this->params->seed, this->params->npart,
                                     pos::UniformInSphere<N>(this->params->simsize / 20.0, box_center),
                                     vel::ExpandFromPoint<N>(this->params->T0, box_center));

    }
public:
    ExpandSphere(const BoundingBox<N> &simbox, const std::unique_ptr<TParam>& params, MPI_Datatype datatype,
                     MPI_Comm appComm, const std::string &name)
            : Experiment<N, TParam>(simbox, params, datatype, appComm, name) {}
};
template<unsigned N, class TParam> class CollidingSpheres : public Experiment<N, TParam> {
protected:
    void setup(MESH_DATA<elements::Element<N>> *mesh_data) override {
        using namespace vec::generic;
        std::array<Real, N> box_length = get_box_width<N>(this->simbox);
        std::array<Real, N> shift = {box_length[0] / static_cast<Real>(9.0), 0};
        std::array<Real, N> box_center = get_box_center<N>(this->simbox);
        generate_random_particles<N>(mesh_data, this->rank, this->params->seed, this->params->npart / 2,
                                     pos::EquidistantOnDisk<N>(this->rank * this->params->npart / (2 * this->nproc), 0.005, box_center - shift, 0.0),
                                     vel::ParallelToAxis<N, 0>(this->params->T0), this->APP_COMM);

        generate_random_particles<N>(mesh_data, this->rank, this->params->seed, this->params->npart / 2,
                                     pos::EquidistantOnDisk<N>(this->rank * this->params->npart / (2 * this->nproc), 0.005, box_center + shift, 0.0),
                                     vel::ParallelToAxis<N, 0>(-this->params->T0), this->APP_COMM);
    }
public:
    CollidingSpheres(const BoundingBox<N> &simbox, const std::unique_ptr<TParam>& params, MPI_Datatype datatype,
                     MPI_Comm appComm, const std::string &name)
            : Experiment<N, TParam>(simbox, params, datatype, appComm, name) {}
};
template<class TParam>
class Expand2DSphere : public Experiment<2, TParam> {
    static const unsigned N = 2;
protected:
    void setup(MESH_DATA<elements::Element<N>> *mesh_data) override {
        std::array<Real, N> box_center{};
        std::fill(box_center.begin(), box_center.end(), this->params->simsize / 2.0);
        generate_random_particles<N>(mesh_data, this->rank, this->params->seed, this->params->npart,
                                     pos::UniformOnSphere<N>(this->params->simsize / 20.0, box_center),
                                     vel::ExpandFromPoint<N>(this->params->T0, box_center));
    }
};

/*
struct CollidingSphere {
template<unsigned N, class BalancerType, class DoPartition, class GetPosFunc, class GetPointFunc>
auto init(
        BalancerType* zlb,
        BoundingBox<N> simbox,
        sim_param_t params,
        MPI_Datatype datatype,
        MPI_Comm APP_COMM,
        GetPosFunc getPos,
        GetPointFunc pointAssign,
        std::string preamble = "")
{
    using namespace vec::generic;
    auto[rank, nproc] = pre_init_experiment(preamble, APP_COMM);
    auto mesh_data = new MESH_DATA<elements::Element<N>>();

    std::array<Real, N> box_length = get_box_width<N>(simbox);
    std::array<Real, N> shift = {box_length[0] / static_cast<Real>(9.0), 0};
    std::array<Real, N> box_center = get_box_center<N>(simbox);

    generate_random_particles<N>(mesh_data, rank, params.seed, params.npart / 2,
                               pos::EquidistantOnDisk<N>(rank * params.npart / (2*nproc), 0.005, box_center - shift, 0.0),
                               vel::ParallelToAxis<N, 0>(params.T0), APP_COMM);
    generate_random_particles<N>(mesh_data, rank, params.seed+1, params.npart / 2,
                                  pos::Ordered<N>(params.sig_lj / 2, box_center + shift, params.sig_lj*params.sig_lj*0.1, 45.0),
                                  vel::ParallelToAxis<N, 0>(-10.*params.T0));

    lb::InitLB<BalancerType> init{};
    lb::DoPartition<BalancerType> doPartition{};
    init(zlb, mesh_data);
    PAR_START_TIMER(lbtime, APP_COMM);
    doPartition(zlb, mesh_data, getPos);
    migrate_data(zlb, mesh_data->els, pointAssign, datatype, APP_COMM);
    END_TIMER(lbtime);

    return post_init_experiment<N>(mesh_data, lbtime, std::string("Collision"), nproc, APP_COMM);
}
};
*/



}
#endif //YALBB_EXAMPLE_EXPERIENCE_HPP
