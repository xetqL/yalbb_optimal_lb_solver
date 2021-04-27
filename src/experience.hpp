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

namespace experiment {

template<class NumericType>
class StepProducer{
    const std::vector<std::pair<NumericType, unsigned>> steps_repetition;
    unsigned i = 0;
    NumericType step = 0;
    typename decltype(steps_repetition)::const_iterator current_rep;
public:
    explicit StepProducer(std::vector<std::pair<NumericType, unsigned>> steps_rep) :
            steps_repetition(std::move(steps_rep)) {
        current_rep = steps_repetition.begin();
    }

    NumericType next() {
        step += current_rep->first;
        i++;
        if(i >= current_rep->second){
            current_rep++;
            i=0;
        }
        return step;
    }

    bool finished() const {
        return current_rep == steps_repetition.end();
    }
};

using Config = std::tuple<std::string, std::string, sim_param_t, lb::Criterion>;
void load_configs(std::vector<Config>& configs, sim_param_t params){

	
    configs.emplace_back("BBCriterion",  "BBCriterion",      params, lb::BastienMenon{});
	return;
    configs.emplace_back("Static",              "Static",           params, lb::Static{});

    // Automatic criterion
    configs.emplace_back("BBCriterion",  "BBCriterion",      params, lb::BastienMenon{});
    configs.emplace_back("VanillaMenon", "VMenon",           params, lb::VanillaMenon{});
    configs.emplace_back("OfflineMenon", "OMenon",           params, lb::OfflineMenon{});
    configs.emplace_back("PositivMenon", "PMenon",           params, lb::ImprovedMenonNoMax{});
    configs.emplace_back("ZhaiMenon",    "ZMenon",           params, lb::ZhaiMenon{});

    // Periodic
    configs.emplace_back("Periodic 1",       "Periodic_1",    params, lb::Periodic{1});
    for(StepProducer<unsigned> producer({{25, 4}, {50, 10}, {100, 4}}); !producer.finished();){
        unsigned step = producer.next();
        configs.emplace_back(fmt("Periodic %d", step), fmt("Periodic_%d", step), params, lb::Periodic{step});
    }
    // Procassini
    for(StepProducer<unsigned> producer({{25, 6},{50, 9}, {100, 5}, {200, 2}}); !producer.finished();){
        unsigned step = producer.next();
        configs.emplace_back(fmt("Procassini %d", step), fmt("Procassini_%d", step), params, lb::Procassini{step / 100.0f});
    }
    // Marquez
    for(StepProducer<unsigned> producer({{500, 1},{100, 5}, {125, 4}, {250, 2},{500, 2}, {1000, 1}}); !producer.finished();){
        unsigned step = producer.next();
        configs.emplace_back(fmt("Marquez %d", step), fmt("Marquez_%d",step), params, lb::Marquez{step / 100.0f});
    }
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

template<unsigned N, class TParam> class UniformCube : public Experiment<N, TParam> {
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
}
#endif //YALBB_EXAMPLE_EXPERIENCE_HPP
