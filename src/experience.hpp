//
// Created by xetql on 10/9/20.
//

#ifndef YALBB_EXAMPLE_EXPERIENCE_HPP
#define YALBB_EXAMPLE_EXPERIENCE_HPP
#include "spatial_elements.hpp"

#include "initial_conditions.hpp"
#include <yalbb/experiment.hpp>

namespace experiment {

namespace {
    template<unsigned N>
    struct MDAllocator {
        virtual std::unique_ptr<MESH_DATA<elements::Element<N>>> alloc() override {
            return std::make_unique<MESH_DATA<elements::Element<N>>>();
        }
    };
}

template<unsigned N, class TParam> class Gravitation : public Experiment<N, MESH_DATA<elements::Element<N>>, TParam>{
protected:
    void setup(MESH_DATA<elements::Element<N>> *mesh_data) override {
        std::array<Real, N> box_center {};
        std::fill(box_center.begin(), box_center.end(), this->params->simsize / 2.0);
        generate_random_particles<N>(mesh_data, this->rank, this->params->seed, this->params->npart,
                                     pos::UniformInSphere<N>(this->params->simsize / 2.0, box_center),
                                     vel::PerpendicularTo(this->params->T0, box_center), MPI_COMM_WORLD);
    }
    std::unique_ptr<MESH_DATA<elements::Element<N>>> alloc() override {
        return std::make_unique<MESH_DATA<elements::Element<N>>>();
    }
public:
    Gravitation(const BoundingBox<N> &simbox, const std::unique_ptr<TParam>& params, MPI_Datatype datatype,
                MPI_Comm appComm, const std::string &name)
            : Experiment<N, MESH_DATA<elements::Element<N>>, TParam>(simbox, params, datatype, appComm, name) {}
};

template<unsigned N, class TParam> class UniformCube : public Experiment<N, MESH_DATA<elements::Element<N>>, TParam>{
protected:
    void setup(MESH_DATA<elements::Element<N>> *mesh_data) override {
        decltype(this->simbox) box = {0.0, this->params->simsize, this->params->simsize/2.0, this->params->simsize};
        generate_random_particles<N>(mesh_data, this->rank, this->params->seed, this->params->npart,
                 pos::UniformInCube<N>(box),
                 vel::Uniform<N>(-this->params->T0), MPI_COMM_WORLD);
    }
    std::unique_ptr<MESH_DATA<elements::Element<N>>> alloc() override {
            return std::make_unique<MESH_DATA<elements::Element<N>>>();
    }
public:
    UniformCube(const BoundingBox<N> &simbox, const std::unique_ptr<TParam>& params, MPI_Datatype datatype,
                     MPI_Comm appComm, const std::string &name)
            : Experiment<N, MESH_DATA<elements::Element<N>>, TParam>(simbox, params, datatype, appComm, name) {}
};

template<unsigned N, class TParam> class ContractSphere   : public Experiment<N, MESH_DATA<elements::Element<N>>, TParam>{
protected:
    void setup(MESH_DATA<elements::Element<N>> *mesh_data) override {
        std::array<Real, N> box_center {};
        std::fill(box_center.begin(), box_center.end(), this->params->simsize / 2.0);
        generate_random_particles<N>(mesh_data, this->rank, this->params->seed, this->params->npart,
                                  pos::UniformInSphere<N>(this->params->simsize / 2.0, box_center),
                                  vel::ContractToPoint<N>(this->params->T0, box_center));

    }
    std::unique_ptr<MESH_DATA<elements::Element<N>>> alloc() override {
            return std::make_unique<MESH_DATA<elements::Element<N>>>();
    }
public:
    ContractSphere(const BoundingBox<N> &simbox, const std::unique_ptr<TParam>& params, MPI_Datatype datatype,
                     MPI_Comm appComm, const std::string &name)
            : Experiment<N, MESH_DATA<elements::Element<N>>, TParam>(simbox, params, datatype, appComm, name) {}
};

template<unsigned N, class TParam> class ExpandSphere     : public Experiment<N, MESH_DATA<elements::Element<N>>, TParam>{
protected:
    void setup(MESH_DATA<elements::Element<N>> *mesh_data) override {
        std::array<Real, N> box_center{};
        std::fill(box_center.begin(), box_center.end(), this->params->simsize / 2.0);
        generate_random_particles<N>(mesh_data, this->rank, this->params->seed, this->params->npart,
                                     pos::UniformInSphere<N>(this->params->simsize / 20.0, box_center),
                                     vel::ExpandFromPoint<N>(this->params->T0, box_center));

    }
    std::unique_ptr<MESH_DATA<elements::Element<N>>> alloc() override {
            return std::make_unique<MESH_DATA<elements::Element<N>>>();
    }
public:
    ExpandSphere(const BoundingBox<N> &simbox, const std::unique_ptr<TParam>& params, MPI_Datatype datatype,
                     MPI_Comm appComm, const std::string &name)
            : Experiment<N, MESH_DATA<elements::Element<N>>, TParam>(simbox, params, datatype, appComm, name) {}
};

template<unsigned N, class TParam> class GravityCircle : public Experiment<N, MESH_DATA<elements::Element<N>>, TParam>{
protected:
    void setup(MESH_DATA<elements::Element<N>> *mesh_data) override {
        std::array<Real, N> box_center{};
        std::fill(box_center.begin(), box_center.end(), this->params->simsize / 2.0);
        generate_random_particles<N>(mesh_data, this->rank, this->params->seed, this->params->npart,
                                     pos::UniformInSphere<N>(this->params->simsize / 2, box_center),
                                     vel::ParallelToAxis<N, 1>(-this->params->T0));

    }
    std::unique_ptr<MESH_DATA<elements::Element<N>>> alloc() override {
            return std::make_unique<MESH_DATA<elements::Element<N>>>();
    }
public:
    GravityCircle(const BoundingBox<N> &simbox, const std::unique_ptr<TParam>& params, MPI_Datatype datatype,
                 MPI_Comm appComm, const std::string &name)
            : Experiment<N, MESH_DATA<elements::Element<N>>, TParam>(simbox, params, datatype, appComm, name) {}
};

template<unsigned N, class TParam> class ExpandingCircles : public Experiment<N, MESH_DATA<elements::Element<N>>, TParam>{
protected:
    void setup(MESH_DATA<elements::Element<N>> *mesh_data) override {
        unsigned n_circles = 2;
        std::array<Real, N> sphere_center_1 = {this->params->simsize / 3.0,       this->params->simsize / 3.0};
        std::array<Real, N> sphere_center_2 = {2.0 * this->params->simsize / 3.0, this->params->simsize / 3.0};

        generate_random_particles<N>(mesh_data, this->rank, this->params->seed, this->params->npart / n_circles,
                                     pos::UniformInSphere<N>(this->params->simsize / 6, sphere_center_1),
                                     vel::ExpandFromPoint<N>(this->params->T0, sphere_center_1));
        generate_random_particles<N>(mesh_data, this->rank, this->params->seed, this->params->npart / n_circles,
                                     pos::UniformInSphere<N>(this->params->simsize / 6, sphere_center_2),
                                     vel::ExpandFromPoint<N>(this->params->T0, sphere_center_2));
    }
    std::unique_ptr<MESH_DATA<elements::Element<N>>> alloc() override {
            return std::make_unique<MESH_DATA<elements::Element<N>>>();
    }
public:
    ExpandingCircles(const BoundingBox<N> &simbox, const std::unique_ptr<TParam>& params, MPI_Datatype datatype,
                  MPI_Comm appComm, const std::string &name)
            : Experiment<N, MESH_DATA<elements::Element<N>>, TParam>(simbox, params, datatype, appComm, name) {}
};
template<unsigned N, class TParam> class ContractingCircles : public Experiment<N, MESH_DATA<elements::Element<N>>, TParam>{
protected:
    void setup(MESH_DATA<elements::Element<N>> *mesh_data) override {
        unsigned n_circles = 2;

        std::array<Real, N> sphere_center_1 = {this->params->simsize / 3.0,       this->params->simsize / 2.0};
        std::array<Real, N> sphere_center_2 = {2.0 * this->params->simsize / 3.0, this->params->simsize / 2.0};

        generate_random_particles<N>(mesh_data, this->rank, this->params->seed, this->params->npart / n_circles,
                                     pos::UniformInSphere<N>(this->params->simsize / 6, sphere_center_1),
                                     vel::ContractToPoint<N>(this->params->T0, sphere_center_1));
        generate_random_particles<N>(mesh_data, this->rank, this->params->seed, this->params->npart / n_circles,
                                     pos::UniformInSphere<N>(this->params->simsize / 6, sphere_center_2),
                                     vel::ContractToPoint<N>(this->params->T0, sphere_center_2));
    }
    std::unique_ptr<MESH_DATA<elements::Element<N>>> alloc() override {
            return std::make_unique<MESH_DATA<elements::Element<N>>>();
    }
public:
    ContractingCircles(const BoundingBox<N> &simbox, const std::unique_ptr<TParam>& params, MPI_Datatype datatype,
                     MPI_Comm appComm, const std::string &name)
            : Experiment<N, MESH_DATA<elements::Element<N>>, TParam>(simbox, params, datatype, appComm, name) {}
};

template<unsigned N, class TParam> class HalfUniformHalfOrthogonal: public Experiment<N, MESH_DATA<elements::Element<N>>, TParam>{
protected:
    void setup(MESH_DATA<elements::Element<N>> *mesh_data) override {
        unsigned n_box = 2;
        decltype(this->simbox) box1 = {0.0, this->params->simsize/2.0, 0.0, this->params->simsize};
        decltype(this->simbox) box2 = {9.0 * this->params->simsize/16.0, this->params->simsize, 0.0, this->params->simsize};

        generate_random_particles<N>(mesh_data, this->rank, this->params->seed, this->params->npart / n_box,
                                     pos::UniformInCube<N>(box1),
                                     vel::Uniform<N>(this->params->T0));

        generate_random_particles<N>(mesh_data, this->rank, this->params->seed, this->params->npart / n_box,
                                     pos::UniformInCube<N>(box2),
                                     vel::ParallelToAxis<N, 0>(-this->params->T0));
    }
    std::unique_ptr<MESH_DATA<elements::Element<N>>> alloc() override {
            return std::make_unique<MESH_DATA<elements::Element<N>>>();
    }
public:
    HalfUniformHalfOrthogonal(const BoundingBox<N> &simbox, const std::unique_ptr<TParam>& params, MPI_Datatype datatype,
                       MPI_Comm appComm, const std::string &name)
            : Experiment<N, MESH_DATA<elements::Element<N>>, TParam>(simbox, params, datatype, appComm, name) {}
};

template<unsigned N, class TParam> class CollidingSpheres : public Experiment<N, MESH_DATA<elements::Element<N>>, TParam>{
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
    std::unique_ptr<MESH_DATA<elements::Element<N>>> alloc() override {
            return std::make_unique<MESH_DATA<elements::Element<N>>>();
    }
public:
    CollidingSpheres(const BoundingBox<N> &simbox, const std::unique_ptr<TParam>& params, MPI_Datatype datatype,
                     MPI_Comm appComm, const std::string &name)
            : Experiment<N, MESH_DATA<elements::Element<N>>, TParam>(simbox, params, datatype, appComm, name) {}
};
template<class TParam>
class Expand2DSphere : public Experiment<2, MESH_DATA<elements::Element<2>>, TParam>{
    static const unsigned N = 2;
protected:
    void setup(MESH_DATA<elements::Element<N>> *mesh_data) override {
        std::array<Real, N> box_center{};
        std::fill(box_center.begin(), box_center.end(), this->params->simsize / 2.0);
        generate_random_particles<N>(mesh_data, this->rank, this->params->seed, this->params->npart,
                                     pos::UniformOnSphere<N>(this->params->simsize / 20.0, box_center),
                                     vel::ExpandFromPoint<N>(this->params->T0, box_center));
    }
    std::unique_ptr<MESH_DATA<elements::Element<2>>> alloc() override {
        return std::make_unique<MESH_DATA<elements::Element<2>>>();
    }
};
}
#endif //YALBB_EXAMPLE_EXPERIENCE_HPP
