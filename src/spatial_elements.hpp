//
// Created by xetql on 12/28/17.
//

#ifndef NBMPI_GEOMETRIC_ELEMENT_HPP
#define NBMPI_GEOMETRIC_ELEMENT_HPP

#include <yalbb/utils.hpp>

#include <array>
#include <iostream>
#include <algorithm>
#include <mpi.h>
#include <type_traits>
#include <numeric>
#include <algorithm>

namespace elements {

    using Real = Real;
    using Index = Integer;

    template<unsigned N> struct Element {
        static const auto dimension = N;

        Index gid;
        Index lid;
        std::array<Real, N> position,  velocity;

        constexpr Element(std::array<Real, N> p, std::array<Real, N> v, const Index gid, const Index lid) : gid(gid), lid(lid), position(p), velocity(v) {}

        constexpr Element() : gid(0), lid(0), position(), velocity() {}

        friend std::ostream &operator<<(std::ostream &os, const Element &element) {
            os << element.position.at(0);
            for(int i = 1; i < N; i++){
                os << "," << element.position.at(i);
            }
            os << "; ";
            os << element.position.at(0);
            for(int i = 1; i < N; i++){
                os << "," << element.velocity.at(i);
            }
            os << element.gid << ";" << element.lid;
            return os;
        }

        inline static MPI_Datatype register_datatype() {
            constexpr const bool UseDoublePrecision = std::is_same<Real, double>::value;
            MPI_Datatype element_datatype, vec_datatype, oldtype_element[2];

            MPI_Aint offset[2], lb, intex;

            int blockcount_element[2];

            // register particle element type
            constexpr int array_size = N;
            auto mpi_raw_datatype = UseDoublePrecision ? MPI_DOUBLE : MPI_FLOAT;

            MPI_Type_contiguous(array_size, mpi_raw_datatype, &vec_datatype);

            MPI_Type_commit(&vec_datatype);

            blockcount_element[0] = 2; //gid, lid
            blockcount_element[1] = 2; //position, velocity

            oldtype_element[0] = MPI_LONG_LONG;
            oldtype_element[1] = vec_datatype;

            MPI_Type_get_extent(MPI_LONG_LONG, &lb, &intex);

            offset[0] = static_cast<MPI_Aint>(0);
            offset[1] = blockcount_element[0] * intex;

            MPI_Type_create_struct(2, blockcount_element, offset, oldtype_element, &element_datatype);

            MPI_Type_commit(&element_datatype);

            return element_datatype;
        }

        inline static std::array<Real, N>* getElementPositionPtr(Element<N>* e){
            return &(e->position);
        }

        inline static std::array<Real, N>* getElementVelocityPtr(Element<N>* e){
            return &(e->velocity);
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Particles function definition
    // Getter (position and velocity)



    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<int N>
    void import_from_file_float(std::string filename, std::vector<Element<N>>& particles) {

        std::ifstream pfile;
        pfile.open(filename, std::ifstream::in);
        if(!pfile.good()) throw std::runtime_error("bad particle file");

        std::string line;
        while (std::getline(pfile, line)) {
            auto parameters = split(line, ';');
            auto str_pos = split(parameters[0], ' ');
            auto str_vel = split(parameters[1], ' ');
            auto str_gid = parameters[3];
            auto str_lid = parameters[4];
            Element<N> e;

            for(int i = 0; i < N; ++i)
                e.position[i] = std::stof(str_pos[i], 0);
            for(int i = 0; i < N; ++i)
                e.velocity[i] = std::stof(str_vel[i], 0);

            e.gid = std::stoll(str_gid);
            e.lid = std::stoll(str_lid);
            particles.push_back(e);
        }
    }

    template<int N>
    void import_from_file_double(std::string filename, std::vector<Element<N>>& particles) {

        std::ifstream pfile;
        pfile.open(filename, std::ifstream::in);
        if(!pfile.good()) throw std::runtime_error("bad particle file");

        std::string line;
        while (std::getline(pfile, line)) {
            auto parameters = split(line, ';');
            auto str_pos = split(parameters[0], ' ');
            auto str_vel = split(parameters[1], ' ');
            auto str_gid = parameters[3];
            auto str_lid = parameters[4];
            Element<N> e;

            for(int i = 0; i < N; ++i)
                e.position[i] = std::stod(str_pos[i], 0);
            for(int i = 0; i < N; ++i)
                e.velocity[i] = std::stod(str_vel[i], 0);

            e.gid = std::stoll(str_gid);
            e.lid = std::stoll(str_lid);
            particles.push_back(e);
        }
    }

    template<int N, class RealType, bool UseDoublePrecision = std::is_same<RealType, double>::value>
    void import_from_file(std::string filename, std::vector<Element<N>>& particles) {
        if constexpr (UseDoublePrecision)
            import_from_file_double<N>(filename, particles);
        else
            import_from_file_float<N>(filename, particles);
    }

    template<int N>
    void export_to_file(std::string filename, const std::vector<Element<N>> elements) {
        std::ofstream particles_data;
        if (file_exists(filename)) std::remove(filename.c_str());
        particles_data.open(filename, std::ofstream::out);
        for(auto const& e : elements) {
            particles_data << e << std::endl;
        }
        particles_data.close();
    }

    template<int N>
    inline Real distance2(const std::array<Real, N>& e1, const std::array<Real, N>& e2)  {
        std::array<Real, N> e1e2;
        for(int i = 0; i < N; ++i) e1e2[i] = std::pow(e1[i] - e2[i], 2);
        return std::accumulate(e1e2.cbegin(), e1e2.cend(), 0.0);
    }

    template<int N>
    inline std::array<Real, N> distance_dim(const std::array<Real, N>& e1, const std::array<Real, N>& e2)  {
        std::array<Real, N> e1e2;
        for(int i = 0; i < N; ++i) e1e2[i] = e1[i] - e2[i];
        return e1e2;
    }

    template<int N>
    const inline Real distance2(const elements::Element<N> &e1, const elements::Element<N> &e2)  {
        return elements::distance2<N>(e1.position, e2.position);
    }

    template<int N, typename T>
    std::vector<Element<N>> transform(const Index length, const T* positions, const T* velocities) {
        std::vector<Element<N>> elements;
        elements.reserve(length);
        for(Index i=0; i < length; ++i) {
            elements.emplace_back({positions[2*i], positions[2*i+1]}, {velocities[2*i],velocities[2*i+1]}, i, i);
        }
        return elements;
    }

    template<int N, typename T>
    void transform(std::vector<Element<N>>& elements, const T* positions, const T* velocities) throw() {
        if(elements.empty()) {
            throw std::runtime_error("Can not transform data into an empty vector");
        }
        std::generate(elements.begin(), elements.end(), [i = 0, id=0, &positions, &velocities]() mutable {
            Element<N> e({positions[i], positions[i+1]}, {velocities[i], velocities[i+1]}, id, id);
            i=i+N;
            id++;
            return e;
        });
    }

    template<int N>
    void serialize_positions(const std::vector<Element<N>>& elements, Real* positions){
        size_t element_id = 0;
        for (auto const& el : elements){
            for(size_t dim = 0; dim < N; ++dim)
                positions[element_id * N + dim] = el.position.at(dim);
            element_id++;
        }
    }

    template<int N, typename T>
    void serialize(const std::vector<Element<N>>& elements, T* positions, T* velocities){
        size_t element_id = 0;
        for (auto const& el : elements){
            for(size_t dim = 0; dim < N; ++dim){
                positions[element_id * N + dim] = (Real) el.position.at(dim);  positions[element_id * N + dim] = (Real) el.position.at(dim);
                velocities[element_id * N + dim] = (Real) el.velocity.at(dim); velocities[element_id * N + dim] = (Real) el.velocity.at(dim);
            }
            element_id++;
        }
    }

}

#endif //NBMPI_GEOMETRIC_ELEMENT_HPP
