//
// Created by xetql on 10/6/20.
//

#ifndef YALBB_EXAMPLE_STRIPELB_HPP
#define YALBB_EXAMPLE_STRIPELB_HPP

#include <yalbb/parallel_utils.hpp>
#include <yalbb/utils.hpp>

#include <mpi.h>
#include <array>
#include <algorithm>
#include <vector>
#include <numeric>
template<class T, int N, int cutDim>
class StripeLB {
    static_assert(0 <= cutDim && cutDim < N);

    using Stripe = std::pair<Real, Real>;

    inline Real distance(const Stripe& a, const Stripe& b) const {
        const auto&[beg_a, end_a] = a;
        const auto&[beg_b, end_b] = b;
        return std::min( std::abs(beg_a - end_b), std::abs(beg_b - end_a) );
    }

    inline bool is_inside(Real v, const Stripe& a) const {
        const auto&[beg_a, end_a] = a;
        return beg_a <= v && v < end_a;
    }

    MPI_Comm comm;
    int rank, world_size;
    std::vector<Stripe> stripes;

public:
    StripeLB(MPI_Comm comm) : comm(comm) {
        MPI_Comm_size(comm, &world_size);
        MPI_Comm_rank(comm, &rank);
        stripes = std::vector<Stripe>(world_size);
    }

    template<class GetWeightFunc, class GetPositionFunc>
    void balance(std::vector<T>& elements, GetWeightFunc getWeight, GetPositionFunc getPosition) {
        const auto n_elements = elements.size();
              auto t_elements = n_elements;
        std::vector<Real> buffer(2*n_elements);
        for(size_t i = 0; i < n_elements; ++i){
            buffer[2*i  ] = getPosition(&elements[i])[cutDim];
            buffer[2*i+1] = getWeight(i);
        }
        std::vector<int> displs(world_size), count(world_size);
        MPI_Gather(&t_elements, 1, MPI_INT, count.data(), 1, MPI_INT, 0, comm);
        std::exclusive_scan(count.begin(), count.end(), displs.begin(), 0);

        MPI_Gather(buffer.data(), 2*n_elements, get_mpi_type<Real>(), );


        Real my_total_weight = std::for_each(elements.begin(), elements.end(), (Real) 0.0, getWeight);
        Real total_weight = (Real) 0.0;
        MPI_Reduce(&my_total_weight, &total_weight, 1, get_mpi_type<Real>(), MPI_SUM, 0, comm);
        // std::partition()
    }

    void lookup_domain(const std::array<Real, N>& point, int* PE) const {
        for(*PE = 0; !is_inside(point.at(cutDim), stripes.at(*PE)); (*PE)++);
    }
    std::vector<int> get_neighbors(int PE, Real min_distance) const {
        std::vector<int> neighbors; neighbors.reserve(world_size);
        for(int neighbor_rank = 0; neighbor_rank < world_size; ++neighbor_rank){
            if(PE != neighbor_rank && distance(stripes.at(PE), stripes.at(neighbor_rank)) <= min_distance)
                neighbors.push_back(neighbor_rank);
        }
        return neighbors;
    }

};

#endif //YALBB_EXAMPLE_STRIPELB_HPP
