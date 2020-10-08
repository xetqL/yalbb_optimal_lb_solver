//
// Created by xetql on 10/6/20.
//

#ifndef YALBB_EXAMPLE_STRIPELB_HPP
#define YALBB_EXAMPLE_STRIPELB_HPP

//#include <yalbb/parallel_utils.hpp>
#include <yalbb/utils.hpp>
#include "parallel/algorithm.hpp"

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

    template<class GetPositionFunc>
    void partition(std::vector<T>& elements, GetPositionFunc getPosition) {
        Real prev = 0.0, next;
        Real n = elements.size(), total_n;
        MPI_Allreduce(&n, &total_n, 1, par::get_mpi_type<Real>(), MPI_SUM, comm);
        size_t avg_n = total_n / world_size;
        for(int pe = 0; pe < world_size; ++pe) {
            next = getPosition(&par::find_nth(elements.begin(), elements.end(), pe * avg_n, [getPosition](auto& v){return getPosition(&v).at(cutDim);})).at(cutDim);
            stripes.at(pe) = {prev, next};
            prev = next;
        }
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
