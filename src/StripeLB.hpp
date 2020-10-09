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
struct StripeLB {
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


    StripeLB(MPI_Comm comm) : comm(comm) {
        MPI_Comm_size(comm, &world_size);
        MPI_Comm_rank(comm, &rank);
        stripes = std::vector<Stripe>(world_size);
    }

    template<class GetPositionFunc>
    void partition(std::vector<T>& elements, GetPositionFunc getPosition) {
        Real prev = std::numeric_limits<Real>::lowest(), next;
        Real n = elements.size(), total_n;
        MPI_Allreduce(&n, &total_n, 1, par::get_mpi_type<Real>(), MPI_SUM, comm);
        size_t avg_n = total_n / world_size;
        int pe;
        for(pe = 1; pe < (world_size); ++pe) {
            next = par::find_nth(elements.begin(), elements.end(), pe * avg_n, comm, [getPosition](auto v){return getPosition(&v)->at(cutDim);});
            stripes.at(pe-1) = {prev, next};
            prev = next;
            par::pcout() <<stripes.at(pe-1).first << ", " << stripes.at(pe-1).second << std::endl;
        }
        stripes.at(pe-1) = {prev, std::numeric_limits<Real>::max()};
        par::pcout() << stripes.at(pe-1).first << ", " << stripes.at(pe-1).second << std::endl;



    }
    void lookup_domain(const std::array<Real, N>& point, int* PE) const {

        for(*PE = 0; (*PE) < world_size ; (*PE)++) {
            if(is_inside(point.at(cutDim), stripes.at(*PE))) return;
        }

        if(*PE == world_size) {
            std::cout << point[cutDim] << std::endl;
            throw std::logic_error("An element must belong to someone ! at");
        }
    }

    std::vector<int> get_neighbors(int PE, Real min_distance) const {
        std::vector<int> neighbors; neighbors.reserve(world_size);
        for(int neighbor_rank = 0; neighbor_rank < world_size; ++neighbor_rank){
            if(PE != neighbor_rank && distance(stripes.at(PE), stripes.at(neighbor_rank)) <= min_distance)
                neighbors.push_back(neighbor_rank);
        }
        return neighbors;
    }
    friend StripeLB<T,N,cutDim>* allocate_from(StripeLB<T, N, cutDim>& t);
    friend StripeLB<T,N,cutDim>* allocate_from(StripeLB<T, N, cutDim>* t);
};

template<class T, int N, int C>
StripeLB<T,N,C>* allocate_from(StripeLB<T,N,C>& t) {
    auto* ptr = new StripeLB<T,N,C>(t.comm);
    std::copy(t.stripes.begin(), t.stripes.end(), ptr->stripes.begin());
    ptr->rank = t.rank;
    ptr->world_size = t.world_size;
    return ptr;
}



template<class T, int N, int C>
void destroy(StripeLB<T,N,C>* t) {
    delete t;
}

#endif //YALBB_EXAMPLE_STRIPELB_HPP
