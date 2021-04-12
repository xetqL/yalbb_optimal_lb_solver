//
// Created by xetql on 11/11/20.
//

#ifndef YALBB_EXAMPLE_LOADBALANCING_HPP
#define YALBB_EXAMPLE_LOADBALANCING_HPP

#include "zoltan_fn.hpp"
#include "StripeLB.hpp"

namespace lb {

template<class T=void> struct InitLB {};         // Init load balancer functor
template<class T=void> struct Creator {};        // Creator functor
template<class T=void> struct Copier{};          // Copy ptr functor
template<class T=void> struct Destroyer {};      // Destructor functor

template<class T=void> struct DoPartition {};    // Do partitioning functor
template<class T=void> struct IntersectDomain {};// Domain intersection functor
template<class T=void> struct AssignPoint {};    // Point assignation functor


template<> struct InitLB<StripeLB> {
    template<class MD>
    void operator() (StripeLB* lb, MD* md) {}
};

template<> struct InitLB<Zoltan_Struct> {
    template<class MD>
    void operator() (Zoltan_Struct* lb, MD* md) {
        zoltan_fn_init(lb, md);
    }
};

template<> struct DoPartition<StripeLB> {
    template<class MD, class GetPosPtrF>
    void operator() (StripeLB* lb, MD* md, GetPosPtrF getPositionPtrFunc) {
        lb->partition<StripeLB::CUT_ALONG>(md->els, getPositionPtrFunc);
    }
};
template<> struct DoPartition<Zoltan_Struct> {
    template<class MD, class GetPosPtrF>
    void operator() (Zoltan_Struct* lb, MD* md, GetPosPtrF getPositionPtrFunc) {
        Zoltan_Do_LB(md, lb);
    }
};
template<> struct IntersectDomain<StripeLB> {
    Real rc {};
    void operator() (StripeLB* zlb, double x1, double y1, double z1, double x2, double y2, double z2, int* PEs, int* num_found) const {
        auto neighbors = zlb->get_neighbors(zlb->rank, rc);
        std::copy(neighbors.begin(), neighbors.end(), PEs);
        *num_found = neighbors.size();
    }
};
template<> struct IntersectDomain<Zoltan_Struct> {
    Real rc{ 0 };
    void operator() (Zoltan_Struct* zlb, double x1, double y1, double z1, double x2, double y2, double z2, int* PEs, int* num_found) {
        Zoltan_LB_Box_Assign(zlb, x1, y1, z1, x2, y2, z2, PEs, num_found);
    }
};

template<> struct AssignPoint<StripeLB> {
    template<class El>
    void operator() (StripeLB* zlb, const El* e, int* PE) {
        zlb->lookup_domain<El::dimension, StripeLB::CUT_ALONG>(e->position.data(), PE);
    }
};
template<> struct AssignPoint<Zoltan_Struct> {
    template<class El>
    void operator() (Zoltan_Struct* zlb, const El* e, int* PE) {
        auto pos_in_double = get_as_double_array<El::dimension>(e->position);
        Zoltan_LB_Point_Assign(zlb, &pos_in_double.front(), PE);
    }
};

template<> struct Copier<StripeLB> {
    StripeLB* operator() (StripeLB* zlb) {
        return allocate_from(zlb);
    }
};
template<> struct Copier<Zoltan_Struct> {
    Zoltan_Struct* operator() (Zoltan_Struct* zlb) {
        return Zoltan_Copy(zlb);
    }
};

template<> struct Destroyer<StripeLB> {
    void operator() (StripeLB* zlb) {
        destroy(zlb);
    }
};
template<> struct Destroyer<Zoltan_Struct> {
    void operator() (Zoltan_Struct* zlb) {
        Zoltan_Destroy(&zlb);
    }
};

template<> struct Creator<StripeLB> {
    template<class... Args>
    StripeLB* operator() (Args... args) {
        return new StripeLB(args...);
    }
};
template<> struct Creator<Zoltan_Struct> {
    template<class... Args>
    Zoltan_Struct* operator() (Args... args) {
        return zoltan_create_wrapper(args...);
    }
};

//Load balancing functor
template<class LB, class GetPosF>
struct DoLB {
    Real rc;

    MPI_Datatype datatype;
    MPI_Comm APP_COMM;

    GetPosF             getPositionPtrFunc;
    IntersectDomain<LB> boxIntersectFunc{rc};
    DoPartition<LB>     doPart{};

    DoLB(Real rc, MPI_Datatype datatype, MPI_Comm APP_COMM, GetPosF getPosF) :
        rc(rc), datatype(datatype), APP_COMM(APP_COMM), getPositionPtrFunc(getPosF){}

    template<class T>
    void operator() (LB* zlb, MESH_DATA<T>* mesh_data) {
        constexpr auto N = T::dimension;
        /*std::vector<T> sampled;
        for(int i = 0; i < 5; ++i)
            std::copy(mesh_data->els.begin(), mesh_data->els.end(), std::back_inserter(sampled));
        MESH_DATA<T> interactions;
        auto bbox      = get_bounding_box<N>(rc, getPositionPtrFunc, sampled);
        auto remote_el = retrieve_ghosts<N>(zlb, sampled, bbox, boxIntersectFunc, rc, datatype, APP_COMM);
        std::vector<Index> lscl, head;
        const auto nlocal  = sampled.size();
        apply_resize_strategy(&lscl,   nlocal + remote_el.size() );
        CLL_init<N, T>({{sampled.data(), nlocal}, {remote_el.data(), remote_el.size()}}, getPositionPtrFunc, bbox, rc, &head, &lscl);
        CLL_foreach_interaction(sampled.data(), nlocal, remote_el.data(), getPositionPtrFunc, bbox, rc, &head, &lscl,
                                [&interactions](const auto *r, const auto *s) {
                                    interactions.els.push_back(midpoint<N>(*r, *s));
                                });*/
        doPart(zlb, mesh_data, getPositionPtrFunc);
    }

};

}
#endif //YALBB_EXAMPLE_LOADBALANCING_HPP
