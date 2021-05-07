//
// Created by xetql on 11/11/20.
//

#ifndef YALBB_EXAMPLE_LOADBALANCING_HPP
#define YALBB_EXAMPLE_LOADBALANCING_HPP

#include "zoltan_fn.hpp"
#include "StripeLB.hpp"
#include "norcb.hpp"

namespace lb {

template<class InnerLoadBalancer, class Element>
class LoadBalancer {
    InnerLoadBalancer* zlb;
public:
    virtual void init() = 0;
    virtual LoadBalancer* clone() = 0;
    virtual void partition(std::vector<Element>& elements) = 0;
    virtual void intersect(Real rc, double x1, double y1, double z1, double x2, double y2, double z2, int* PEs, int* num_found) = 0;
    virtual void assign(const Element* el, int* PE) = 0;
    virtual std::string name() = 0;
};

template<class T=void> struct InitLB {};                        // Init load balancer functor
template<class T=void> struct Creator {};                       // Creator functor
template<class T=void> struct Copier {};                         // Copy ptr functor, used in optimal finder
template<class T=void> struct Destroyer {};                     // Destructor functor
template<class T=void> struct DoPartition {};                   // Do partitioning functor
template<class T=void> struct IntersectDomain {}; // Domain intersection functor
template<class T=void> struct AssignPoint {};                   // Point assignation functor
template<class T=void> struct NameGetter {};                    // Point assignation functor

template <> struct NameGetter<StripeLB> {
    std::string operator() () { return std::string("StripeLB");}
};

template <> struct NameGetter<Zoltan_Struct> {
    std::string operator() () { return std::string("HSFC");}
};

template <> struct NameGetter<norcb::NoRCB> {
    std::string operator() () { return std::string("NoRCB");}
};

template<> struct InitLB<StripeLB> {
    template<class MD>
    void operator() (StripeLB* lb, MD* md) {}
};
template<> struct InitLB<norcb::NoRCB> {
    template<class MD>
    void operator() (norcb::NoRCB* lb, MD* md) {}
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
        lb->partition<StripeLB::CUT_ALONG>(md->els, [](auto *e){ return &(e->position); });
    }
};
template<> struct DoPartition<norcb::NoRCB> {
    template<class MD, class GetPosPtrF>
    void operator() (norcb::NoRCB* lb, MD* md, GetPosPtrF getPositionPtrFunc) {
        auto partitions = norcb::parallel::partition<Real>(lb->world_size, md->els.begin(), md->els.end(), lb->domain,
                                                           elements::register_datatype<2>(), lb->comm,
                                                           [](auto* e){ return &(e->position);},
                                                           [](auto* e){ return &(e->velocity);} );
        for(auto i = 0; i < lb->world_size; ++i) {
            auto& p = partitions.at(i);
            lb->subdomains.at(i) = std::get<0>(p);
        }
    }
};
template<> struct DoPartition<Zoltan_Struct> {
    template<class MD, class GetPosPtrF>
    void operator() (Zoltan_Struct* lb, MD* md, GetPosPtrF getPositionPtrFunc) {
        Zoltan_Do_LB(lb);
    }
};

template<> struct IntersectDomain<norcb::NoRCB> {
    Real rc {};
    void operator() (norcb::NoRCB* zlb, double x1, double y1, double z1, double x2, double y2, double z2, int* PEs, int* num_found) const {
        START_TIMER(functime);
        auto rc_x = (x2-x1) / 2;
        auto rc_y = (y2-y1) / 2;
        auto rc_z = (z2-z1) / 2;
        zlb->get_neighbors(x2-rc_x,y2-rc_y,z2-rc_z, rc_x, PEs, num_found);
        END_TIMER(functime);
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
    Real rc;
    void operator() (Zoltan_Struct* zlb, double x1, double y1, double z1, double x2, double y2, double z2, int* PEs, int* num_found) {
        START_TIMER(functime);
        auto rc_x = 3.0 * (x2-x1);
        auto rc_y = (y2-y1) / 2;
        auto rc_z = (z2-z1) / 2;
        Zoltan_LB_Box_Assign(zlb, x1-rc_x, y1-rc_x, z1, x2+rc_x, y2+rc_x, z2, PEs, num_found);
        END_TIMER(functime);
    }
};

template<> struct AssignPoint<norcb::NoRCB> {
    template<class El>
    void operator() (norcb::NoRCB* zlb, const El* e, int* PE) {
        zlb->get_owner(e->position.at(0), e->position.at(1), PE);
    }
    void operator() (norcb::NoRCB* zlb, Real x, Real y, Real z, int* PE) {
        zlb->get_owner(x, y, PE);
    }
};
template<> struct AssignPoint<StripeLB> {
    template<class El>
    void operator() (StripeLB* zlb, const El* e, int* PE) {
        zlb->lookup_domain<El::dimension, StripeLB::CUT_ALONG>(e->position.data(), PE);
    }
    void operator() (StripeLB* zlb, Real x, Real y, Real z, int* PE) {
        Real pos[3] = {x,y,z};
        zlb->lookup_domain<3, StripeLB::CUT_ALONG>(pos, PE);
    }
};
template<> struct AssignPoint<Zoltan_Struct> {
    template<int N=3>
    void operator() (Zoltan_Struct* zlb, const elements::Element<N>* e, int* PE) {
        auto pos_in_double = get_as_double_array<N>(e->position);
        Zoltan_LB_Point_Assign(zlb, &pos_in_double.front(), PE);
    }

    void operator() (Zoltan_Struct* zlb, Real x, Real y, Real z, int* PE) {
        double pos[3] = {x,y,z};
        Zoltan_LB_Point_Assign(zlb, pos, PE);
    }
};

template<> struct Copier<norcb::NoRCB> {
    norcb::NoRCB* operator() (norcb::NoRCB* zlb) {
        return norcb::allocate_from(zlb);
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

template<> struct Destroyer<norcb::NoRCB> {
    void operator() (norcb::NoRCB* zlb) {
        norcb::destroy(zlb);
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
/*
template<> struct Creator<norcb::NoRCB> {
    norcb::NoRCB* operator() (MPI_Comm APP_COMM) {
        return new norcb::NoRCB(APP_COMM);
    }
};
template<> struct Creator<StripeLB> {
    StripeLB* operator() (MPI_Comm APP_COMM) {
        return new StripeLB(APP_COMM);
    }
};
template<> struct Creator<Zoltan_Struct> {
    Zoltan_Struct* operator() (MPI_Comm APP_COMM) {
        return zoltan_create_wrapper(APP_COMM);
    }
};
*/
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
        doPart(zlb, mesh_data, getPositionPtrFunc);
    }

};

}
#endif //YALBB_EXAMPLE_LOADBALANCING_HPP
