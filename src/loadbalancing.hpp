//
// Created by xetql on 11/11/20.
//

#ifndef YALBB_EXAMPLE_LOADBALANCING_HPP
#define YALBB_EXAMPLE_LOADBALANCING_HPP
#include <yalbb/load_balancing.hpp>

#include "rcb.hpp"
#include "zoltan_fn.hpp"
#include "StripeLB.hpp"
#include "norcb.hpp"

namespace lb {
//template<> struct InitLB<YourPartitioner> {
//    template<class MD> void operator() (YourPartitioner* lb, MD* md) {/* Init your partitioner */}
//};
//template<> struct DoPartition<YourPartitioner> {
//    template<class MD> void operator() (YourPartitioner* lb, MD* md) {/* Do the partitioning */}};
//template<> struct IntersectDomain<YourPartitioner> {
//    Real rc {};
//    void operator() (YourPartitioner* zlb, double x1, double y1, double z1,
//            double x2, double y2, double z2, int* PEs, int* num_found) const
//            { /* Intersect the cube with the sub-domains */ }
//};
//template<> struct AssignPoint<YourPartitioner> {
//    template<class El> void operator() (YourPartitioner* zlb, const El* e, int* PE) {/* Get the owner of e */}
//
//};
//template<> struct Copier<YourPartitioner> {
//    YourPartitioner* operator() (YourPartitioner* zlb) {/* Make a clone of the partitioner */}
//};
//template<> struct Destroyer<YourPartitioner> {
//    void operator() (YourPartitioner* zlb) {/* Destroy your partitioner */}
//};

template<> struct InitLB<StripeLB> { template<class MD> void operator() (StripeLB* lb, MD* md) {} };
template<> struct DoPartition<StripeLB> {
    template<class MD, class GetPosPtrF> void operator() (StripeLB* lb, MD* md, GetPosPtrF getPositionPtrFunc) {
            lb->partition<StripeLB::CUT_ALONG>(md->els, [](auto *e){ return &(e->position); });
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
template<> struct Copier<StripeLB> {
    StripeLB* operator() (StripeLB* zlb) {
        return allocate_from(zlb);
    }
};
template<> struct Destroyer<StripeLB> {
    void operator() (StripeLB* zlb) {
        destroy(zlb);
    }
};



template<> struct InitLB<norcb::NoRCB> { template<class MD> void operator() (norcb::NoRCB* lb, MD* md) {} };
template<> struct InitLB<rcb::RCB> {
    template<class MD>
    void operator() (rcb::RCB* lb, MD* md) {}
};
template<> struct InitLB<Zoltan_Struct> {
    template<class MD>
    void operator() (Zoltan_Struct* lb, MD* md) {
        zoltan_fn_init(lb, md);
    }
};


template<> struct DoPartition<norcb::NoRCB> {
    template<class MD, class GetPosPtrF>
    void operator() (norcb::NoRCB* lb, MD* md, GetPosPtrF getPositionPtrFunc) {
        norcb::partition<Real>(lb, lb->world_size, md->els.begin(), md->els.end(),
                               elements::Element<2>::register_datatype(), lb->comm,
                               [](auto* e){ return &(e->position);},
                               [](auto* e){ return &(e->velocity);} );
    }
};
template<> struct DoPartition<rcb::RCB> {
    template<class MD, class GetPosPtrF>
    void operator() (rcb::RCB* lb, MD* md, GetPosPtrF getPositionPtrFunc) {
        rcb::partition<Real>(lb, lb->world_size, md->els.begin(), md->els.end(),
                               elements::Element<2>::register_datatype(), lb->comm,
                               [](auto* e){ return &(e->position);},
                               [](auto* e){ return &(e->velocity);} );
    }
};
template<> struct DoPartition<Zoltan_Struct> {
    template<class MD, class GetPosPtrF>
    void operator() (Zoltan_Struct* lb, MD* md, GetPosPtrF getPositionPtrFunc) {
        zoltan_fn_init(lb, md);
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
        zlb->get_neighbors((x2+x1)/2.0,(y2+y1) / 2.0,(z2+z1) / 2.0, 4.0*rc_x, PEs, num_found);
        END_TIMER(functime);
    }
};
template<> struct IntersectDomain<rcb::RCB> {
    Real rc {};
    void operator() (rcb::RCB* zlb, double x1, double y1, double z1, double x2, double y2, double z2, int* PEs, int* num_found) const {
        START_TIMER(functime);
        auto rc_x = (x2-x1) / 2;
        auto rc_y = (y2-y1) / 2;
        auto rc_z = (z2-z1) / 2;
        zlb->get_neighbors(x2-rc_x,y2-rc_y,z2-rc_z, rc_x, PEs, num_found);
        END_TIMER(functime);
    }
};
template<> struct IntersectDomain<Zoltan_Struct> {
    Real rc;

    void operator() (Zoltan_Struct* zlb, double x1, double y1, double z1, double x2, double y2, double z2, int* PEs, int* num_found) {
        double bsize = 6.0*rc;
        Zoltan_LB_Box_Assign(zlb, x1-bsize,
                                  y1-bsize,
                                  z1-bsize,
                                  x2+bsize,
                                  y2+bsize,
                                  z2+bsize,
                                  PEs, num_found);
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
template<> struct AssignPoint<rcb::RCB> {
    template<class El>
    void operator() (rcb::RCB* zlb, const El* e, int* PE) {
        zlb->get_owner(e->position.at(0), e->position.at(1), PE);
    }
    void operator() (rcb::RCB* zlb, Real x, Real y, Real z, int* PE) {
        zlb->get_owner(x, y, PE);
    }
};
template<> struct AssignPoint<Zoltan_Struct> {
    template<unsigned N=3>
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
template<> struct Copier<rcb::RCB> {
    rcb::RCB* operator() (rcb::RCB* zlb) {
        return rcb::allocate_from(zlb);
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
template<> struct Destroyer<rcb::RCB> {
    void operator() (rcb::RCB* zlb) {
        rcb::destroy(zlb);
    }
};
template<> struct Destroyer<Zoltan_Struct> {
    void operator() (Zoltan_Struct* zlb) {
        Zoltan_Destroy(&zlb);
    }
};

//Load balancing functor
}
#endif //YALBB_EXAMPLE_LOADBALANCING_HPP
