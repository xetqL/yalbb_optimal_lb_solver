//
// Created by xetql on 10/1/21.
//

#ifndef YALBB_LB_NORCB_H
#define YALBB_LB_NORCB_H

#include <yalbb/load_balancing.hpp>
#include "norcb.hpp"
#include "spatial_elements.hpp"

namespace lb {
    template<> struct InitLB<norcb::NoRCB> { template<class MD> void operator() (norcb::NoRCB* lb, MD* md) {} };
    template<> struct DoPartition<norcb::NoRCB> {
        template<class MD, class GetPosPtrF>
        void operator() (norcb::NoRCB* lb, MD* md, GetPosPtrF getPositionPtrFunc) {
            norcb::partition<Real>(lb, lb->world_size, md->els.begin(), md->els.end(),
                                   elements::Element<2>::register_datatype(), lb->comm,
                                   [](auto* e){ return &(e->position);},
                                   [](auto* e){ return &(e->velocity);} );
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
    template<> struct AssignPoint<norcb::NoRCB> {
        template<class El>
        void operator() (norcb::NoRCB* zlb, const El* e, int* PE) {
            zlb->get_owner(e->position.at(0), e->position.at(1), PE);
        }
        void operator() (norcb::NoRCB* zlb, Real x, Real y, Real z, int* PE) {
            zlb->get_owner(x, y, PE);
        }
    };
    template<> struct Copier<norcb::NoRCB> {
        norcb::NoRCB* operator() (norcb::NoRCB* zlb) {
            return norcb::allocate_from(zlb);
        }
    };
    template<> struct Destroyer<norcb::NoRCB> {
        void operator() (norcb::NoRCB* zlb) {
            norcb::destroy(zlb);
        }
    };
}

#endif //YALBB_LB_NORCB_H
