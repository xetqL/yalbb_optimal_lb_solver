//
// Created by xetql on 10/1/21.
//

#ifndef YALBB_LB_RCB_HPP
#define YALBB_LB_RCB_HPP
#include <yalbb/load_balancing.hpp>
#include "rcb.hpp"

namespace lb {
    template<> struct InitLB<rcb::RCB> {
        template<class MD>
        void operator() (rcb::RCB* lb, MD* md) {}
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
    template<> struct AssignPoint<rcb::RCB> {
        template<class El>
        void operator() (rcb::RCB* zlb, const El* e, int* PE) {
            zlb->get_owner(e->position.at(0), e->position.at(1), PE);
        }
        void operator() (rcb::RCB* zlb, Real x, Real y, Real z, int* PE) {
            zlb->get_owner(x, y, PE);
        }
    };
    template<> struct Copier<rcb::RCB> {
        rcb::RCB* operator() (rcb::RCB* zlb) {
            return rcb::allocate_from(zlb);
        }
    };
    template<> struct Destroyer<rcb::RCB> {
        void operator() (rcb::RCB* zlb) {
            rcb::destroy(zlb);
        }
    };
}
#endif //YALBB_LB_RCB_HPP
