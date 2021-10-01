//
// Created by xetql on 10/1/21.
//

#ifndef YALBB_LB_STRIPE_HPP
#define YALBB_LB_STRIPE_HPP
#include <yalbb/load_balancing.hpp>
#include "StripeLB.hpp"

namespace lb {
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
}
#endif //YALBB_LB_STRIPE_HPP
