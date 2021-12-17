//
// Created by xetql on 10/1/21.
//

#ifndef YALBB_LB_ZOLTAN_HPP
#define YALBB_LB_ZOLTAN_HPP
#include <yalbb/load_balancing.hpp>
#include "zoltan_fn.hpp"

namespace lb {
    template<> struct InitLB<Zoltan_Struct> {
        template<class MD>
        void operator() (Zoltan_Struct* lb, MD* md) {
            zoltan_fn_init(lb, md);
        }
    };
    template<> struct DoPartition<Zoltan_Struct> {
        template<class MD, class GetPosPtrF>
        void operator() (Zoltan_Struct* lb, MD* md, GetPosPtrF getPositionPtrFunc) {
            zoltan_fn_init(lb, md);
            Zoltan_Do_LB(lb);
        }
    };
    template<> struct IntersectDomain<Zoltan_Struct> {
        Real rc;

        void operator() (Zoltan_Struct* zlb, double x1, double y1, double z1, double x2, double y2, double z2, int* PEs, int* num_found) {
            Zoltan_LB_Box_Assign(zlb,
                                 x1,
                                 y1,
                                 z1,
                                 x2,
                                 y2,
                                 z2,
                                 PEs, num_found);
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
    template<> struct Copier<Zoltan_Struct> {
        Zoltan_Struct* operator() (Zoltan_Struct* zlb) {
            return Zoltan_Copy(zlb);
        }
    };
    template<> struct Destroyer<Zoltan_Struct> {
        void operator() (Zoltan_Struct* zlb) {
            Zoltan_Destroy(&zlb);
        }
    };
}
#endif //YALBB_LB_ZOLTAN_HPP
