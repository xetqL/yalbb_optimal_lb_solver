//
// Created by xetql on 11/11/20.
//

#ifndef YALBB_EXAMPLE_LOADBALANCING_HPP
#define YALBB_EXAMPLE_LOADBALANCING_HPP
#include <yalbb/load_balancing.hpp>


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
//Load balancing functor
}
#endif //YALBB_EXAMPLE_LOADBALANCING_HPP
