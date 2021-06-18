//
// Created by xetql on 11/11/20.
//

#ifndef YALBB_EXAMPLE_LOADBALANCING_HPP
#define YALBB_EXAMPLE_LOADBALANCING_HPP

#include "rcb.hpp"
#include "zoltan_fn.hpp"
#include "StripeLB.hpp"
#include "norcb.hpp"

namespace lb {

struct LoadBalancer {
    std::string lb_method;
    virtual void init(std::any MD) = 0;
    virtual LoadBalancer* clone() = 0;
    virtual void partition(std::any elements) = 0;
    virtual void intersect(Real rc, double x1, double y1, double z1, double x2, double y2, double z2, int* PEs, int* num_found) = 0;
    virtual void assign(std::any el, int* PE) = 0;
    std::string get_name() { return lb_method; }
};

template<class T=void> struct InitLB {};                        // Init load balancer functor
template<class T=void> struct Creator {};                       // Creator functor
template<class T=void> struct Copier {};                         // Copy ptr functor, used in optimal finder
template<class T=void> struct Destroyer {};                     // Destructor functor
template<class T=void> struct DoPartition {};                   // Do partitioning functor
template<class T=void> struct IntersectDomain {}; // Domain intersection functor
template<class T=void> struct AssignPoint {};                   // Point assignation functor
template<class T=void> struct NameGetter {};                    // Point assignation functor

template<unsigned N>
struct ZoltanLoadBalancer : public LoadBalancer
{
    Zoltan_Struct* zlb;

    ZoltanLoadBalancer(Zoltan_Struct* from, const std::string& method){
        zlb = Zoltan_Copy(from);
        lb_method = method;
    }

    ZoltanLoadBalancer(const char* method, MPI_Comm APP_COMM) {
        lb_method = std::string(method);
        float ver;
        if(Zoltan_Initialize(0, nullptr, &ver) != ZOLTAN_OK) {
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
        zlb = Zoltan_Create(APP_COMM);
        Zoltan_Set_Param(zlb, "DEBUG_LEVEL", "0");
        Zoltan_Set_Param(zlb, "LB_METHOD", lb_method.c_str());
        Zoltan_Set_Param(zlb, "DETERMINISTIC", "1");
        Zoltan_Set_Param(zlb, "NUM_GID_ENTRIES", "1");

        Zoltan_Set_Param(zlb, "NUM_LID_ENTRIES", "1");
        Zoltan_Set_Param(zlb, "OBJ_WEIGHT_DIM", "0");
        Zoltan_Set_Param(zlb, "RCB_REUSE", "1");
        Zoltan_Set_Param(zlb, "RETURN_LISTS", "ALL");

        Zoltan_Set_Param(zlb, "RCB_OUTPUT_LEVEL", "0");
        Zoltan_Set_Param(zlb, "KEEP_CUTS", "1");

        Zoltan_Set_Param(zlb, "AUTO_MIGRATE", "FALSE");
    }

    ~ZoltanLoadBalancer(){
        Zoltan_Destroy(&zlb);
    }

    void init(std::any MD) override {
        zoltan_fn_init(zlb, std::any_cast<MESH_DATA<elements::Element<N>>*>(MD));
    }

    LoadBalancer *clone() override {
        return new ZoltanLoadBalancer<N>(zlb, lb_method);
    }

    void partition(std::any __md) override {
        auto md = std::any_cast<MESH_DATA<elements::Element<N>>*>(__md);
        zoltan_fn_init(zlb, md);
        Zoltan_Do_LB(zlb);
    }

    void intersect(Real rc, double x1, double y1, double z1, double x2, double y2, double z2, int *PEs,
                   int *num_found) override {
        Zoltan_LB_Box_Assign(zlb, x1, y1, z1, x2, y2, z2, PEs, num_found);
    }

    void assign(std::any el, int *PE) override {
        auto e = std::any_cast<elements::Element<N>*>(el);
        auto pos_in_double = get_as_double_array<N>(e->position);
        Zoltan_LB_Point_Assign(zlb, &pos_in_double.front(), PE);
    }

};
/*
template<unsigned N>
struct NoRCBLoadBalancer : public LoadBalancer {
    norcb::NoRCB* zlb;

    NoRCBLoadBalancer(norcb::NoRCB* from, const std::string& method){
        zlb = norcb::allocate_from(from);
        lb_method = method;
    }

    NoRCBLoadBalancer(const char* method, MPI_Comm APP_COMM) {
        lb_method = std::string(method);
        zlb = new norcb::NoRCB();
    }

    ~NoRCBLoadBalancer(){
        norcb::destroy(zlb);
    }

    void init(std::any MD) override {

    }

    LoadBalancer *clone() override {
        return new NoRCBLoadBalancer<N>(zlb, lb_method);
    }

    void partition(std::any elements) override {

    }

    void intersect(Real rc, double x1, double y1, double z1, double x2, double y2, double z2, int *PEs,
                   int *num_found) override {
        auto rc_x = (x2-x1) / 2;
        auto rc_y = (y2-y1) / 2;
        auto rc_z = (z2-z1) / 2;
        zlb->get_neighbors(x2-rc_x,y2-rc_y,z2-rc_z, rc_x, PEs, num_found);
    }

    void assign(std::any el, int *PE) override {

    }
};
*/

template <> struct NameGetter<StripeLB> {
    std::string operator() () { return std::string("StripeLB");}
};
template <> struct NameGetter<Zoltan_Struct> {
    std::string operator() () { return std::string("HSFC"); }
};
template <> struct NameGetter<norcb::NoRCB> {
    std::string operator() () { return std::string("NoRCB");}
};
template <> struct NameGetter<rcb::RCB> {
    std::string operator() () { return std::string("RCB");}
};

template<> struct InitLB<StripeLB> {
    template<class MD>
    void operator() (StripeLB* lb, MD* md) {}
};
template<> struct InitLB<norcb::NoRCB> {
    template<class MD>
    void operator() (norcb::NoRCB* lb, MD* md) {}
};
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

template<> struct DoPartition<StripeLB> {
    template<class MD, class GetPosPtrF>
    void operator() (StripeLB* lb, MD* md, GetPosPtrF getPositionPtrFunc) {
        lb->partition<StripeLB::CUT_ALONG>(md->els, [](auto *e){ return &(e->position); });
    }
};
template<> struct DoPartition<norcb::NoRCB> {
    template<class MD, class GetPosPtrF>
    void operator() (norcb::NoRCB* lb, MD* md, GetPosPtrF getPositionPtrFunc) {
        norcb::partition<Real>(lb, lb->world_size, md->els.begin(), md->els.end(),
                               elements::register_datatype<2>(), lb->comm,
                               [](auto* e){ return &(e->position);},
                               [](auto* e){ return &(e->velocity);} );
    }
};
template<> struct DoPartition<rcb::RCB> {
    template<class MD, class GetPosPtrF>
    void operator() (rcb::RCB* lb, MD* md, GetPosPtrF getPositionPtrFunc) {
        rcb::partition<Real>(lb, lb->world_size, md->els.begin(), md->els.end(),
                               elements::register_datatype<2>(), lb->comm,
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
        Zoltan_LB_Box_Assign(zlb, x1, y1, z1, x2, y2, z2, PEs, num_found);
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
template<> struct AssignPoint<rcb::RCB> {
    template<class El>
    void operator() (rcb::RCB* zlb, const El* e, int* PE) {
        zlb->get_owner(e->position.at(0), e->position.at(1), PE);
    }
    void operator() (rcb::RCB* zlb, Real x, Real y, Real z, int* PE) {
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
template<> struct Copier<rcb::RCB> {
    rcb::RCB* operator() (rcb::RCB* zlb) {
        return rcb::allocate_from(zlb);
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
template<> struct Destroyer<rcb::RCB> {
    void operator() (rcb::RCB* zlb) {
        rcb::destroy(zlb);
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
