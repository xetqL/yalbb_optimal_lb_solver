//
// Created by xetql on 02.03.18.
//

#ifndef NBMPI_ZOLTAN_FN_HPP
#define NBMPI_ZOLTAN_FN_HPP

#include <yalbb/parallel_utils.hpp>
#include <yalbb/utils.hpp>

#include "spatial_elements.hpp"
#include "utils.hpp"

#include <cassert>
#include <random>
#include <string>
#include <vector>
#include <zoltan.h>
#include <set>


template<int N>
int get_number_of_objects(void *data, int *ierr) {
    auto *mesh= (MESH_DATA<elements::Element<N>> *)data;
    *ierr = ZOLTAN_OK;
    return mesh->els.size();
}

template<int N>
void get_object_list(void *data, int sizeGID, int sizeLID,
                     ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                     int wgt_dim, float *obj_wgts, int *ierr) {
    size_t i;
    auto mesh= (MESH_DATA<elements::Element<N>> *)data;
    *ierr = ZOLTAN_OK;
    /* In this example, return the IDs of our objects, but no weights.
     * Zoltan will assume equally weighted objects.
     */
    for (i=0; i < mesh->els.size(); i++){
        globalID[i] = mesh->els[i].gid;
        localID[i] = i;
    }
}
template<int N>
int get_num_geometry(void *data, int *ierr) {
    *ierr = ZOLTAN_OK;
    return N;
} 

template<int N>
void get_geometry_list(void *data, int sizeGID, int sizeLID,
                       int num_obj,
                       ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                       int num_dim, double *geom_vec, int *ierr) {
    unsigned int i;

    auto mesh= (MESH_DATA<elements::Element<N>> *)data;

    if ( (sizeGID != 1) || (sizeLID != 1) || (num_dim > 3)){
        *ierr = ZOLTAN_FATAL;
        return;
    }

    *ierr = ZOLTAN_OK;
    double* p = geom_vec;
    for (int obj_idx=0;  obj_idx < num_obj; obj_idx++) {
        i = localID[obj_idx];
        if ((i < 0) || (i >= mesh->els.size())) {
            *ierr = 1; return;
        }
        *p++ = mesh->els[i].position.at(0);
        *p++ = mesh->els[i].position.at(1);
        if constexpr(N == 3) *p++ = mesh->els[i].position.at(2);
    }
}


template<unsigned N>
void zoltan_fn_init(Zoltan_Struct* zz, MESH_DATA<elements::Element<N>>* mesh_data){
    Zoltan_Set_Num_Obj_Fn(     zz, get_number_of_objects<N>,  mesh_data);
    Zoltan_Set_Obj_List_Fn(    zz, get_object_list<N>,        mesh_data);
    Zoltan_Set_Num_Geom_Fn(    zz, get_num_geometry<N>,       mesh_data);
    Zoltan_Set_Geom_Multi_Fn(  zz, get_geometry_list<N>,      mesh_data);
}

inline void Zoltan_Do_LB(Zoltan_Struct* load_balancer) {

    // ZOLTAN VARIABLES
    int changes, numGidEntries, numLidEntries, numImport, numExport;
    ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
    int *importProcs, *importToPart, *exportProcs, *exportToPart;
    // END OF ZOLTAN VARIABLES

    //zoltan_fn_init(load_balancer, mesh_data);
    Zoltan_LB_Partition(load_balancer,      /* input (all remaining fields are output) */
                        &changes,           /* 1 if partitioning was changed, 0 otherwise */
                        &numGidEntries,     /* Number of integers used for a global ID */
                        &numLidEntries,     /* Number of integers used for a local ID */
                        &numImport,         /* Number of vertices to be sent to me */
                        &importGlobalGids,  /* Global IDs of vertices to be sent to me */
                        &importLocalGids,   /* Local IDs of vertices to be sent to me */
                        &importProcs,       /* Process rank for source of each incoming vertex */
                        &importToPart,      /* New partition for each incoming vertex */
                        &numExport,         /* Number of vertices I must send to other processes*/
                        &exportGlobalGids,  /* Global IDs of the vertices I must send */
                        &exportLocalGids,   /* Local IDs of the vertices I must send */
                        &exportProcs,       /* Process to which I send each of the vertices */
                        &exportToPart);     /* Partition to which each vertex will belong */
    Zoltan_LB_Free_Part(&importGlobalGids, &importLocalGids, &importProcs, &importToPart);
    Zoltan_LB_Free_Part(&exportGlobalGids, &exportLocalGids, &exportProcs, &exportToPart);

}

#endif //NBMPI_ZOLTAN_FN_HPP
