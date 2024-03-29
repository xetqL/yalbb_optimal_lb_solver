//
// Created by xetql on 4/29/20.
//

#include "zoltan_fn.hpp"
Zoltan_Struct* zoltan_create_wrapper(MPI_Comm comm) {
    auto zz = Zoltan_Create(comm);

    Zoltan_Set_Param(zz, "DEBUG_LEVEL", "0");
    Zoltan_Set_Param(zz, "LB_METHOD", "HSFC");
    Zoltan_Set_Param(zz, "DETERMINISTIC", "1");
    Zoltan_Set_Param(zz, "NUM_GID_ENTRIES", "1");

    Zoltan_Set_Param(zz, "NUM_LID_ENTRIES", "1");
    Zoltan_Set_Param(zz, "OBJ_WEIGHT_DIM", "0");
    Zoltan_Set_Param(zz, "RCB_REUSE", "1");
    Zoltan_Set_Param(zz, "RETURN_LISTS", "ALL");

    Zoltan_Set_Param(zz, "RCB_OUTPUT_LEVEL", "0");
    Zoltan_Set_Param(zz, "KEEP_CUTS", "1");

    Zoltan_Set_Param(zz, "AUTO_MIGRATE", "FALSE");

    return zz;
}
