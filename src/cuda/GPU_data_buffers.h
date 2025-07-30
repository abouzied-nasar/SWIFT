#ifndef CUDA_GPU_DATA_BUFFERS_H
#define CUDA_GPU_DATA_BUFFERS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "GPU_pack_vars.h"

void gpu_init_pack_vars_self(
    struct pack_vars_self **pv,
    const int target_n_tasks,
    const int bundle_size,
    const int n_bundles,
    const int count_max_parts_tmp
    );


void gpu_init_pack_vars_pair(
    struct pack_vars_pair **pv,
    const int target_n_tasks,
    const int bundle_size,
    const int n_bundles,
    const int count_max_parts_tmp
    );

#ifdef __cplusplus
}
#endif

#endif
