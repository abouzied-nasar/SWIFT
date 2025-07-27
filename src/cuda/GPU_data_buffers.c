#ifdef __cplusplus
extern "C" {
#endif

#include "GPU_data_buffers.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "error.h"

void gpu_init_pack_vars_self(
    struct pack_vars_self **pv,
    const int target_n_tasks,
    const int bundle_size,
    const int n_bundles,
    const int count_max_parts_tmp
  ) {

  /* A. Nasar: nBundles is the number of task bundles each
  thread has ==> Used to loop through bundles */
  const int nBundles = (target_n_tasks + bundle_size - 1) / bundle_size;
  const int tasksperbundle = (target_n_tasks + nBundles - 1) / nBundles;

  cudaError_t cu_error = cudaErrorMemoryAllocation;
  cu_error = cudaMallocHost((void **)pv, sizeof(struct pack_vars_self));
  swift_assert(cu_error == cudaSuccess);

  //A. Nasar: target_n_tasks defines the total number of leaf-level tasks we will
  //compute for each GPU off-load cycle
  // TODO(mivkov): move comment to parameter documentation
  (*pv)->target_n_tasks = target_n_tasks;
  // A. Nasar: bundle_size defines the number of leaf-level tasks we will
  // compute in each stream
  // TODO(mivkov): move comment to parameter documentation
  (*pv)->bundle_size = bundle_size;
  (*pv)->n_bundles = n_bundles;

  // first part and last part are the first and last particle ids (locally
  // within this thread) for each bundle. A. Nasar: All these are used in GPU offload setup
  cu_error=cudaMallocHost((void **)(*pv)->bundle_first_part, n_bundles * sizeof(int));
  swift_assert(cu_error == cudaSuccess);
  cu_error=cudaMallocHost((void **)(*pv)->bundle_last_part, n_bundles * sizeof(int));
  swift_assert(cu_error == cudaSuccess);
  cu_error=cudaMallocHost((void **)(*pv)->bundle_first_task_list, n_bundles * sizeof(int));
  swift_assert(cu_error == cudaSuccess);

  (*pv)->tasksperbundle = tasksperbundle;
  (*pv)->count_parts = 0;
  (*pv)->count_max_parts = count_max_parts_tmp;

  (*pv)->task_list = (struct task **)calloc(target_n_tasks, sizeof(struct task *));
  (*pv)->cell_list = (struct cell **)calloc(target_n_tasks, sizeof(struct cell *));
}


void gpu_init_pack_vars_pair(
    struct pack_vars_pair **pv,
    const int target_n_tasks,
    const int bundle_size,
    const int n_bundles,
    const int count_max_parts_tmp
  ) {

  /* A. Nasar: nBundles is the number of task bundles each
  thread has ==> Used to loop through bundles */
  const int tasksperbundle = (target_n_tasks + n_bundles - 1) / n_bundles;

  cudaError_t cu_error = cudaErrorMemoryAllocation;
  cu_error = cudaMallocHost((void **)pv, sizeof(struct pack_vars_pair));
  swift_assert(cu_error == cudaSuccess);

  (*pv)->target_n_tasks = target_n_tasks;
  (*pv)->bundle_size = bundle_size;
  (*pv)->n_bundles = n_bundles;

  cu_error=cudaMallocHost((void **)(*pv)->bundle_first_part, 2 * n_bundles * sizeof(int));
  swift_assert(cu_error == cudaSuccess);
  cu_error=cudaMallocHost((void **)(*pv)->bundle_last_part, 2 * n_bundles * sizeof(int));
  swift_assert(cu_error == cudaSuccess);
  cu_error=cudaMallocHost((void **)(*pv)->bundle_first_task_list, 2 * n_bundles * sizeof(int));
  swift_assert(cu_error == cudaSuccess);

  (*pv)->tasksperbundle = tasksperbundle;
  (*pv)->count_parts = 0;
  (*pv)->count_max_parts = count_max_parts_tmp;

  (*pv)->top_task_list = (struct task **)calloc(target_n_tasks, sizeof(struct task *));
}

#ifdef __cplusplus
}
#endif
