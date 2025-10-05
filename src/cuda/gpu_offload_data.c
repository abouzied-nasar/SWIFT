/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2025 Abouzied M. A. Nasar (abouzied.nasar@manchester.ac.uk)
 *                    Mladen Ivkovic (mladen.ivkovic@durham.ac.uk)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

/**
 * @file cuda/gpu_offload_data.c
 * @brief functions related to the gpu_offload_data struct, containing data
 * required for offloading
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "gpu_offload_data.h"

#include "error.h"
#include "task.h"

#include <cuda.h>
#include <cuda_runtime.h>

/**
 * @brief initialise GPU data buffers (including their associated metadata)
 *
 * @param buf: the buffers to be initialised
 * @param params: global gpu packing parameters
 * @param send_struct_size: size of struct used for send arrays (both host and
 * device)
 * @param recv_struct_size: size of struct used for recv arrays (both host and
 * device)
 * @param is_pair_task: if 1, we allocate enough space for pair tasks
 */
void gpu_init_data_buffers(struct gpu_offload_data *buf,
                           const struct gpu_global_pack_params *params,
                           const size_t send_struct_size,
                           const size_t recv_struct_size,
                           const char is_pair_task) {

  /* Grab some handles */
  const size_t target_n_tasks =
      is_pair_task ? params->pack_size_pair : params->pack_size;
  const size_t n_bundles =
      is_pair_task ? params->n_bundles_pair : params->n_bundles;
  const size_t count_max_parts = params->count_max_parts;

  /* Multiplication factor depending on whether this is for a self or a pair
   * task */
  const size_t self_pair_fact = is_pair_task ? 2 : 1;

  /* Initialise and set up metadata */
  struct gpu_pack_metadata *md = &(buf->md);
  gpu_pack_metadata_init(md, params);

  /* Now allocate arrays */
  cudaError_t cu_error;
  cu_error = cudaMallocHost((void **)&md->bundle_first_part,
                            self_pair_fact * n_bundles * sizeof(int));
  swift_assert(cu_error == cudaSuccess);
#ifdef SWIFT_DEBUG_CHECKS
  md->bundle_first_part_size = self_pair_fact * n_bundles;
#endif

  cu_error = cudaMallocHost((void **)&md->bundle_last_part,
                            self_pair_fact * n_bundles * sizeof(int));
  swift_assert(cu_error == cudaSuccess);
#ifdef SWIFT_DEBUG_CHECKS
  md->bundle_last_part_size = self_pair_fact * n_bundles;
#endif

  cu_error = cudaMallocHost((void **)&md->bundle_first_leaf,
                            self_pair_fact * n_bundles * sizeof(int));
  swift_assert(cu_error == cudaSuccess);
#ifdef SWIFT_DEBUG_CHECKS
  md->bundle_first_leaf_size = self_pair_fact * n_bundles;
#endif

  md->count_parts = 0;

  /* Watch out, task_list and top_tasks_lists are temporarily a union until we
   * purge one of them. */
  if (is_pair_task) {
    md->ci_list = NULL;
    md->task_list =
        (struct task **)calloc(target_n_tasks, sizeof(struct task *));
#ifdef SWIFT_DEBUG_CHECKS
    md->ci_list_size = 0;
    md->task_list_size = target_n_tasks;
#endif
  } else {
    md->ci_list = (struct cell **)calloc(target_n_tasks, sizeof(struct cell *));
    md->task_list =
        (struct task **)calloc(target_n_tasks, sizeof(struct task *));
#ifdef SWIFT_DEBUG_CHECKS
    md->ci_list_size = target_n_tasks;
    md->task_list_size = target_n_tasks;
#endif
  }

  /* A. Nasar: Keep track of first and last particles for each self task
   * (particle data is arranged in long arrays containing particles from all the
   * tasks we will work with) Needed for offloading self tasks as we use these
   * to sort through which parts need to interact with which */
  if (is_pair_task) {
    buf->self_task_first_last_part = NULL;
    buf->d_self_task_first_last_part = NULL;
#ifdef SWIFT_DEBUG_CHECKS
    buf->self_task_first_last_part_size = 0;
    buf->d_self_task_first_last_part_size = 0;
#endif
  } else {
    cu_error = cudaMallocHost((void **)&buf->self_task_first_last_part,
                              target_n_tasks * sizeof(int2));
    swift_assert(cu_error == cudaSuccess);
    cu_error = cudaMalloc((void **)&buf->d_self_task_first_last_part,
                          target_n_tasks * sizeof(int2));
    swift_assert(cu_error == cudaSuccess);
#ifdef SWIFT_DEBUG_CHECKS
    buf->self_task_first_last_part_size = target_n_tasks;
    buf->d_self_task_first_last_part_size = target_n_tasks;
#endif
  }

  /* Get array of first and last particles for pair interactions. */
  /*A. N.: Needed but only for small part in launch functions. Might
           be useful for recursion on the GPU so keep for now     */
  buf->fparti_fpartj_lparti_lpartj = NULL;
  if (is_pair_task) {
    cu_error = cudaMallocHost((void **)&buf->fparti_fpartj_lparti_lpartj,
                              target_n_tasks * sizeof(int4));
    swift_assert(cu_error == cudaSuccess);
#ifdef SWIFT_DEBUG_CHECKS
    buf->fparti_fpartj_lparti_lpartj_size = target_n_tasks;
  } else {
    buf->fparti_fpartj_lparti_lpartj_size = 0;
#endif
  }

  /* Now allocate memory for Buffer and GPU particle arrays */
  cu_error = cudaMalloc((void **)&buf->d_parts_send_d,
                        self_pair_fact * count_max_parts * send_struct_size);
  swift_assert(cu_error == cudaSuccess);
#ifdef SWIFT_DEBUG_CHECKS
  buf->d_parts_send_size = self_pair_fact * count_max_parts;
#endif

  cu_error = cudaMalloc((void **)&buf->d_parts_recv_d,
                        self_pair_fact * count_max_parts * recv_struct_size);
  swift_assert(cu_error == cudaSuccess);
#ifdef SWIFT_DEBUG_CHECKS
  buf->d_parts_recv_size = self_pair_fact * count_max_parts;
#endif

  cu_error =
      cudaMallocHost((void **)&buf->parts_send_d,
                     self_pair_fact * count_max_parts * send_struct_size);
  swift_assert(cu_error == cudaSuccess);
#ifdef SWIFT_DEBUG_CHECKS
  buf->parts_send_size = self_pair_fact * count_max_parts;
#endif

  cu_error =
      cudaMallocHost((void **)&buf->parts_recv_d,
                     self_pair_fact * count_max_parts * recv_struct_size);
  swift_assert(cu_error == cudaSuccess);
#ifdef SWIFT_DEBUG_CHECKS
  buf->parts_recv_size = self_pair_fact * count_max_parts;
#endif

  if (is_pair_task) {

    /* A. Nasar: Over-setimate better than under-estimate
     * Over-allocated for now but a good guess is multiply by 2 to ensure we
     * always have room for recursing through more tasks than we plan to
     * offload. */
    int max_length = 2 * target_n_tasks * 2;

    md->ci_leaves = (struct cell **)malloc(max_length * sizeof(struct cell *));
    md->cj_leaves = (struct cell **)malloc(max_length * sizeof(struct cell *));
    md->task_first_last_packed_leaf_pair =
        (int **)malloc(target_n_tasks * 2 * sizeof(int *));
    for (size_t i = 0; i < target_n_tasks * 2; i++) {
      md->task_first_last_packed_leaf_pair[i] = (int *)malloc(2 * sizeof(int));
    }

    md->ci_super =
        (struct cell **)malloc(2ul * target_n_tasks * sizeof(struct cell *));
    md->cj_super =
        (struct cell **)malloc(2ul * target_n_tasks * sizeof(struct cell *));

#ifdef SWIFT_DEBUG_CHECKS
    md->ci_leaves_size = target_n_tasks * 2;
    md->cj_leaves_size = target_n_tasks * 2;
    md->task_first_last_packed_leaf_pair_size = target_n_tasks * 2;
    md->ci_super_size = 2ul * target_n_tasks;
    md->cj_super_size = 2ul * target_n_tasks;
#endif

  } else {
    md->ci_leaves = NULL;
    md->cj_leaves = NULL;
    md->task_first_last_packed_leaf_pair = NULL;
    md->ci_super = NULL;
    md->cj_super = NULL;

#ifdef SWIFT_DEBUG_CHECKS
    md->ci_leaves_size = 0;
    md->cj_leaves_size = 0;
    md->task_first_last_packed_leaf_pair_size = 0;
    md->ci_super_size = 0;
    md->cj_super_size = 0;
#endif
  }

  /* Create space for cuda events */
  buf->event_end = (cudaEvent_t *)malloc(n_bundles * sizeof(cudaEvent_t));

  for (size_t i = 0; i < n_bundles; i++) {
    cu_error = cudaEventCreate(&(buf->event_end[i]));
    swift_assert(cu_error == cudaSuccess);
  }
}

/**
 * @brief perform the initialisations required at the start of each step
 */
void gpu_init_data_buffers_step(struct gpu_offload_data *buf) {

  struct gpu_pack_metadata *md = &buf->md;
  gpu_pack_metadata_init_step(md);
}

/**
 * Free everything you allocated.
 */
void gpu_free_data_buffers(struct gpu_offload_data *buf,
                           const char is_pair_task) {

  struct gpu_pack_metadata *md = &(buf->md);

  cudaError_t cu_error = cudaErrorMemoryAllocation;
  cu_error = cudaFreeHost(md->bundle_first_part);
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaFreeHost(md->bundle_last_part);
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaFree(buf->d_parts_send_d);
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaFree(buf->d_parts_recv_d);
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaFreeHost(buf->parts_send_d);
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaFreeHost(buf->parts_recv_d);
  swift_assert(cu_error == cudaSuccess);

  free((void *)buf->event_end);

  if (is_pair_task) {
    free((void *)md->task_list);

    cu_error = cudaFreeHost(buf->fparti_fpartj_lparti_lpartj);
    swift_assert(cu_error == cudaSuccess);

    for (int i = 0; i < md->params.pack_size_pair * 2; i++) {
      free(md->task_first_last_packed_leaf_pair[i]);
    }
    free((void *)md->task_first_last_packed_leaf_pair);
    free((void *)md->ci_leaves);
    free((void *)md->cj_leaves);
    free((void *)md->ci_super);
    free((void *)md->cj_super);

  } else {

    free((void *)md->task_list);
    free((void *)md->ci_list);

    cu_error = cudaFreeHost(buf->self_task_first_last_part);
    swift_assert(cu_error == cudaSuccess);

    cu_error = cudaFree(buf->d_self_task_first_last_part);
    swift_assert(cu_error == cudaSuccess);
  }
}

#ifdef __cplusplus
}
#endif
