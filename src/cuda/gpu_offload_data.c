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
 * @brief initialise GPU data buffers.
 *
 * @brief TODO: parameter docu
 *
 * @params params: global gpu packing parameters
 * @param is_pair_task: Whether we allocate enough space for pair tasks
 * @param send_struct_size: size of struct used for send arrays (both host and
 * device)
 * @param recv_struct_size: size of struct used for recv arrays (both host and
 * device)
 */
void gpu_init_data_buffers(struct gpu_offload_data *buf,
                           const struct gpu_global_pack_params *params,
                           const size_t send_struct_size,
                           const size_t recv_struct_size,
                           const char is_pair_task) {

  size_t target_n_tasks = params->pack_size;
  size_t n_bundles = params->n_bundles;
  size_t bundle_size = params->bundle_size;
  size_t count_max_parts = params->count_max_parts;

  if (is_pair_task) {
    target_n_tasks = params->pack_size_pair;
    n_bundles = params->n_bundles_pair;
    bundle_size = params->bundle_size_pair;
  }

  /* Multiplication factor depending on whether this is for a self or a pair
   * task */
  const size_t self_pair_fact = is_pair_task ? 2 : 1;

  const size_t tasksperbundle = (target_n_tasks + n_bundles - 1) / n_bundles;

  /* Initialise and set up pack_vars */
  struct gpu_pack_vars *pv = &(buf->pv);
  gpu_init_pack_vars(pv);

  /* Now fill out contents */
  pv->target_n_tasks = target_n_tasks;
  pv->bundle_size = bundle_size;
  pv->n_bundles = n_bundles;

  /* TODO: Need to come up with a good estimate for this */
  pv->n_expected_pair_tasks = 4096;

  cudaError_t cu_error = cudaErrorMemoryAllocation;
  cu_error = cudaMallocHost((void **)&pv->bundle_first_part,
                            self_pair_fact * n_bundles * sizeof(int));
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaMallocHost((void **)&pv->bundle_last_part,
                            self_pair_fact * n_bundles * sizeof(int));
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaMallocHost((void **)&pv->bundle_first_task_list,
                            self_pair_fact * n_bundles * sizeof(int));
  swift_assert(cu_error == cudaSuccess);

  pv->tasksperbundle = tasksperbundle;
  pv->count_parts = 0;
  pv->count_max_parts = count_max_parts;

  if (is_pair_task) {
    pv->task_list = NULL;
    pv->ci_list = NULL;
    pv->top_task_list =
        (struct task **)calloc(target_n_tasks, sizeof(struct task *));
  } else {
    pv->task_list =
        (struct task **)calloc(target_n_tasks, sizeof(struct task *));
    pv->ci_list = (struct cell **)calloc(target_n_tasks, sizeof(struct cell *));
    pv->top_task_list = NULL;
  }

  /* A. Nasar: Keep track of first and last particles for each self task
   * (particle data is arranged in long arrays containing particles from all the
   * tasks we will work with) Needed for offloading self tasks as we use these
   * to sort through which parts need to interact with which */
  if (is_pair_task) {
    buf->task_first_part = NULL;
    buf->d_task_first_part = NULL;
#ifdef SWIFT_DEBUG_CHECKS
    buf->task_first_part_size = 0;
    buf->d_task_first_part_size = 0;
#endif
  } else {
    cu_error = cudaMallocHost((void **)&buf->task_first_part,
                              target_n_tasks * sizeof(int2));
    swift_assert(cu_error == cudaSuccess);
    cu_error = cudaMalloc((void **)&buf->d_task_first_part,
                          target_n_tasks * sizeof(int2));
    swift_assert(cu_error == cudaSuccess);
#ifdef SWIFT_DEBUG_CHECKS
    buf->task_first_part_size = target_n_tasks;
    buf->d_task_first_part_size = target_n_tasks;
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
    size_t max_length = 2 * target_n_tasks * 2;

    buf->ci_d = (struct cell **)malloc(max_length * sizeof(struct cell *));
    buf->cj_d = (struct cell **)malloc(max_length * sizeof(struct cell *));
    buf->first_and_last_daughters =
        (int **)malloc(target_n_tasks * 2 * sizeof(int *));
    for (size_t i = 0; i < target_n_tasks * 2; i++) {
      buf->first_and_last_daughters[i] = (int *)malloc(2 * sizeof(int));
    }

    buf->ci_top =
        (struct cell **)malloc(2ul * target_n_tasks * sizeof(struct cell *));
    buf->cj_top =
        (struct cell **)malloc(2ul * target_n_tasks * sizeof(struct cell *));

  } else {
    buf->ci_d = NULL;
    buf->cj_d = NULL;
    buf->first_and_last_daughters = NULL;
    buf->ci_top = NULL;
    buf->cj_top = NULL;
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

  struct gpu_pack_vars *pv = &buf->pv;
  gpu_init_pack_vars_step(pv);
}

/**
 * Free everything you allocated.
 */
void gpu_free_data_buffers(struct gpu_offload_data *buf,
                           const char is_pair_task) {

  struct gpu_pack_vars *pv = &(buf->pv);

  cudaError_t cu_error = cudaErrorMemoryAllocation;
  cu_error = cudaFreeHost(pv->bundle_first_part);
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaFreeHost(pv->bundle_last_part);
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaFreeHost(pv->bundle_first_task_list);
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
    free((void *)pv->top_task_list);

    cu_error = cudaFreeHost(buf->fparti_fpartj_lparti_lpartj);
    swift_assert(cu_error == cudaSuccess);

    for (size_t i = 0; i < pv->target_n_tasks * 2; i++) {
      free(buf->first_and_last_daughters[i]);
    }
    free((void *)buf->first_and_last_daughters);
    free((void *)buf->ci_d);
    free((void *)buf->cj_d);
    free((void *)buf->ci_top);
    free((void *)buf->cj_top);

  } else {

    free((void *)pv->task_list);
    free((void *)pv->ci_list);

    cu_error = cudaFreeHost(buf->task_first_part);
    swift_assert(cu_error == cudaSuccess);

    cu_error = cudaFree(buf->d_task_first_part);
    swift_assert(cu_error == cudaSuccess);
  }
}

#ifdef __cplusplus
}
#endif
