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
void gpu_data_buffers_init(struct gpu_offload_data *buf,
                           const struct gpu_global_pack_params *params,
                           const size_t send_struct_size,
                           const size_t recv_struct_size,
                           const char is_pair_task) {

  /* Grab some handles */
  const size_t pack_size =
      is_pair_task ? params->pack_size_pair : params->pack_size;
  const size_t n_bundles =
      is_pair_task ? params->n_bundles_pair : params->n_bundles;
  const size_t part_buffer_size = params->part_buffer_size;
  const size_t leaf_buffer_size = params->leaf_buffer_size;

  /* Initialise and set up metadata */
  struct gpu_pack_metadata *md = &(buf->md);
  gpu_pack_metadata_init(md, params);

  /* Now allocate arrays */
  cudaError_t cu_error;
  cu_error =
      cudaMallocHost((void **)&md->bundle_first_part, n_bundles * sizeof(int));
  swift_assert(cu_error == cudaSuccess);

  cu_error =
      cudaMallocHost((void **)&md->bundle_last_part, n_bundles * sizeof(int));
  swift_assert(cu_error == cudaSuccess);

  cu_error =
      cudaMallocHost((void **)&md->bundle_first_leaf, n_bundles * sizeof(int));
  swift_assert(cu_error == cudaSuccess);

  md->count_parts = 0;

  if (is_pair_task) {
    md->ci_list = NULL;
  } else {
    md->ci_list = (struct cell **)malloc(pack_size * sizeof(struct cell *));
  }
  md->task_list = (struct task **)malloc(pack_size * sizeof(struct task *));

  /* Keep track of first and last particles for each self task (particle data
   * is arranged in long arrays containing particles from all the tasks we will
   * work with) Needed for offloading self tasks as we use these to sort
   * through which parts need to interact with which */
  if (is_pair_task) {
    buf->self_task_first_last_part = NULL;
    buf->d_self_task_first_last_part = NULL;
  } else {
    cu_error = cudaMallocHost((void **)&buf->self_task_first_last_part,
                              pack_size * sizeof(int2));
    swift_assert(cu_error == cudaSuccess);
    cu_error = cudaMalloc((void **)&buf->d_self_task_first_last_part,
                          pack_size * sizeof(int2));
    swift_assert(cu_error == cudaSuccess);
  }

  /* Now allocate memory for Buffer and GPU particle arrays */
  cu_error = cudaMalloc((void **)&buf->d_parts_send_d,
                        part_buffer_size * send_struct_size);
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaMalloc((void **)&buf->d_parts_recv_d,
                        part_buffer_size * recv_struct_size);
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaMallocHost((void **)&buf->parts_send_d,
                            part_buffer_size * send_struct_size);
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaMallocHost((void **)&buf->parts_recv_d,
                            part_buffer_size * recv_struct_size);
  swift_assert(cu_error == cudaSuccess);

  if (is_pair_task) {

    md->ci_leaves =
        (struct cell **)malloc(leaf_buffer_size * sizeof(struct cell *));
    md->cj_leaves =
        (struct cell **)malloc(leaf_buffer_size * sizeof(struct cell *));
    md->task_first_last_packed_leaf_pair =
        (int **)malloc(pack_size * sizeof(int *));
    for (size_t i = 0; i < pack_size; i++) {
      md->task_first_last_packed_leaf_pair[i] = (int *)malloc(2 * sizeof(int));
    }

    md->ci_super = (struct cell **)malloc(pack_size * sizeof(struct cell *));
    md->cj_super = (struct cell **)malloc(pack_size * sizeof(struct cell *));

  } else {
    md->ci_leaves = NULL;
    md->cj_leaves = NULL;
    md->task_first_last_packed_leaf_pair = NULL;
    md->ci_super = NULL;
    md->cj_super = NULL;
  }

  /* Create space for cuda events */
  buf->event_end = (cudaEvent_t *)malloc(n_bundles * sizeof(cudaEvent_t));

  for (size_t i = 0; i < n_bundles; i++) {
    cu_error = cudaEventCreate(&(buf->event_end[i]));
    swift_assert(cu_error == cudaSuccess);
  }

#ifdef SWIFT_DEBUG_CHECKS
  md->send_struct_size = send_struct_size;
  md->recv_struct_size = recv_struct_size;
  md->is_pair_task = is_pair_task;
#endif
}

/**
 * @brief perform the initialisations required at the start of each step
 */
void gpu_data_buffers_init_step(struct gpu_offload_data *buf) {

  struct gpu_pack_metadata *md = &buf->md;
  gpu_pack_metadata_init_step(md);
  gpu_data_buffers_reset(buf);
}

/**
 * @brief reset (zero out) the data buffers.
 */
void gpu_data_buffers_reset(struct gpu_offload_data *buf) {

#ifdef SWIFT_DEBUG_CHECKS

  /* In principle, if our book-keeping is correct, we shouldn't ever
   * need to zero out the contents. So we don't do it outside of debug
   * mode. */

  const struct gpu_global_pack_params pars = buf->md.params;
  const struct gpu_pack_metadata md = buf->md;

  if (!md.is_pair_task) {
    for (int i = 0; i < pars.pack_size; i++) {
      buf->self_task_first_last_part[i].x = 0;
      buf->self_task_first_last_part[i].y = 0;
    }
  }

  /* Can't do this from the host side */
  /* for (int i = 0; i < pars.pack_size; i++) { */
  /*   buf->d_self_task_first_last_part[i].x = 0; */
  /*   buf->d_self_task_first_last_part[i].y = 0; */
  /* } */

  bzero(buf->parts_send_d, pars.part_buffer_size * sizeof(md.send_struct_size));
  bzero(buf->parts_recv_d, pars.part_buffer_size * sizeof(md.recv_struct_size));

  /* Can't do this from the host side */
  /* bzero(buf->d_parts_recv_d, pars.part_buffer_size *
   * sizeof(md.send_struct_size)); */
  /* bzero(buf->d_parts_send_d, pars.part_buffer_size) *
   * sizeof(md.recv_struct_size); */

#endif
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
