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
 * @param is_pair_task: if 1, we allocate arrays used for pair tasks
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
  /* TODO: Do these still need to be cudaMallocHost'd?*/
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

  md->task_first_packed_part = malloc(pack_size * sizeof(int));
  md->task_list = (struct task **)malloc(pack_size * sizeof(struct task *));

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

  md->ci_leaves = (struct cell **)malloc(leaf_buffer_size * sizeof(struct cell *));
  md->cj_leaves = (struct cell **)malloc(leaf_buffer_size * sizeof(struct cell *));
  md->task_first_packed_leaf = (int *)malloc(pack_size * sizeof(int));
  md->task_last_packed_leaf = (int *)malloc(pack_size * sizeof(int));

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

  memset(buf->parts_send_d, 0, pars.part_buffer_size * md.send_struct_size);
  memset(buf->parts_recv_d, 0, pars.part_buffer_size * md.recv_struct_size);

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

  free((void *)md->task_list);

  free(md->task_first_packed_leaf);
  free(md->task_last_packed_leaf);
  free((void *)md->ci_leaves);
  free((void *)md->cj_leaves);
  free((void *)md->task_first_packed_part);

}

#ifdef __cplusplus
}
#endif
