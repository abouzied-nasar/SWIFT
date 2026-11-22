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
#include "gpu_launch.h"
#include "cuda_config.h"
#include "task.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
/**
 * @brief initialise GPU data buffers (including their associated metadata)
 *
 * @param buf: the buffers to be initialised
 * @param params: global gpu packing parameters
 * @param send_struct_size: size of struct used for send arrays (both host and
 * device)
 * @param recv_struct_size: size of struct used for recv arrays (both host and
 * device)
 */
void gpu_data_buffers_init(struct gpu_offload_data *buf, const struct gpu_global_pack_params *params,
    const size_t send_struct_size, const size_t recv_struct_size, const int allocate_density_sort) {

  /* Grab some handles */
  /*TODO: Do this properly rather than set it to 1*/
  const size_t n_bundles = 1;//params->n_bundles;
  const size_t part_buffer_size = params->part_buffer_size;

  /* Initialise and set up metadata */
  struct gpu_pack_metadata *md = &(buf->md);
  gpu_pack_metadata_init(md, params);

  /* Initialise density segmented sort storage */
  buf->gpu_md.has_density_sort_storage = allocate_density_sort;
  buf->gpu_md.d_density_sort_keys_in = NULL;
  buf->gpu_md.d_density_sort_keys_out = NULL;
  buf->gpu_md.d_density_sort_values_in = NULL;
  buf->gpu_md.d_density_sort_values_out = NULL;
  buf->gpu_md.d_density_sort_offsets = NULL;
  buf->gpu_md.d_density_sort_temp_storage = NULL;
  buf->gpu_md.density_sort_temp_storage_bytes = 0;

  /* Now allocate arrays */
  cudaError_t cu_error;

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

  /* Create space for cuda events */
  buf->event_end = (cudaEvent_t *)malloc(n_bundles * sizeof(cudaEvent_t));

  for (size_t i = 0; i < n_bundles; i++) {
    cu_error = cudaEventCreate(&(buf->event_end[i]));
    swift_assert(cu_error == cudaSuccess);
  }

#ifdef SWIFT_DEBUG_CHECKS
  md->send_struct_size = send_struct_size;
  md->recv_struct_size = recv_struct_size;
#endif

  /* Now allocate arrays GPU-only metadata arrays*/

  /*Data required for unique sorting*/
  int size_of_cell_start_end = sizeof(int4) * params->cell_start_end_buffer_size;

  /*Allocate memory for cell start and end data on host*/
  cu_error = cudaMallocHost((void **)&buf->gpu_md.cell_i_j_start_end,
                            size_of_cell_start_end);
  swift_assert(cu_error == cudaSuccess);

  /*Allocate memory for cell start and end data on device*/
  cu_error = cudaMalloc((void **)&buf->gpu_md.d_cell_i_j_start_end,
                            size_of_cell_start_end);
  swift_assert(cu_error == cudaSuccess);

  /*Maximum number of CUDA blocks we require to work on an offload*/
  const size_t n_blocks = params->cuda_blockid_buffer_size;

  /*Allocate memory for array containing
   * leaf_computation_id for each cuda block*/
  cu_error = cudaMallocHost((void **)&buf->gpu_md.block_leaf_id,
                            n_blocks * sizeof(int4));
  swift_assert(cu_error == cudaSuccess);

  /*Repeat for device copy*/
  cu_error = cudaMalloc((void **)&buf->gpu_md.d_block_leaf_id,
                            n_blocks * sizeof(int4));
  swift_assert(cu_error == cudaSuccess);

  /* Each pair leaf requires ni + nj sort entries and ni + nj is at most
    *twice the particle count used to calculate number of CUDA blocks we need.
    *Still an estimate since we don't know how many parts will be in a
    *a cell, could be 10s or 1000s*/
  const size_t sort_buffer_size = 2 * n_blocks * GPU_THREAD_BLOCK_SIZE;

  /* A pair leaf can add at most two unique cell and sort requests */
  const int sort_request_capacity = 2 * params->leaf_buffer_size;
  const size_t sort_offset_capacity =
      (size_t)sort_request_capacity + 1;


  /* Allocate host memory for compact directional sort entries */
  cu_error = cudaMallocHost(
      (void **)&buf->gpu_md.sort_entries,
      sort_buffer_size * sizeof(struct gpu_sort_entry));
  swift_assert(cu_error == cudaSuccess);

  /* Allocate one sort metadata entry per leaf computation */
  cu_error = cudaMallocHost(
      (void **)&buf->gpu_md.leaf_sort_metadata,
      params->leaf_buffer_size *
          sizeof(struct gpu_leaf_sort_metadata));
  swift_assert(cu_error == cudaSuccess);

  /* Allocate device memory for the compact directional sort entries */
  cu_error = cudaMalloc(
      (void **)&buf->gpu_md.d_sort_entries,
      sort_buffer_size * sizeof(struct gpu_sort_entry));
  swift_assert(cu_error == cudaSuccess);

  /* Allocate density segmented sort key, value and offset storage */
  if (allocate_density_sort) {
    cu_error = cudaMalloc((void **)&buf->gpu_md.d_density_sort_keys_in, sort_buffer_size * sizeof(float));
    if (cu_error != cudaSuccess)
      error("Failed to allocate density segmented sort input keys: CUDA error '%s' bytes=%zu", cudaGetErrorString(cu_error), sort_buffer_size * sizeof(float));

    cu_error = cudaMalloc((void **)&buf->gpu_md.d_density_sort_keys_out, sort_buffer_size * sizeof(float));
    if (cu_error != cudaSuccess)
      error("Failed to allocate density segmented sort output keys: CUDA error '%s' bytes=%zu", cudaGetErrorString(cu_error), sort_buffer_size * sizeof(float));

    cu_error = cudaMalloc((void **)&buf->gpu_md.d_density_sort_values_in, sort_buffer_size * sizeof(int));
    if (cu_error != cudaSuccess)
      error("Failed to allocate density segmented sort input values: CUDA error '%s' bytes=%zu", cudaGetErrorString(cu_error), sort_buffer_size * sizeof(int));

    cu_error = cudaMalloc((void **)&buf->gpu_md.d_density_sort_values_out, sort_buffer_size * sizeof(int));
    if (cu_error != cudaSuccess)
      error("Failed to allocate density segmented sort output values: CUDA error '%s' bytes=%zu", cudaGetErrorString(cu_error), sort_buffer_size * sizeof(int));

    cu_error = cudaMalloc((void **)&buf->gpu_md.d_density_sort_offsets, sort_offset_capacity * sizeof(int));
    if (cu_error != cudaSuccess)
      error("Failed to allocate density segmented sort offsets: CUDA error '%s' bytes=%zu", cudaGetErrorString(cu_error), sort_offset_capacity * sizeof(int));
  }

  /* Allocate device memory for compact directional sort requests */
  cu_error = cudaMalloc((void **)&buf->gpu_md.d_sort_requests, 2 * params->leaf_buffer_size * sizeof(struct gpu_sort_request));
  swift_assert(cu_error == cudaSuccess);

  /* Query and allocate density segmented sort temporary storage */
  if (allocate_density_sort) {
    size_t temp_storage_bytes = 0;

    cu_error = gpu_density_segmented_sort_query(
        NULL, &temp_storage_bytes,
        buf->gpu_md.d_density_sort_keys_in,
        buf->gpu_md.d_density_sort_keys_out,
        buf->gpu_md.d_density_sort_values_in,
        buf->gpu_md.d_density_sort_values_out,
        sort_buffer_size, sort_request_capacity,
        buf->gpu_md.d_density_sort_offsets, 0);

    if (cu_error != cudaSuccess)
      error(
          "Failed to query density segmented sort temporary storage: "
          "CUDA error '%s' entries=%zu segments=%d",
          cudaGetErrorString(cu_error), sort_buffer_size,
          sort_request_capacity);

    if (temp_storage_bytes == 0)
      error(
          "Density segmented sort returned zero temporary storage: "
          "entries=%zu segments=%d",
          sort_buffer_size, sort_request_capacity);

    buf->gpu_md.density_sort_temp_storage_bytes =
        temp_storage_bytes;

    cu_error = cudaMalloc(
        (void **)&buf->gpu_md.d_density_sort_temp_storage,
        temp_storage_bytes);

    if (cu_error != cudaSuccess)
      error(
          "Failed to allocate density segmented sort temporary storage: "
          "CUDA error '%s' bytes=%zu entries=%zu segments=%d",
          cudaGetErrorString(cu_error), temp_storage_bytes,
          sort_buffer_size, sort_request_capacity);

  #ifdef SWIFT_DEBUG_CHECKS
    const size_t permanent_storage_bytes =
        2 * sort_buffer_size * sizeof(float) +
        2 * sort_buffer_size * sizeof(int) +
        sort_offset_capacity * sizeof(int);

    message(
        "Density segmented sort storage: entries=%zu segments=%d "
        "permanent_bytes=%zu temp_bytes=%zu",
        sort_buffer_size, sort_request_capacity,
        permanent_storage_bytes, temp_storage_bytes);
  #endif
  }

  /* Allocate the compact GPU sort request list */
  buf->gpu_md.sort_requests = (struct gpu_sort_request *)malloc(
      sort_request_capacity * sizeof(struct gpu_sort_request));
  swift_assert(buf->gpu_md.sort_requests != NULL);
  /* Allocate direct lookup array for every unique cell and sort direction */
  buf->gpu_md.sort_request_index = (int *)malloc(
      2 * params->leaf_buffer_size * 13 * sizeof(int));
  swift_assert(buf->gpu_md.sort_request_index != NULL);
  buf->gpu_md.n_sort_requests = 0;
  buf->gpu_md.max_sort_request_count = 0;
  /* Set to -1 to mark all sort requests as un-set */
  memset(buf->gpu_md.sort_request_index, -1,
      2 * params->leaf_buffer_size * 13 * sizeof(int));

  /* Allocate device memory for the sort metadata of each leaf computation */
  cu_error = cudaMalloc(
      (void **)&buf->gpu_md.d_leaf_sort_metadata,
      params->leaf_buffer_size *
          sizeof(struct gpu_leaf_sort_metadata));
  swift_assert(cu_error == cudaSuccess);

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

  buf->gpu_md.n_sort_requests = 0;
  buf->gpu_md.max_sort_request_count = 0;
  memset(buf->gpu_md.sort_request_index, -1,
         2 * buf->md.params.leaf_buffer_size * 13 * sizeof(int));

#ifdef SWIFT_DEBUG_CHECKS

  /* In principle, if our book-keeping is correct, we shouldn't ever
   * need to zero out the contents. So we don't do it outside of debug
   * mode. */

  const struct gpu_global_pack_params pars = buf->md.params;
  const struct gpu_pack_metadata md = buf->md;

  memset(buf->parts_send_d, 0, pars.part_buffer_size * md.send_struct_size);
  memset(buf->parts_recv_d, 0, pars.part_buffer_size * md.recv_struct_size);

  /* Can't do this from the host side, would need to launch cuda kernel
   * We can actually. cudaMemSet() */
  /* initialise to just some meaningless value to silence the compiler */
//  cudaError_t cu_error = cudaErrorMemoryAllocation;
//
//  cu_error =
//      cudaMemset(&buf->d_parts_send_d,
//      0, pars.part_buffer_size * md.send_struct_size);
//  swift_assert(cu_error == cudaSuccess);
//
//  cu_error =
//      cudaMemset(&buf->d_parts_recv_d,
//      0, pars.part_buffer_size * md.recv_struct_size);
//  swift_assert(cu_error == cudaSuccess);

#endif
}

/**
 * @brief Free everything you allocated.
 */
void gpu_data_buffers_free(struct gpu_offload_data *buf) {

  struct gpu_pack_metadata *md = &(buf->md);
  gpu_pack_metadata_free(md);

  cudaError_t cu_error = cudaErrorMemoryAllocation;

  cu_error = cudaFree(buf->d_parts_send_d);
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaFree(buf->d_parts_recv_d);
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaFreeHost(buf->parts_send_d);
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaFreeHost(buf->parts_recv_d);
  swift_assert(cu_error == cudaSuccess);

  free((void *)buf->event_end);

  cu_error = cudaFreeHost(buf->gpu_md.sort_entries);
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaFreeHost(buf->gpu_md.leaf_sort_metadata);
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaFree(buf->gpu_md.d_sort_entries);
  swift_assert(cu_error == cudaSuccess);

  if (buf->gpu_md.has_density_sort_storage) {
    cu_error = cudaFree(buf->gpu_md.d_density_sort_keys_in);
    swift_assert(cu_error == cudaSuccess);

    cu_error = cudaFree(buf->gpu_md.d_density_sort_keys_out);
    swift_assert(cu_error == cudaSuccess);

    cu_error = cudaFree(buf->gpu_md.d_density_sort_values_in);
    swift_assert(cu_error == cudaSuccess);

    cu_error = cudaFree(buf->gpu_md.d_density_sort_values_out);
    swift_assert(cu_error == cudaSuccess);

    cu_error = cudaFree(buf->gpu_md.d_density_sort_offsets);
    swift_assert(cu_error == cudaSuccess);

    cu_error = cudaFree(
        buf->gpu_md.d_density_sort_temp_storage);
    swift_assert(cu_error == cudaSuccess);
  }

  cu_error = cudaFree(buf->gpu_md.d_sort_requests);
  swift_assert(cu_error == cudaSuccess);

  free((void *)buf->gpu_md.sort_requests);

  free((void *)buf->gpu_md.sort_request_index);

}

#ifdef __cplusplus
}
#endif
