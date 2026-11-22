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
#ifndef CUDA_GPU_LAUNCH_H
#define CUDA_GPU_LAUNCH_H

#ifdef __cplusplus
extern "C" {
#endif

#include "gpu_part_structs.h"
#include "gpu_sort_structs.h"

#include <cuda_runtime.h>

cudaError_t gpu_density_segmented_sort_query(
    void *d_temp_storage, size_t *temp_storage_bytes,
    const float *d_keys_in, float *d_keys_out,
    const int *d_values_in, int *d_values_out,
    const size_t n_items, const int n_segments,
    const int *d_offsets, cudaStream_t stream);

cudaError_t gpu_launch_prepare_density_sorts(
    const struct gpu_part_send_d* __restrict__ d_parts_send,
    const struct gpu_sort_request* __restrict__ d_sort_requests,
    float* __restrict__ d_sort_keys_in, float* __restrict__ d_sort_keys_out,
    int* __restrict__ d_sort_values_in, int* __restrict__ d_sort_values_out,
    int* __restrict__ d_sort_offsets, void* d_sort_temp_storage,
    const size_t sort_temp_storage_bytes, const size_t n_sort_entries,
    const int n_sort_requests, cudaStream_t stream);

void gpu_launch_prepare_density_target_ranges(
    const float* __restrict__ d_sort_keys,
    struct gpu_leaf_sort_metadata* __restrict__ d_leaf_sort_metadata,
    const int n_leaves, cudaStream_t stream);

void gpu_launch_pack_density_sort_entries(
    const float* __restrict__ d_sort_keys,
    const int* __restrict__ d_sort_values,
    struct gpu_sort_entry* __restrict__ d_sort_entries,
    const int n_sort_entries, cudaStream_t stream);

void gpu_launch_density(
    const struct gpu_part_send_d* __restrict__ d_parts_send,
    struct gpu_part_recv_d* __restrict__ d_parts_recv,
    const float d_a, const float d_H,
    int num_blocks_x,
    const int4* __restrict__ d_cell_i_j_start_end,
    const int4* __restrict__ d_block_leaf_id,
    const float* __restrict__ d_sort_keys,
    const int* __restrict__ d_sort_values,
    const struct gpu_leaf_sort_metadata* __restrict__ d_leaf_sort_metadata,
    const double3 space_dim,
    cudaStream_t stream, const int tester_param);
void gpu_launch_gradient(
    const struct gpu_part_send_g* __restrict__ d_parts_send,
    struct gpu_part_recv_g*      __restrict__ d_parts_recv,
    const float d_a, const float d_H,
    int num_blocks_x,
    const int4* __restrict__ d_cell_i_j_start_end,
    const int4* __restrict__ d_block_leaf_id,
    const struct gpu_sort_entry* __restrict__ d_sort_entries,
    const struct gpu_leaf_sort_metadata* __restrict__ d_leaf_sort_metadata,
    const double3 space_dim,
    cudaStream_t stream, const int tester_param);
void gpu_launch_force(
    const struct gpu_part_send_f* __restrict__ d_parts_send,
    struct gpu_part_recv_f*      __restrict__ d_parts_recv,
    const float d_a, const float d_H,
    int num_blocks_x,
    const int4* __restrict__ d_cell_i_j_start_end,
    const int4* __restrict__ d_block_leaf_id,
    const struct gpu_sort_entry* __restrict__ d_sort_entries,
    const struct gpu_leaf_sort_metadata* __restrict__ d_leaf_sort_metadata,
    const double3 space_dim,
    cudaStream_t stream, const int tester_param);

#ifdef __cplusplus
}
#endif

#endif  // CUDA_GPU_LAUNCH_H
