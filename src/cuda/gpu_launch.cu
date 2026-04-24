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

/*******************************************************************************
 * This file contains functions used to setup and execute GPU tasks from within
 * runner_main.c. Consider this a translator allowing .cu based functions to be
 * called from within runner_main.c
 ******************************************************************************/

#include <cuda_pipeline.h>
#include <cooperative_groups.h>
#include <cuda_awbarrier_primitives.h>
#include <cuda/barrier>
/* ifdef __cplusplus prevents name mangling. C code sees exact names
 of functions rather than mangled template names produced by C++ */
#ifdef __cplusplus
extern "C" {
#endif

/* Required header files */
#include "cuda_config.h"
#include "cuda_particle_kernels.cuh"
#include "gpu_launch.h"

#include <config.h>
#include <cuda.h>
/* #include <cuda_device_runtime_api.h> */
/* #include <cuda_profiler_api.h> */
/* #include <cuda_runtime.h> */

/**
 * @brief Call the particle SPH density kernel.
 *
 * @param d_parts_send array on device containing particle data
 * @param d_parts_recv array on device to write results into
 * @param d_a current cosmological scale factor
 * @param d_H current Hubble constant
 * @param bundle_first_part index of first particle of this bundle in the
 * d_parts_* arrays
 * @param bundle_n_parts nr of particles in this bundle
 */

__global__ void cuda_launch_density(
    const struct gpu_part_send_d* __restrict__ d_parts_send,
    struct gpu_part_recv_d* __restrict__ d_parts_recv,
    const float d_a, const float d_H,
    const int4* __restrict__ d_cell_i_j_start_end,
    const int2* __restrict__ d_block_leaf_id,
    const double3 space_dim){

    const int bid     = blockIdx.x;
    const int leafid  = d_block_leaf_id[bid].x;
    const int bid_0   = d_block_leaf_id[bid].y;
    int4 cell_se      = d_cell_i_j_start_end[leafid];

    const int ci_start = cell_se.x;
    const int ci_end   = cell_se.y; // cell pos at index [ci_end-1]
    const int cj_start = cell_se.z;
    const int cj_end   = cell_se.w; // cell pos at index [cj_end-1]

    const int b_id_local = bid - bid_0;
    const int tid = threadIdx.x;

    // Let's get the cell positions
    const auto ci_loc = d_parts_send[ci_end - 1].c_loc;
    const auto cj_loc = d_parts_send[cj_end - 1].c_loc;

    double3 shift = {0.0, 0.0, 0.0};
    const double distx = cj_loc.x.x - ci_loc.x.x;
    const double disty = cj_loc.x.y - ci_loc.x.y;
    const double distz = cj_loc.x.z - ci_loc.x.z;

    if (distx < -space_dim.x * 0.5)      shift.x =  space_dim.x;
    else if (distx >  space_dim.x * 0.5)  shift.x = -space_dim.x;
    if (disty < -space_dim.y * 0.5)      shift.y =  space_dim.y;
    else if (disty >  space_dim.y * 0.5)  shift.y = -space_dim.y;
    if (distz < -space_dim.z * 0.5)      shift.z =  space_dim.z;
    else if (distz >  space_dim.z * 0.5)  shift.z = -space_dim.z;

    const double3 shift_i_res = {shift.x + cj_loc.x.x, shift.y + cj_loc.x.y, shift.z + cj_loc.x.z};
    const double3 shift_j_res = {cj_loc.x.x, cj_loc.x.y, cj_loc.x.z};

    // Pass 1: ci <- cj
    neighbour_interactions_density(
        d_parts_send, d_parts_recv,
        ci_start, ci_end - 1,
        cj_start, cj_end - 1,
        shift_i_res, shift_j_res,
        b_id_local, tid
    );

    // Pass 2: cj <- ci (only if not self)
    if (ci_start != cj_start) {
        const double3 shift_ii_res = {cj_loc.x.x, cj_loc.x.y, cj_loc.x.z};
        const double3 shift_jj_res = {shift.x + cj_loc.x.x, shift.y + cj_loc.x.y, shift.z + cj_loc.x.z};

        neighbour_interactions_density(
            d_parts_send, d_parts_recv,
            cj_start, cj_end - 1,
            ci_start, ci_end - 1,
            shift_ii_res, shift_jj_res,
            b_id_local, tid
        );
    }
}


void gpu_launch_density(
    const struct gpu_part_send_d* __restrict__ d_parts_send,
    struct gpu_part_recv_d* __restrict__ d_parts_recv,
    const float d_a, const float d_H,
    int num_blocks_x,
    const int4* __restrict__ d_cell_i_j_start_end,
    const int2* __restrict__ d_block_leaf_id,
    const double3 space_dim,
    cudaStream_t stream)
{
    /* Shared memory allocation. Need two tiles as another tile (1) is
     used for prefetching while tile 0 is used for computations and vice-versa*/
    const size_t shmem = 2 * TILE_J * (sizeof(struct gpu_part_recv_d));//(sizeof(float4) + sizeof(float4)); // 2048 bytes when TILE_J=64

    cuda_launch_density<<<num_blocks_x, GPU_THREAD_BLOCK_SIZE, shmem, stream>>>(
        d_parts_send, d_parts_recv, d_a, d_H,
        d_cell_i_j_start_end, d_block_leaf_id, space_dim);
}

__global__ void cuda_launch_gradient(
    const struct gpu_part_send_g* __restrict__ d_parts_send,
    struct gpu_part_recv_g*      __restrict__ d_parts_recv,
    const float d_a, const float d_H,
    const int4* __restrict__ d_cell_i_j_start_end,
    const int2* __restrict__ d_block_leaf_id,
    const double3 space_dim)
{
    const int bid     = blockIdx.x;
    const int leafid  = d_block_leaf_id[bid].x;
    const int bid_0   = d_block_leaf_id[bid].y;
    int4 cell_se      = d_cell_i_j_start_end[leafid];

    const int ci_start = cell_se.x;
    const int ci_end   = cell_se.y; // cell position at [ci_end-1]
    const int cj_start = cell_se.z;
    const int cj_end   = cell_se.w;

    const int b_id_local = bid - bid_0;
    const int tid        = threadIdx.x;

    // Periodic shift (same as your other kernels)
    const auto ci_loc = d_parts_send[ci_end - 1].c_loc;
    const auto cj_loc = d_parts_send[cj_end - 1].c_loc;

    double3 shift = {0.0, 0.0, 0.0};
    const double distx = cj_loc.x.x - ci_loc.x.x;
    const double disty = cj_loc.x.y - ci_loc.x.y;
    const double distz = cj_loc.x.z - ci_loc.x.z;

    if (distx < -space_dim.x * 0.5)      shift.x =  space_dim.x;
    else if (distx >  space_dim.x * 0.5)  shift.x = -space_dim.x;
    if (disty < -space_dim.y * 0.5)      shift.y =  space_dim.y;
    else if (disty >  space_dim.y * 0.5)  shift.y = -space_dim.y;
    if (distz < -space_dim.z * 0.5)      shift.z =  space_dim.z;
    else if (distz >  space_dim.z * 0.5)  shift.z = -space_dim.z;

    const double3 shift_i_res = {shift.x + cj_loc.x.x, shift.y + cj_loc.x.y, shift.z + cj_loc.x.z};
    const double3 shift_j_res = {cj_loc.x.x, cj_loc.x.y, cj_loc.x.z};

    // Pass 1: ci <- cj (exclude the cell-position slot at end-1)
    neighbour_interactions_gradient(
        d_parts_send, d_parts_recv,
        ci_start, ci_end - 1,
        cj_start, cj_end - 1,
        shift_i_res, shift_j_res,
        b_id_local, tid,
        d_a, d_H
    );

    // Pass 2: cj <- ci (only if not self)
    if (ci_start != cj_start) {
        const double3 shift_ii_res = {cj_loc.x.x, cj_loc.x.y, cj_loc.x.z};
        const double3 shift_jj_res = {shift.x + cj_loc.x.x, shift.y + cj_loc.x.y, shift.z + cj_loc.x.z};

        neighbour_interactions_gradient(
            d_parts_send, d_parts_recv,
            cj_start, cj_end - 1,
            ci_start, ci_end - 1,
            shift_ii_res, shift_jj_res,
            b_id_local, tid,
            d_a, d_H
        );
    }
}

/**
 * @brief Launch the gradient computation on the GPU for a bundle of leaf cells.
 *
 * @param d_parts_send array on device containing particle data
 * @param d_parts_recv array on device to write results into
 * @param d_a current cosmological scale factor
 * @param d_H current Hubble constant
 * @param stream cuda stream to use
 * @param num_blockx_x number of thread blocks to use in x-dimension
 * @param num_blockx_y number of thread blocks to use in y-dimension
 * @param bundle_first_part index of first particle of this bundle in the
 * d_parts_* arrays
 * @param bundle_n_parts nr of particles in this bundle
 */

void gpu_launch_gradient(
    const struct gpu_part_send_g* __restrict__ d_parts_send,
    struct gpu_part_recv_g*      __restrict__ d_parts_recv,
    const float d_a, const float d_H,
    int num_blocks_x,
    const int4* __restrict__ d_cell_i_j_start_end,
    const int2* __restrict__ d_block_leaf_id,
    const double3 space_dim,
    cudaStream_t stream)
{
    // Shared memory: pos4 + vel4 + rac4
    const size_t shmem = 2 * TILE_J * sizeof(struct gpu_part_data_g);  // 3072 bytes when TILE_J=64

    cuda_launch_gradient<<<num_blocks_x, GPU_THREAD_BLOCK_SIZE, shmem, stream>>>(
        d_parts_send, d_parts_recv, d_a, d_H,
        d_cell_i_j_start_end, d_block_leaf_id, space_dim);
}

__global__ void cuda_launch_force(
    const struct gpu_part_send_f* __restrict__ d_parts_send,
    struct gpu_part_recv_f*      __restrict__ d_parts_recv,
    const float d_a, const float d_H,
    const int4* __restrict__ d_cell_i_j_start_end,
    const int2* __restrict__ d_block_leaf_id,
    const double3 space_dim)
{
    const int bid     = blockIdx.x;
    const int leafid  = d_block_leaf_id[bid].x;
    const int bid_0   = d_block_leaf_id[bid].y;
    int4 cell_se      = d_cell_i_j_start_end[leafid];

    const int ci_start = cell_se.x;
    const int ci_end   = cell_se.y; // cell position at [ci_end-1]
    const int cj_start = cell_se.z;
    const int cj_end   = cell_se.w;

    const int b_id_local = bid - bid_0;
    const int tid = threadIdx.x;

    // Periodic shift (same as your density wrapper)
    const auto ci_loc = d_parts_send[ci_end - 1].c_loc;
    const auto cj_loc = d_parts_send[cj_end - 1].c_loc;

    double3 shift = {0.0, 0.0, 0.0};
    const double distx = cj_loc.x.x - ci_loc.x.x;
    const double disty = cj_loc.x.y - ci_loc.x.y;
    const double distz = cj_loc.x.z - ci_loc.x.z;

    if (distx < -space_dim.x * 0.5)      shift.x =  space_dim.x;
    else if (distx >  space_dim.x * 0.5)  shift.x = -space_dim.x;
    if (disty < -space_dim.y * 0.5)      shift.y =  space_dim.y;
    else if (disty >  space_dim.y * 0.5)  shift.y = -space_dim.y;
    if (distz < -space_dim.z * 0.5)      shift.z =  space_dim.z;
    else if (distz >  space_dim.z * 0.5)  shift.z = -space_dim.z;

    const double3 shift_i_res = {shift.x + cj_loc.x.x, shift.y + cj_loc.x.y, shift.z + cj_loc.x.z};
    const double3 shift_j_res = {cj_loc.x.x, cj_loc.x.y, cj_loc.x.z};

    // Pass 1: ci <- cj (exclude cell-position slot at end-1)
    neighbour_interactions_force(
        d_parts_send, d_parts_recv,
        ci_start, ci_end - 1,
        cj_start, cj_end - 1,
        shift_i_res, shift_j_res,
        b_id_local, tid,
        d_a, d_H
    );

    // Pass 2: cj <- ci (only if not self)
    if (ci_start != cj_start) {
        const double3 shift_ii_res = {cj_loc.x.x, cj_loc.x.y, cj_loc.x.z};
        const double3 shift_jj_res = {shift.x + cj_loc.x.x, shift.y + cj_loc.x.y, shift.z + cj_loc.x.z};

        neighbour_interactions_force(
            d_parts_send, d_parts_recv,
            cj_start, cj_end - 1,
            ci_start, ci_end - 1,
            shift_ii_res, shift_jj_res,
            b_id_local, tid,
            d_a, d_H
        );
    }
}

void gpu_launch_force(
    const struct gpu_part_send_f* __restrict__ d_parts_send,
    struct gpu_part_recv_f*      __restrict__ d_parts_recv,
    const float d_a, const float d_H,
    int num_blocks_x,
    const int4* __restrict__ d_cell_i_j_start_end,
    const int2* __restrict__ d_block_leaf_id,
    const double3 space_dim,
    cudaStream_t stream)
{
    // Shared memory: pos4 + vel4 + fbrp4 + cuid4
    const size_t shmem = 2 * TILE_J * sizeof(struct gpu_part_data_f); // 4096 B when TILE_J=64

    cuda_launch_force<<<num_blocks_x, GPU_THREAD_BLOCK_SIZE, shmem, stream>>>(
        d_parts_send, d_parts_recv, d_a, d_H,
        d_cell_i_j_start_end, d_block_leaf_id, space_dim);
}

#ifdef __cplusplus
}
#endif
