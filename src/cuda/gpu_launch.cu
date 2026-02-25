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
    const struct gpu_part_send_d *__restrict__ d_parts_send,
    struct gpu_part_recv_d *__restrict__ d_parts_recv, const float d_a,
    const float d_H,
    const int4 *__restrict__ d_cell_i_j_start_end, const int4 *__restrict__ d_cell_i_j_start_end_non_compact,
	const int *__restrict__ d_block_leaf_id, const int bundle_n_cells,
    const double3 space_dim) {

  const int threadid = blockDim.x * blockIdx.x + threadIdx.x;
  const int bid = blockIdx.x;
  //cid for now corresponds to leaf computation id
  const int cid = threadid;

  /*Necessary to prevent out of bounds access
   * for thread ids above what is necessary to
   * complete computations*/
  if(cid < bundle_n_cells){
    /* First, grab handles for where cells start and end */
    int4 cell_starts_ends_read = d_cell_i_j_start_end[cid];
    int4 cell_starts_ends_write = d_cell_i_j_start_end_non_compact[cid];
    //Find the block id for the first block to work with this leaf computation
    const int leaf_bid_0 = d_block_leaf_id[cid];
    //Find which block this is within the list of blocks acting on leaf computation cid
    const int b_id_local = bid - leaf_bid_0;
    //Now find the particle this thread needs to work on
    const int p_id = b_id_local * GPU_THREAD_BLOCK_SIZE + threadIdx.x;
    /*Assign without accounting for ci/cj_end being the index of cell positions.
     * This will be accounted for in cuda_kernel_density()*/
    const int ci_start = cell_starts_ends_read.x;
    const int ci_end = cell_starts_ends_read.y;
    const int cj_start = cell_starts_ends_read.z;
    const int cj_end = cell_starts_ends_read.w;

//    const int ci_write_start = cell_starts_ends_write.x;
//    const int ci_write_end = cell_starts_ends_write.y;
    const int cj_write_start = cell_starts_ends_write.z;
    const int cj_write_end = cell_starts_ends_write.w;

    /*First things first. Find the shifts in case we're periodic*/
    /*Step I: Get the cell positions*/
    if(ci_end <= 0 || cj_end <= 0){
      printf("indices smaller than zero ci_end %i cj_end %i\n", ci_end, cj_end);
    }
    const struct gpu_cell_pos_d ci_loc = d_parts_send[ci_end - 1].c_loc;
    const struct gpu_cell_pos_d cj_loc = d_parts_send[cj_end - 1].c_loc;

    double3 shift = {0.0, 0.0, 0.0};

    const double distx = cj_loc.x.x - ci_loc.x.x;
    const double disty = cj_loc.x.y - ci_loc.x.y;
    const double distz = cj_loc.x.z - ci_loc.x.z;

    /*Fine for now as thread divergence
     * will be one (three) line of code*/
    if(distx < -space_dim.x * 0.5)
      shift.x = space_dim.x;
    else if(distx > space_dim.x * 0.5)
      shift.x = -space_dim.x;

    if(disty < -space_dim.y * 0.5)
      shift.y = space_dim.y;
    else if(disty > space_dim.y * 0.5)
      shift.y = -space_dim.y;

    if(distz < -space_dim.z * 0.5)
      shift.z = space_dim.z;
    else if (distz > space_dim.z * 0.5)
      shift.z = -space_dim.z;

    const double cell_dist = sqrt(distx*distx + disty*disty + distz*distz);

    /*Note: Since shift would be zero for self tasks where ci==cj
     * this is fine as is to avoid divergence*/
    const double shift_ix = shift.x + cj_loc.x.x;
    const double shift_iy = shift.y + cj_loc.x.y;
    const double shift_iz = shift.z + cj_loc.x.z;


    //  const double shift_ix = ci_loc.x.x;
    //  const double shift_iy = ci_loc.x.y;
    //  const double shift_iz = ci_loc.x.z;

    const double3 shift_i_res = {shift_ix, shift_iy, shift_iz};
    const double3 shift_j_res = {cj_loc.x.x, cj_loc.x.y, cj_loc.x.z};
    //  if (cid < bundle_n_cells) {
    /*Interact ci with cj*/
    //    printf("Doing ci\n");
    cuda_kernel_density(cid, d_parts_send, d_parts_recv, d_a, d_H,
        cell_starts_ends_read, cell_starts_ends_write,
        space_dim, shift_i_res, shift_j_res);
    /*Check if this is a self interaction.
     * If it is, skip as we don't need to re-do computations
     * We could just let threads do this again to avoid
     * divergence since we only unpack ci once on host
     * (for it's ci not the dummy cj used for indexing)*/
    if(ci_start != cj_start){
      //      printf("Doing cj\n");
      /*We've done ci with cj, now interact cj with ci*/
      cell_starts_ends_read.x = cj_start;
      cell_starts_ends_read.y = cj_end;
      cell_starts_ends_read.z = ci_start;
      cell_starts_ends_read.w = ci_end;
      cell_starts_ends_write.x = cj_write_start;
      cell_starts_ends_write.y = cj_write_end;
      //TODO: Un-necesary as we only need to write to cj but leave for now///
      //      cell_starts_ends_write.z = ci_start;
      //      cell_starts_ends_write.w = ci_end;
      const double3 shift_ii_res = {cj_loc.x.x, cj_loc.x.y, cj_loc.x.z};
      const double3 shift_jj_res = {shift_ix, shift_iy, shift_iz};
      ///////////////////////////////////////////////////////////////////////
      cuda_kernel_density(cid, d_parts_send, d_parts_recv, d_a, d_H,
          cell_starts_ends_read, cell_starts_ends_write,
          space_dim, shift_ii_res, shift_jj_res);
    }
  }
//  }
}

/**
 * @brief Call the particle SPH gradient kernel.
 *
 * @param d_parts_send array on device containing particle data
 * @param d_parts_recv array on device to write results into
 * @param d_a current cosmological scale factor
 * @param d_H current Hubble constant
 * @param bundle_first_part index of first particle of this bundle in the
 * d_parts_* arrays
 * @param bundle_n_parts nr of particles in this bundle
 */
__global__ void cuda_launch_gradient(
    const struct gpu_part_send_g *__restrict__ d_parts_send,
    struct gpu_part_recv_g *__restrict__ d_parts_recv, float d_a, float d_H,
    int bundle_first_part, int bundle_n_parts) {

  const int threadid = blockDim.x * blockIdx.x + threadIdx.x;
  const int pid = bundle_first_part + threadid;

  if (pid < bundle_first_part + bundle_n_parts) {
    cuda_kernel_gradient(pid, d_parts_send, d_parts_recv, d_a, d_H);
  }
}

/**
 * @brief Call the particle SPH force kernel.
 *
 * @param d_parts_send array on device containing particle data
 * @param d_parts_recv array on device to write results into
 * @param d_a current cosmological scale factor
 * @param d_H current Hubble constant
 * @param bundle_first_part index of first particle of this bundle in the
 * d_parts_* arrays
 * @param bundle_n_parts nr of particles in this bundle
 */
__global__ void cuda_launch_force(
    const struct gpu_part_send_f *__restrict__ d_parts_send,
    struct gpu_part_recv_f *__restrict__ d_parts_recv, float d_a, float d_H,
    int bundle_first_part, int bundle_n_parts) {

  const int threadid = blockDim.x * blockIdx.x + threadIdx.x;
  const int pid = bundle_first_part + threadid;

  if (pid < bundle_first_part + bundle_n_parts) {
    cuda_kernel_force(pid, d_parts_send, d_parts_recv, d_a, d_H);
  }
}

/**
 * @brief Launch the density computation on the GPU for a bundle of leaf cells.
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
void gpu_launch_density(const struct gpu_part_send_d *__restrict__ d_parts_send,
                        struct gpu_part_recv_d *__restrict__ d_parts_recv,
                        const float d_a, const float d_H,
                        const int num_blocks_x,
                        const int4 *__restrict__ d_cell_i_j_start_end,
                        const int4 *__restrict__ d_cell_i_j_start_end_non_compact,
                        const int *__restrict__ d_block_leaf_id,
                        const int bundle_n_cells, const double3 space_dim) {

  /* TODO: Do we want to allocate shared memory here? */
  cuda_launch_density<<<num_blocks_x, GPU_THREAD_BLOCK_SIZE, 0>>>(
      d_parts_send, d_parts_recv, d_a, d_H,
      d_cell_i_j_start_end, d_cell_i_j_start_end_non_compact,
	  d_block_leaf_id, bundle_n_cells, space_dim);
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
    const struct gpu_part_send_g *__restrict__ d_parts_send,
    struct gpu_part_recv_g *__restrict__ d_parts_recv, const float d_a,
    const float d_H, cudaStream_t stream, const int num_blocks_x,
    const int num_blocks_y, const int bundle_first_part,
    const int bundle_n_parts) {

  /* TODO: Do we want to allocate shared memory here? */
  cuda_launch_gradient<<<num_blocks_x, GPU_THREAD_BLOCK_SIZE, 0, stream>>>(
      d_parts_send, d_parts_recv, d_a, d_H, bundle_first_part, bundle_n_parts);
}

/**
 * @brief Launch the force computation on the GPU for a bundle of leaf cells.
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
void gpu_launch_force(const struct gpu_part_send_f *__restrict__ d_parts_send,
                      struct gpu_part_recv_f *__restrict__ d_parts_recv,
                      const float d_a, const float d_H, cudaStream_t stream,
                      const int num_blocks_x, const int num_blocks_y,
                      const int bundle_first_part, const int bundle_n_parts) {

  /* TODO: Do we want to allocate shared memory here? */
  cuda_launch_force<<<num_blocks_x, GPU_THREAD_BLOCK_SIZE, 0, stream>>>(
      d_parts_send, d_parts_recv, d_a, d_H, bundle_first_part, bundle_n_parts);
}

#ifdef __cplusplus
}
#endif
