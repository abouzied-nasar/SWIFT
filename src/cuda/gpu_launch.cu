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

/*TODO: Figure out why this only works when included from here.
 * When included in cuda_particle_kernels.cuh compiler complains with
 * error: this declaration may not have extern "C" linkage */
#include <cuda_pipeline.h>

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
void gpu_launch_density(
    const struct gpu_part_send_d* __restrict__ d_parts_send,
    struct gpu_part_recv_d* __restrict__ d_parts_recv,
    const float d_a, const float d_H,
    int num_blocks_x,
    const int4* __restrict__ d_cell_i_j_start_end,
    const int2* __restrict__ d_block_leaf_id,
    const double3 space_dim,
    cudaStream_t stream){

	static_assert(
	    GPU_THREAD_BLOCK_SIZE > 0 &&
	    (GPU_THREAD_BLOCK_SIZE &
	     (GPU_THREAD_BLOCK_SIZE - 1)) == 0,
	    "GPU_THREAD_BLOCK_SIZE must be a power of two");
  /* Shared memory allocation. Need two tiles as another tile (1) is
   used for prefetching while tile 0 is used for computations and vice-versa*/
//  const size_t sh_mem = 2 * GPU_THREAD_BLOCK_SIZE * (sizeof(struct gpu_part_data_d));//(sizeof(float4) + sizeof(float4)); // 2048 bytes when TILE_J=64

  const size_t density_tiling_shared_bytes =
      2 * GPU_THREAD_BLOCK_SIZE * (sizeof(struct gpu_part_data_d));
  const size_t density_reduction_shared_bytes = GPU_THREAD_BLOCK_SIZE * sizeof(density_block_partial);

  const size_t sh_mem = density_tiling_shared_bytes > density_reduction_shared_bytes ?
		  density_tiling_shared_bytes : density_reduction_shared_bytes;

  cuda_kernel_density<<<num_blocks_x, GPU_THREAD_BLOCK_SIZE, sh_mem, stream>>>(
      d_parts_send, d_parts_recv, d_a, d_H,
      d_cell_i_j_start_end, d_block_leaf_id, space_dim);

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

  static_assert(
	    GPU_THREAD_BLOCK_SIZE > 0 &&
	    (GPU_THREAD_BLOCK_SIZE &
	     (GPU_THREAD_BLOCK_SIZE - 1)) == 0,
	    "GPU_THREAD_BLOCK_SIZE must be a power of two");

  const size_t gradient_tiling_shared_bytes = 2 * GPU_THREAD_BLOCK_SIZE * sizeof(gpu_part_data_g);
  const size_t gradient_reduction_shared_bytes =
      GPU_THREAD_BLOCK_SIZE *
      sizeof(gradient_block_partial);

  const size_t sh_mem =
      gradient_tiling_shared_bytes >
      gradient_reduction_shared_bytes
          ? gradient_tiling_shared_bytes
          : gradient_reduction_shared_bytes;
	/* Shared memory allocation. Need two tiles as another tile (1) is
     used for prefetching while tile 0 is used for computations and vice-versa*/
//    const size_t sh_mem = 2 * GPU_THREAD_BLOCK_SIZE * sizeof(struct gpu_part_data_g);  // 3072 bytes when TILE_J=64

    cuda_kernel_gradient<<<num_blocks_x, GPU_THREAD_BLOCK_SIZE, sh_mem, stream>>>(
        d_parts_send, d_parts_recv, d_a, d_H,
        d_cell_i_j_start_end, d_block_leaf_id, space_dim);
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

	static_assert(
	    (GPU_THREAD_BLOCK_SIZE &
	     (GPU_THREAD_BLOCK_SIZE - 1)) == 0,
	    "GPU_THREAD_BLOCK_SIZE must be a power of two");
	/* Shared memory allocation. Need two tiles as another tile (1) is
	     used for prefetching while tile 0 is used for computations and vice-versa*/
	const size_t original_shared_bytes =
	    2 * GPU_THREAD_BLOCK_SIZE * sizeof(struct gpu_part_data_f);
//	    (
//	        sizeof(double2) +
//	        sizeof(double2) +
//	        sizeof(float4)  +
//	        sizeof(float4)  +
//	        sizeof(float4)  +
//	        sizeof(int2)
//	    );

	const size_t reduction_shared_bytes =
	    GPU_THREAD_BLOCK_SIZE *
	    sizeof(force_block_partial);

	const size_t shmem =
	    max(original_shared_bytes, reduction_shared_bytes);

//    const size_t shmem = 2 * GPU_THREAD_BLOCK_SIZE * sizeof(struct gpu_part_data_f);

    cuda_kernel_force<<<num_blocks_x, GPU_THREAD_BLOCK_SIZE, shmem, stream>>>(
        d_parts_send, d_parts_recv, d_a, d_H,
        d_cell_i_j_start_end, d_block_leaf_id, space_dim);
}

#ifdef __cplusplus
}
#endif
