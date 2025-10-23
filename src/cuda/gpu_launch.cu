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
#include "gpu_part_structs.h"

#include <config.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

/**
 * Launch the density computations on the GPU
 * TODO: Parameter documentation
 */
__global__ void cuda_launch_density(
    const struct gpu_part_send_d *restrict d_parts_send,
    struct gpu_part_recv_d *restrict d_parts_recv, const float d_a,
    const float d_H, const int bundle_first_part, const int bundle_n_parts) {

  const int threadid = blockDim.x * blockIdx.x + threadIdx.x;
  const int pid = bundle_first_part + threadid;

  if (pid < bundle_first_part + bundle_n_parts) {
    const struct gpu_part_send_d pi = d_parts_send[pid];
    const int cj_start = pi.cjs_cje.x;
    const int cj_end = pi.cjs_cje.y;

    /* Start calculations for particles in cell i */
    cuda_kernel_density(pi, d_parts_send, d_parts_recv, pid, cj_start,
                             cj_end, d_a, d_H);
  }
}

/**
 * Launch the pair gradient computations on the GPU
 * TODO: Parameter documentation
 */
__global__ void cuda_launch_pair_gradient(
    const struct gpu_part_send_g *restrict d_parts_send,
    struct gpu_part_recv_g *restrict d_parts_recv, float d_a, float d_H,
    int bundle_first_part, int bundle_n_parts) {

  const int threadid = blockDim.x * blockIdx.x + threadIdx.x;
  const int pid = bundle_first_part + threadid;

  if (pid < bundle_first_part + bundle_n_parts) {
    const struct gpu_part_send_g pi = d_parts_send[pid];
    const int cj_start = pi.cjs_cje.x;
    const int cj_end = pi.cjs_cje.y;

    /* Start calculations for particles in cell i*/
    cuda_kernel_pair_gradient(pi, d_parts_send, d_parts_recv, pid, cj_start,
                              cj_end, d_a, d_H);
  }
}

/**
 * Launch the pair force computations on the GPU
 * TODO: Parameter documentation
 */
__global__ void cuda_launch_pair_force(
    const struct gpu_part_send_f *restrict d_parts_send,
    struct gpu_part_recv_f *restrict d_parts_recv, float d_a, float d_H,
    int bundle_first_part, int bundle_n_parts) {

  const int threadid = blockDim.x * blockIdx.x + threadIdx.x;
  const int pid = bundle_first_part + threadid;

  if (pid < bundle_first_part + bundle_n_parts) {
    const struct gpu_part_send_f pi = d_parts_send[pid];
    const int cj_start = pi.cjs_cje.x;
    const int cj_end = pi.cjs_cje.y;

    /* Start calculations for particles in cell i */
    cuda_kernel_pair_force(pi, d_parts_send, d_parts_recv, pid, cj_start,
                           cj_end, d_a, d_H);
  }
}

/**
 * @brief Launch the density computation on the GPU.
 * TODO: Parameter documentation
 */
void gpu_launch_density(
    const struct gpu_part_send_d *restrict d_parts_send,
    struct gpu_part_recv_d *restrict d_parts_recv, const float d_a,
    const float d_H, cudaStream_t stream, const int num_blocks_x,
    const int num_blocks_y, const int bundle_first_part,
    const int bundle_n_parts) {

  cuda_launch_density<<<num_blocks_x, GPU_THREAD_BLOCK_SIZE, 0, stream>>>(
      d_parts_send, d_parts_recv, d_a, d_H, bundle_first_part, bundle_n_parts);
}

/**
 * @brief Launch the pair gradient computation on the GPU.
 */
void gpu_launch_pair_gradient(
    const struct gpu_part_send_g *restrict d_parts_send,
    struct gpu_part_recv_g *restrict d_parts_recv, const float d_a,
    const float d_H, cudaStream_t stream, const int num_blocks_x,
    const int num_blocks_y, const int bundle_first_part,
    const int bundle_n_parts) {

  cuda_launch_pair_gradient<<<num_blocks_x, GPU_THREAD_BLOCK_SIZE, 0, stream>>>(
      d_parts_send, d_parts_recv, d_a, d_H, bundle_first_part, bundle_n_parts);
}

/**
 * @brief Launch the pair force computation on the GPU.
 * TODO: Parameter documentation
 */
void gpu_launch_pair_force(const struct gpu_part_send_f *restrict d_parts_send,
                           struct gpu_part_recv_f *restrict d_parts_recv,
                           const float d_a, const float d_H,
                           cudaStream_t stream, const int num_blocks_x,
                           const int num_blocks_y, const int bundle_first_part,
                           const int bundle_n_parts) {

  cuda_launch_pair_force<<<num_blocks_x, GPU_THREAD_BLOCK_SIZE, 0, stream>>>(
      d_parts_send, d_parts_recv, d_a, d_H, bundle_first_part, bundle_n_parts);
}

/**
 * @brief Launch the self density computation on the GPU.
 * TODO: Parameter documentation
 */
void gpu_launch_self_density(
    const struct gpu_part_send_d *restrict d_parts_send,
    struct gpu_part_recv_d *restrict d_parts_recv, const float d_a,
    const float d_H, cudaStream_t stream, const int num_blocks_x,
    const int num_blocks_y, const int bundle_first_task,
    int2 *d_task_first_part_f4) {

  const dim3 gridShape = dim3(num_blocks_x, num_blocks_y);
  /* TODO: WHY IS THERE A FACTOR OF 2 HERE? Please document. */
  /* Would it not be better to use sizeof(struct parts_send) */
  const size_t bsize = GPU_THREAD_BLOCK_SIZE;
  const size_t shmem_size = 2ul * bsize * sizeof(float4);
  cuda_kernel_self_density<<<gridShape, bsize, shmem_size, stream>>>(
      d_parts_send, d_parts_recv, d_a, d_H, bundle_first_task,
      d_task_first_part_f4);
}

/**
 * @brief Launch the self gradient computation on the GPU.
 */
void gpu_launch_self_gradient(
    const struct gpu_part_send_g *restrict d_parts_send,
    struct gpu_part_recv_g *restrict d_parts_recv, const float d_a,
    const float d_H, cudaStream_t stream, const int num_blocks_x,
    const int num_blocks_y, const int bundle_first_task,
    int2 *d_task_first_part_f4) {

  const dim3 gridShape = dim3(num_blocks_x, num_blocks_y);
  const size_t bsize = GPU_THREAD_BLOCK_SIZE;
  const size_t shmem_size = 3ul * bsize * sizeof(float4);
  cuda_kernel_self_gradient<<<gridShape, bsize, shmem_size, stream>>>(
      d_parts_send, d_parts_recv, d_a, d_H, bundle_first_task,
      d_task_first_part_f4);
}

/**
 * @brief Launch the self force computation on the GPU.
 */
void gpu_launch_self_force(const struct gpu_part_send_f *restrict d_parts_send,
                           struct gpu_part_recv_f *restrict d_parts_recv,
                           const float d_a, const float d_H,
                           cudaStream_t stream, const int num_blocks_x,
                           const int num_blocks_y, const int bundle_first_task,
                           int2 *d_task_first_part_f4) {

  const dim3 gridShape = dim3(num_blocks_x, num_blocks_y);
  const size_t bsize = GPU_THREAD_BLOCK_SIZE;
  const size_t shmem_size =
      4ul * bsize * sizeof(float4) + bsize * sizeof(float3);
  cuda_kernel_self_force<<<gridShape, bsize, shmem_size, stream>>>(
      d_parts_send, d_parts_recv, d_a, d_H, bundle_first_task,
      d_task_first_part_f4);
}

#ifdef __cplusplus
}
#endif
