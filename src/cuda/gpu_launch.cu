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
#include <config.h>


#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include "cuda_config.h"
#include "cuda_particle_kernels.cuh"
#include "gpu_launch.h"
#include "gpu_part_structs.h"


/**
 * Launch the pair density computations on the GPU
 */
__global__ void cuda_launch_pair_density(
    struct gpu_part_send_d *parts_send, struct gpu_part_recv_d *parts_recv,
    float d_a, float d_H, int bundle_first_part, int bundle_n_parts) {

  const int threadid = blockDim.x * blockIdx.x + threadIdx.x;
  const int pid = bundle_first_part + threadid;

  if (pid < bundle_first_part + bundle_n_parts) {
    const struct gpu_part_send_d pi = parts_send[pid];
    const int cj_start = pi.cjs_cje.x;
    const int cj_end = pi.cjs_cje.y;

    /* Start calculations for particles in cell i*/
    cuda_kernel_pair_density(pi, parts_send, parts_recv, pid, cj_start, cj_end, d_a, d_H);
  }
}

/**
 * Launch the pair gradient computations on the GPU
 */
__global__ void cuda_launch_pair_gradient(
    struct gpu_part_send_g *parts_send,
    struct gpu_part_recv_g *parts_recv, float d_a, float d_H,
    int bundle_first_part, int bundle_n_parts) {

  const int threadid = blockDim.x * blockIdx.x + threadIdx.x;
  const int pid = bundle_first_part + threadid;

  if (pid < bundle_first_part + bundle_n_parts) {
    const struct gpu_part_send_g pi = parts_send[pid];
    const int cj_start = pi.cjs_cje.x;
    const int cj_end = pi.cjs_cje.y;

    /* Start calculations for particles in cell i*/
    cuda_kernel_pair_gradient(pi, parts_send, parts_recv, pid, cj_start, cj_end,
                          d_a, d_H);
  }
}

/**
 * Launch the pair force computations on the GPU
 */
__global__ void cuda_launch_pair_force(
    struct gpu_part_send_f *parts_send,
    struct gpu_part_recv_f *parts_recv, float d_a, float d_H,
    int bundle_first_part, int bundle_n_parts) {

  const int threadid = blockDim.x * blockIdx.x + threadIdx.x;
  const int pid = bundle_first_part + threadid;

  if (pid < bundle_first_part + bundle_n_parts) {
    const struct gpu_part_send_f pi = parts_send[pid];
    const int cj_start = pi.cjs_cje.x;
    const int cj_end = pi.cjs_cje.y;

    /* Start calculations for particles in cell i */
    cuda_kernel_pair_force(pi, parts_send, parts_recv, pid, cj_start, cj_end,
                          d_a, d_H);
  }
}

/**
 * Launch the pair density computation on the GPU.
 */
void gpu_launch_pair_density(
    struct gpu_part_send_d *parts_send, struct gpu_part_recv_d *parts_recv,
    float d_a, float d_H, cudaStream_t stream, int numBlocks_x, int numBlocks_y,
    int bundle_first_part, int bundle_n_parts) {

  dim3 gridShape = dim3(numBlocks_x, numBlocks_y);

  cuda_launch_pair_density<<<numBlocks_x, BLOCK_SIZE, 0, stream>>>(
      parts_send, parts_recv, d_a, d_H, bundle_first_part, bundle_n_parts);
}

/**
 * Launch the pair gradient computation on the GPU.
 */
void gpu_launch_pair_gradient(
    struct gpu_part_send_g *parts_send,
    struct gpu_part_recv_g *parts_recv, float d_a, float d_H,
    cudaStream_t stream, int numBlocks_x, int numBlocks_y,
    int bundle_first_part, int bundle_n_parts) {

  dim3 gridShape = dim3(numBlocks_x, numBlocks_y);

  cuda_launch_pair_gradient<<<numBlocks_x, BLOCK_SIZE, 0, stream>>>(
      parts_send, parts_recv, d_a, d_H, bundle_first_part, bundle_n_parts);
}

/**
 * Launch the pair force computation on the GPU.
 */
void gpu_launch_pair_force(
    struct gpu_part_send_f *parts_send,
    struct gpu_part_recv_f *parts_recv, float d_a, float d_H,
    cudaStream_t stream, int numBlocks_x, int numBlocks_y,
    int bundle_first_part, int bundle_n_parts) {

  dim3 gridShape = dim3(numBlocks_x, numBlocks_y);

  cuda_launch_pair_force<<<numBlocks_x, BLOCK_SIZE, 0, stream>>>(
      parts_send, parts_recv, d_a, d_H, bundle_first_part, bundle_n_parts);
}

/**
 * Launch the self density computation on the GPU.
 */
void gpu_launch_self_density(struct gpu_part_send_d *parts_send,
                           struct gpu_part_recv_d *parts_recv, float d_a,
                           float d_H, cudaStream_t stream, int numBlocks_x,
                           int numBlocks_y, int bundle_first_task,
                           int2 *d_task_first_part_f4) {

  dim3 gridShape = dim3(numBlocks_x, numBlocks_y);
  cuda_kernel_self_density<<<gridShape, BLOCK_SIZE, 2ul * BLOCK_SIZE * sizeof(float4),
                      stream>>>(parts_send, parts_recv, d_a, d_H,
                                bundle_first_task, d_task_first_part_f4);
}


/**
 * Launch the self gradient computation on the GPU.
 */
void gpu_launch_self_gradient(struct gpu_part_send_g *parts_send,
                            struct gpu_part_recv_g *parts_recv, float d_a,
                            float d_H, cudaStream_t stream, int numBlocks_x,
                            int numBlocks_y, int bundle_first_task,
                            int2 *d_task_first_part_f4) {

  dim3 gridShape = dim3(numBlocks_x, numBlocks_y);
  cuda_kernel_self_gradient<<<gridShape, BLOCK_SIZE, 3ul * BLOCK_SIZE * sizeof(float4), stream>>>(parts_send, parts_recv, d_a, d_H,
        bundle_first_task, d_task_first_part_f4); }

/**
 * Launch the self force computation on the GPU.
 */
void gpu_launch_self_force(struct gpu_part_send_f *d_parts_send,
                         struct gpu_part_recv_f *d_parts_recv, float d_a,
                         float d_H, cudaStream_t stream, int numBlocks_x,
                         int numBlocks_y, int bundle_first_task,
                         int2 *d_task_first_part_f4) {

  dim3 gridShape = dim3(numBlocks_x, numBlocks_y);
  cuda_kernel_self_force<<<
      gridShape, BLOCK_SIZE,
      4ul * BLOCK_SIZE * sizeof(float4) + BLOCK_SIZE * sizeof(float3), stream>>>(
      d_parts_send, d_parts_recv, d_a, d_H, bundle_first_task,
      d_task_first_part_f4);
}

#ifdef __cplusplus
}
#endif
