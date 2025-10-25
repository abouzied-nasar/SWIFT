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

/* TODO: This header should be outside the src/cuda directory, and should be
 * cuda independent. The base file, src/cuda/gpu_launch.cu, should be the only
 * thing containing cuda code. This header needs to be shared between all
 * implementations */

#ifdef __cplusplus
extern "C" {
#endif
#include "gpu_part_structs.h"

#include <cuda_runtime.h>

void gpu_launch_self_gradient(
    const struct gpu_part_send_g *restrict d_parts_send,
    struct gpu_part_recv_g *restrict d_parts_recv, const float d_a,
    const float d_H, cudaStream_t stream, const int num_blocks_x,
    const int num_blocks_y, const int bundle_first_task,
    int2 *d_task_first_part_f4);
void gpu_launch_self_force(const struct gpu_part_send_f *restrict parts_send,
                           struct gpu_part_recv_f *restrict parts_recv,
                           const float d_a, const float d_H,
                           cudaStream_t stream, const int num_blocks_x,
                           const int num_blocks_y, const int bundle_first_task,
                           int2 *d_task_first_part_f4);
void gpu_launch_density(const struct gpu_part_send_d *restrict d_parts_send,
                        struct gpu_part_recv_d *restrict d_parts_recv,
                        const float d_a, const float d_H, cudaStream_t stream,
                        const int num_blocks_x, const int num_blocks_y,
                        const int bundle_first_part, const int bundle_n_parts);
void gpu_launch_pair_gradient(
    const struct gpu_part_send_g *restrict d_parts_send,
    struct gpu_part_recv_g *restrict d_parts_recv, const float d_a,
    const float d_H, cudaStream_t stream, const int num_blocks_x,
    const int num_blocks_y, const int bundle_first_part,
    const int bundle_n_parts);
void gpu_launch_pair_force(const struct gpu_part_send_f *restrict d_parts_send,
                           struct gpu_part_recv_f *restrict d_parts_recv,
                           const float d_a, const float d_H,
                           cudaStream_t stream, const int num_blocks_x,
                           const int num_blocks_y, const int bundle_first_part,
                           const int bundle_n_parts);
#ifdef __cplusplus
}
#endif

#endif  // CUDA_GPU_LAUNCH_H
