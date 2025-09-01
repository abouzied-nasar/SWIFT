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

#include <cuda_runtime.h>

void gpu_launch_self_density(struct gpu_part_send_d *parts_send,
                             struct gpu_part_recv_d *parts_recv, float d_a,
                             float d_H, cudaStream_t stream, int numBlocks_x,
                             int numBlocks_y, int bundle_first_task,
                             int2 *d_task_first_part_f4);
void gpu_launch_self_gradient(struct gpu_part_send_g *parts_send,
                              struct gpu_part_recv_g *parts_recv, float d_a,
                              float d_H, cudaStream_t stream, int numBlocks_x,
                              int numBlocks_y, int bundle_first_task,
                              int2 *d_task_first_part_f4);
void gpu_launch_self_force(struct gpu_part_send_f *parts_send,
                           struct gpu_part_recv_f *parts_recv, float d_a,
                           float d_H, cudaStream_t stream, int numBlocks_x,
                           int numBlocks_y, int bundle_first_task,
                           int2 *d_task_first_part_f4);
void gpu_launch_pair_density(struct gpu_part_send_d *parts_send,
                             struct gpu_part_recv_d *parts_recv, float d_a,
                             float d_H, cudaStream_t stream, int numBlocks_x,
                             int numBlocks_y, int bundle_first_part,
                             int bundle_n_parts);
void gpu_launch_pair_gradient(struct gpu_part_send_g *parts_send,
                              struct gpu_part_recv_g *parts_recv, float d_a,
                              float d_H, cudaStream_t stream, int numBlocks_x,
                              int numBlocks_y, int bundle_first_part,
                              int bundle_n_parts);
void gpu_launch_pair_force(struct gpu_part_send_f *parts_send,
                           struct gpu_part_recv_f *parts_recv, float d_a,
                           float d_H, cudaStream_t stream, int numBlocks_x,
                           int numBlocks_y, int bundle_first_part,
                           int bundle_n_parts);
#ifdef __cplusplus
}
#endif

#endif  // CUDA_GPU_LAUNCH_H
