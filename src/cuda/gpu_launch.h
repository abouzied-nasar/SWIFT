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

void gpu_launch_density(const struct gpu_part_send_d *restrict d_parts_send,
                        struct gpu_part_recv_d *restrict d_parts_recv,
                        const float d_a, const float d_H, cudaStream_t stream,
                        const int num_blocks_x, const int num_blocks_y,
                        const int bundle_first_part, const int bundle_n_parts,
                        const int4 *restrict d_cell_i_j_start_end,
                        const int4 *restrict d_cell_i_j_start_end_non_compact,
                        const double3 *restrict d_cell_positions, const int bundle_first_cell,
                        const int bundle_n_cells);
void gpu_launch_gradient(const struct gpu_part_send_g *restrict d_parts_send,
                         struct gpu_part_recv_g *restrict d_parts_recv,
                         const float d_a, const float d_H, cudaStream_t stream,
                         const int num_blocks_x, const int num_blocks_y,
                         const int bundle_first_part, const int bundle_n_parts);
void gpu_launch_force(const struct gpu_part_send_f *restrict d_parts_send,
                      struct gpu_part_recv_f *restrict d_parts_recv,
                      const float d_a, const float d_H, cudaStream_t stream,
                      const int num_blocks_x, const int num_blocks_y,
                      const int bundle_first_part, const int bundle_n_parts);

#ifdef __cplusplus
}
#endif

#endif  // CUDA_GPU_LAUNCH_H
