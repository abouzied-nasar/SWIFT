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
#ifndef RUNNER_GPU_PACK_FUNCTIONS_H
#define RUNNER_GPU_PACK_FUNCTIONS_H

#include "../config.h"
#include "runner.h"
#include "timeline.h"

/* Temporary warning during dev works. */
#if !(defined(HAVE_CUDA) || defined(HAVE_HIP))
#pragma warning "Don't have CUDA nor HIP"
#endif

#ifdef WITH_CUDA
#include "cuda/gpu_offload_data.h"
#include "cuda/gpu_part_structs.h"
#endif

#ifdef WITH_HIP
#pragma message "YES"
#include "hip/gpu_part_structs.h"
#endif

void gpu_pack_part_self_density(
    const struct cell* restrict c,
    struct gpu_offload_data *buf);

void gpu_pack_part_self_gradient(
    const struct cell* restrict c,
    struct gpu_offload_data *buf);

void gpu_pack_part_self_force(
    const struct cell* restrict c,
    struct gpu_offload_data *buf);

void gpu_unpack_part_self_density(struct cell* restrict c,
    const struct gpu_part_recv_d* restrict parts_aos_buffer,
    const size_t pack_position,
    const size_t count, const struct engine *e);

void gpu_unpack_part_self_gradient(struct cell* restrict c,
    const struct gpu_part_recv_g* restrict parts_aos_buffer,
    const size_t pack_position,
    const size_t count, const struct engine *e);

void gpu_unpack_part_self_force(struct cell* restrict c,
    const struct gpu_part_recv_f* restrict parts_aos_buffer,
    const size_t pack_position,
    const size_t count, const struct engine *e);

void gpu_pack_part_pair_density(
    const struct cell *c, struct gpu_part_send_d *parts_aos_buffer,
    const int local_pack_position,
    const int count, const double3 shift, const int2 cstarts);

void gpu_pack_part_pair_gradient(
    const struct cell *c, struct gpu_part_send_g *parts_aos_buffer,
    const int local_pack_position,
    const int count, const double3 shift, const int2 cstarts);

void gpu_pack_part_pair_force(
    const struct cell *c, struct gpu_part_send_f *parts_aos_buffer,
    const int local_pack_position,
    const int count, const double3 shift, const int2 cstarts);

void gpu_unpack_part_pair_density(
    struct cell *c,
    const struct gpu_part_recv_d *parts_aos_buffer,
    const size_t pack_ind,
    const size_t count);

void gpu_unpack_part_pair_gradient(
    struct cell *c,
    const struct gpu_part_recv_g *parts_aos_buffer,
    const size_t pack_ind,
    const size_t count);

void gpu_unpack_part_pair_force(
    struct cell *c,
    const struct gpu_part_recv_f *parts_aos_buffer,
    const size_t pack_ind,
    const size_t count);

void gpu_pack_pair_density(struct gpu_offload_data* buf,
    const struct runner *r, const struct cell *ci, const struct cell *cj,
    const double3 shift_tmp);

void gpu_pack_pair_gradient(struct gpu_offload_data* buf,
    const struct runner *r, const struct cell *ci, const struct cell *cj,
    const double3 shift_tmp);

void gpu_pack_pair_force(struct gpu_offload_data* buf,
    const struct runner *r, const struct cell *ci, const struct cell *cj,
    const double3 shift_tmp);

void gpu_unpack_pair_density(
    const struct runner *r,
    struct cell *ci,
    struct cell *cj,
    const struct gpu_part_recv_d *parts_aos_buffer,
    size_t *pack_ind,
    size_t count_max_parts
    );

void gpu_unpack_pair_gradient(
    const struct runner *r,
    struct cell *ci,
    struct cell *cj,
    const struct gpu_part_recv_g *parts_aos_buffer,
    size_t *pack_ind,
    size_t count_max_parts
    );

void gpu_unpack_pair_force(
    const struct runner *r,
    struct cell *ci,
    struct cell *cj,
    const struct gpu_part_recv_f *parts_aos_buffer,
    size_t *pack_ind,
    size_t count_max_parts
    );


#endif /* RUNNER_GPU_PACK_FUNCTIONS_H */
