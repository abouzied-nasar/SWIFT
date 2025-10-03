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
#include "active.h"
#include "engine.h"
#include "inline.h"
#include "runner.h"
#include "timers.h"

/* Temporary warning during dev works. */
#if !(defined(HAVE_CUDA) || defined(HAVE_HIP))
#pragma warning "Don't have CUDA nor HIP"
#endif

#ifdef WITH_CUDA
#include "cuda/gpu_offload_data.h"
#include "cuda/gpu_part_pack_functions.h"
#include "cuda/gpu_part_structs.h"
#endif

#ifdef WITH_HIP
#pragma message "YES"
#include "hip/gpu_part_structs.h"
#endif

/**
 * @brief packs particle data for density tasks into CPU-side buffers for self
 * tasks.
 * Currently only a wrapper around gpu_pack_part_self_density, but we'll need
 * to distinguish between SPH flavours in the future here by including the
 * correct corresponding header file.
 */
__attribute__((always_inline)) INLINE static void gpu_pack_self_density(
    const struct cell *restrict c, struct gpu_offload_data *restrict buf) {

  gpu_pack_part_self_density(c, buf->parts_send_d, buf->pv.count_parts);
}

/**
 * @brief packs particle data for gradient tasks into CPU-side buffers for self
 * tasks
 * Currently only a wrapper around gpu_pack_part_self_gradient, but we'll need
 * to distinguish between SPH flavours in the future here by including the
 * correct corresponding header file.
 */
__attribute__((always_inline)) INLINE static void gpu_pack_self_gradient(
    const struct cell *restrict c, struct gpu_offload_data *restrict buf) {

  gpu_pack_part_self_gradient(c, buf->parts_send_g, buf->pv.count_parts);
}

/**
 * @brief packs particle data for force tasks into CPU-side buffers for self
 * tasks
 * Currently only a wrapper around gpu_pack_part_self_force, but we'll need
 * to distinguish between SPH flavours in the future here by including the
 * correct corresponding header file.
 */
__attribute__((always_inline)) INLINE static void gpu_pack_self_force(
    const struct cell *restrict c, struct gpu_offload_data *restrict buf) {

  gpu_pack_part_self_force(c, buf->parts_send_f, buf->pv.count_parts);
}

/**
 * @brief Unpacks the density data from GPU buffers of self tasks into particles
 * Currently only a wrapper around gpu_unpack_part_self_density, but we'll need
 * to distinguish between SPH flavours in the future here by including the
 * correct corresponding header file.
 */
__attribute__((always_inline)) INLINE static void gpu_unpack_self_density(
    struct cell *restrict c,
    const struct gpu_part_recv_d *restrict parts_buffer,
    const int pack_position, const int count, const struct engine *e) {

  gpu_unpack_part_self_density(c, parts_buffer, pack_position, count, e);
}

/**
 * @brief Unpacks the gradient data from GPU buffers of self tasks into
 * particles Currently only a wrapper around gpu_unpack_part_self_gradient, but
 * we'll need to distinguish between SPH flavours in the future here by
 * including the correct corresponding header file.
 */
__attribute__((always_inline)) INLINE static void gpu_unpack_self_gradient(
    struct cell *restrict c,
    const struct gpu_part_recv_g *restrict parts_buffer,
    const int pack_position, const int count, const struct engine *e) {

  gpu_unpack_part_self_gradient(c, parts_buffer, pack_position, count, e);
}

/**
 * @brief Unpacks the force data from GPU buffers of self tasks into particles
 * Currently only a wrapper around gpu_unpack_part_self_force, but we'll need
 * to distinguish between SPH flavours in the future here by including the
 * correct corresponding header file.
 */
__attribute__((always_inline)) INLINE static void gpu_unpack_self_force(
    struct cell *restrict c,
    const struct gpu_part_recv_f *restrict parts_buffer,
    const int pack_position, const int count, const struct engine *e) {

  gpu_unpack_part_self_force(c, parts_buffer, pack_position, count, e);
}

/**
 * @brief unpacks particle data of two cells for the pair density GPU task from
 * the buffers
 */
__attribute__((always_inline)) INLINE static void gpu_unpack_pair_density(
    const struct runner *r, struct cell *ci, struct cell *cj,
    const struct gpu_part_recv_d *parts_aos_buffer, int *pack_ind,
    int count_max_parts) {

  const struct engine *e = r->e;

  if (!cell_is_active_hydro(ci, e) && !cell_is_active_hydro(cj, e)) {
    message("Inactive cell");
    return;
  }

  int count_ci = ci->hydro.count;
  int count_cj = cj->hydro.count;

#ifdef SWIFT_DEBUG_CHECKS
  if (*pack_ind + count_ci + count_cj >= count_max_parts) {
    error(
        "Exceeded count_max_parts_tmp. Make arrays bigger! pack_ind is "
        "%d, counts are %d %d, max is %d",
        *pack_ind, count_ci, count_cj, count_max_parts);
  }
#endif

  if (cell_is_active_hydro(ci, e)) {
    /* Pack the particle data into CPU-side buffers*/
    gpu_unpack_part_pair_density(ci, parts_aos_buffer, *pack_ind, count_ci);

    /* Increment packed index accordingly */
    *pack_ind += count_ci;
  }

  if (cell_is_active_hydro(cj, e)) {
    /* Pack the particle data into CPU-side buffers*/
    gpu_unpack_part_pair_density(cj, parts_aos_buffer, *pack_ind, count_cj);

    /* Increment packed index accordingly */
    *pack_ind += count_cj;
  }
}

/**
 * @brief unpacks particle data of two cells for the pair gradient GPU task from
 * the buffers
 */
__attribute__((always_inline)) INLINE static void gpu_unpack_pair_gradient(
    const struct runner *r, struct cell *ci, struct cell *cj,
    const struct gpu_part_recv_g *parts_aos_buffer, int *pack_ind,
    int count_max_parts) {

  const struct engine *e = r->e;

  /* Anything to do here? */
  if (!cell_is_active_hydro(ci, e) && !cell_is_active_hydro(cj, e)) {
    return;
  }

  int count_ci = ci->hydro.count;
  int count_cj = cj->hydro.count;

#ifdef SWIFT_DEBUG_CHECKS
  if (*pack_ind + count_ci + count_cj >= count_max_parts) {
    error(
        "Exceeded count_max_parts_tmp. Make arrays bigger! pack_ind is "
        "%d, counts are %d %d, max is %d",
        *pack_ind, count_ci, count_cj, count_max_parts);
  }
#endif

  if (cell_is_active_hydro(ci, e)) {
    /* Pack the particle data into CPU-side buffers*/
    gpu_unpack_part_pair_gradient(ci, parts_aos_buffer, *pack_ind, count_ci);

    /* Increment packed index accordingly */
    *pack_ind += count_ci;
  }

  if (cell_is_active_hydro(cj, e)) {
    /* Pack the particle data into CPU-side buffers*/
    gpu_unpack_part_pair_gradient(cj, parts_aos_buffer, *pack_ind, count_cj);

    /* Increment packed index accordingly */
    (*pack_ind) += count_cj;
  }
}

/**
 * @brief unpacks particle data of two cells for the pair force GPU task from
 * the buffers
 */
__attribute__((always_inline)) INLINE static void gpu_unpack_pair_force(
    const struct runner *r, struct cell *ci, struct cell *cj,
    const struct gpu_part_recv_f *parts_aos_buffer, int *pack_ind,
    int count_max_parts) {

  const struct engine *e = r->e;

  if (!cell_is_active_hydro(ci, e) && !cell_is_active_hydro(cj, e)) {
    return;
  }

  int count_ci = ci->hydro.count;
  int count_cj = cj->hydro.count;

#ifdef SWIFT_DEBUG_CHECKS
  if (*pack_ind + count_ci + count_cj >= count_max_parts) {
    error(
        "Exceeded count_max_parts_tmp. Make arrays bigger! pack_ind is "
        "%d, counts are %d %d, max is %d",
        *pack_ind, count_ci, count_cj, count_max_parts);
  }
#endif

  if (cell_is_active_hydro(ci, e)) {
    /* Pack the particle data into CPU-side buffers*/
    gpu_unpack_part_pair_force(ci, parts_aos_buffer, *pack_ind, count_ci);

    /* Increment packed index accordingly */
    *pack_ind += count_ci;
  }

  if (cell_is_active_hydro(cj, e)) {
    /* Pack the particle data into CPU-side buffers*/
    gpu_unpack_part_pair_force(cj, parts_aos_buffer, *pack_ind, count_cj);

    /* Increment pack length accordingly */
    *pack_ind += count_cj;
  }
}

/**
 * @brief packs up particle data of two cells for the pair density GPU task into
 * the buffers
 * @TODO: WHAT IS SHIFT_TMP?
 */
__attribute__((always_inline)) INLINE static void gpu_pack_pair_density(
    struct gpu_offload_data *buf, const struct cell *ci, const struct cell *cj,
    const double3 shift_tmp) {

  TIMER_TIC;

  /* Anything to do here? */
  const int count_ci = ci->hydro.count;
  const int count_cj = cj->hydro.count;
  if (count_ci == 0 || count_cj == 0) return;

  struct gpu_pack_vars *pack_vars = &buf->pv;

  /* Get how many particles we've packed until now */
  /* DOUBLE-CHECK THIS */
  int pack_ind = pack_vars->count_parts;

#ifdef SWIFT_DEBUG_CHECKS
  if (pack_ind + count_ci + count_cj >= 2 * pack_vars->count_max_parts) {
    error(
        "Exceeded count_max_parts_tmp. Make arrays bigger! pack_ind %d"
        "ci %i cj %i count_max %d",
        pack_ind, count_ci, count_cj, pack_vars->count_max_parts);
  }
#endif

  /* Pack the particle data into CPU-side buffers. Start by assigning the shifts
   * (if positions shifts are required)*/
  const double3 shift_i = {shift_tmp.x + cj->loc[0], shift_tmp.y + cj->loc[1],
                           shift_tmp.z + cj->loc[2]};

  /* Get first and last particles of cell i */
  const int2 cis_cie = {pack_ind, pack_ind + count_ci};

  /* Get first and last particles of cell j */
  const int2 cjs_cje = {pack_ind + count_ci, pack_ind + count_ci + count_cj};

  /* Pack cell i */
  gpu_pack_part_pair_density(ci, buf->parts_send_d, pack_ind, shift_i, cjs_cje);

  /* Update the particles packed counter */
  pack_ind += count_ci;

  /* Do the same for cj */
  const double3 shift_j = {cj->loc[0], cj->loc[1], cj->loc[2]};

  gpu_pack_part_pair_density(cj, buf->parts_send_d, pack_ind, shift_j, cis_cie);

  pack_ind += count_cj;

  /* Update incremented pack length accordingly */
  pack_vars->count_parts = pack_ind;

  TIMER_TOC(timer_dopair_gpu_pack_d);
}

/**
 * @brief packs up particle data of two cells for the pair gradient GPU task
 * into the buffers
 * @TODO: WHAT IS SHIFT_TMP?
 */
__attribute__((always_inline)) INLINE static void gpu_pack_pair_gradient(
    struct gpu_offload_data *buf, const struct cell *ci, const struct cell *cj,
    const double3 shift_tmp) {

  TIMER_TIC;

  /* Anything to do here? */
  const int count_ci = ci->hydro.count;
  const int count_cj = cj->hydro.count;
  if (count_ci == 0 || count_cj == 0) return;

  struct gpu_pack_vars *pack_vars = &buf->pv;

  /* Get how many particles we've packed until now */
  int pack_ind = pack_vars->count_parts;

#ifdef SWIFT_DEBUG_CHECKS
  if (pack_ind + count_ci + count_cj >= 2 * pack_vars->count_max_parts) {
    error(
        "Exceeded count_max_parts_tmp. Make arrays bigger! pack_ind %d"
        "ci %i cj %i count_max %d",
        pack_ind, count_ci, count_cj, pack_vars->count_max_parts);
  }
#endif

  /* Pack the particle data into CPU-side buffers. Start by assigning the shifts
   * (if positions shifts are required)*/
  const double3 shift_i = {shift_tmp.x + cj->loc[0], shift_tmp.y + cj->loc[1],
                           shift_tmp.z + cj->loc[2]};

  /* Get first and last particles of cell i */
  const int2 cis_cie = {pack_ind, pack_ind + count_ci};

  /* Get first and last particles of cell j */
  const int2 cjs_cje = {pack_ind + count_ci, pack_ind + count_ci + count_cj};

  /* Pack cell i */
  gpu_pack_part_pair_gradient(ci, buf->parts_send_g, pack_ind, shift_i,
                              cjs_cje);

  /* Update the particles packed counter */
  pack_ind += count_ci;

  /* Do the same for cj */
  const double3 shift_j = {cj->loc[0], cj->loc[1], cj->loc[2]};

  gpu_pack_part_pair_gradient(cj, buf->parts_send_g, pack_ind, shift_j,
                              cis_cie);

  /* Update the particles packed counter */
  pack_ind += count_cj;

  /* Store incremented pack length accordingly */
  pack_vars->count_parts = pack_ind;

  TIMER_TOC(timer_dopair_gpu_pack_g);
}

/**
 * @brief packs up particle data of two cells for the pair force GPU task into
 * the buffers
 * @TODO: WHAT IS SHIFT_TMP?
 */
__attribute__((always_inline)) INLINE static void gpu_pack_pair_force(
    struct gpu_offload_data *buf, const struct cell *ci, const struct cell *cj,
    const double3 shift_tmp) {

  TIMER_TIC;

  /* Anything to do here? */

  /* Anything to do here? */
  const int count_ci = ci->hydro.count;
  const int count_cj = cj->hydro.count;
  if (count_ci == 0 || count_cj == 0) return;

  struct gpu_pack_vars *pack_vars = &buf->pv;

  /* Get how many particles we've packed until now */
  int pack_ind = pack_vars->count_parts;

#ifdef SWIFT_DEBUG_CHECKS
  if (pack_ind + count_ci + count_cj >= 2 * pack_vars->count_max_parts) {
    error(
        "Exceeded count_max_parts_tmp. Make arrays bigger! pack_ind %d"
        "ci %i cj %i count_max %d",
        pack_ind, count_ci, count_cj, pack_vars->count_max_parts);
  }
#endif

  /* Pack the particle data into CPU-side buffers*/
  const double3 shift_i = {shift_tmp.x + cj->loc[0], shift_tmp.y + cj->loc[1],
                           shift_tmp.z + cj->loc[2]};

  /* Get first and last particles of cell i */
  const int2 cis_cie = {pack_ind, pack_ind + count_ci};

  /* Get first and last particles of cell j */
  const int2 cjs_cje = {pack_ind + count_ci, pack_ind + count_ci + count_cj};

  gpu_pack_part_pair_force(ci, buf->parts_send_f, pack_ind, shift_i, cjs_cje);

  /* Update the particles packed counter */
  pack_ind += count_ci;

  /* Pack the particle data into CPU-side buffers*/
  const double3 shift_j = {cj->loc[0], cj->loc[1], cj->loc[2]};

  gpu_pack_part_pair_force(cj, buf->parts_send_f, pack_ind, shift_j, cis_cie);

  /* Update the particles packed counter */
  pack_ind += count_cj;

  /* Store incremented pack length accordingly */
  pack_vars->count_parts = pack_ind;

  TIMER_TOC(timer_dopair_gpu_pack_f);
}
#endif /* RUNNER_GPU_PACK_FUNCTIONS_H */
