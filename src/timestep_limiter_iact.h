/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2018 Matthieu Schaller (schaller@strw.leidenuniv.nl)
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
#ifndef SWIFT_TIMESTEP_LIMITER_IACT_H
#define SWIFT_TIMESTEP_LIMITER_IACT_H

#include "accumulate.h"
#include "minmax.h"
#include "part.h"

/**
 * @brief Force interaction between two particles.
 *
 * @param r2 Comoving square distance between the two particles.
 * @param dx Comoving vector separating both particles (pi - pj).
 * @param hi Comoving smoothing-length of particle i.
 * @param hj Comoving smoothing-length of particle j.
 * @param pi First particle.
 * @param pj Second particle.
 * @param a Current scale factor.
 * @param H Current Hubble parameter.
 */
__attribute__((always_inline)) INLINE static void runner_iact_timebin(
    const float r2, const float dx[3], const float hi, const float hj,
    struct part *restrict pi, struct part *restrict pj, const float a,
    const float H) {

  const timebin_t ti = part_get_time_bin(pi);
  const timebin_t tj = part_get_time_bin(pj);

  /* Update the minimal time-bin */
  if (tj > 0){
    struct timestep_limiter_data* limiter_data_i = part_get_limiter_data(pi);
    limiter_data_i->min_ngb_time_bin =
        min(limiter_data_i->min_ngb_time_bin, tj);
  }

  if (ti > 0){
    struct timestep_limiter_data* limiter_data_j = part_get_limiter_data(pj);
    limiter_data_j->min_ngb_time_bin =
        min(limiter_data_j->min_ngb_time_bin, ti);
  }
}

/**
 * @brief Timebin interaction between two particles (non-symmetric).
 *
 * @param r2 Comoving square distance between the two particles.
 * @param dx Comoving vector separating both particles (pi - pj).
 * @param hi Comoving smoothing-length of particle i.
 * @param hj Comoving smoothing-length of particle j.
 * @param pi First particle.
 * @param pj Second particle (not updated).
 * @param a Current scale factor.
 * @param H Current Hubble parameter.
 */
__attribute__((always_inline)) INLINE static void runner_iact_nonsym_timebin(
    const float r2, const float dx[3], const float hi, const float hj,
    struct part *restrict pi, const struct part *restrict pj, const float a,
    const float H) {

  const timebin_t tj = part_get_time_bin(pj);

  /* Update the minimal time-bin */
  if (tj > 0){
    struct timestep_limiter_data* limiter_data_i = part_get_limiter_data(pi);
    limiter_data_i->min_ngb_time_bin =
        min(limiter_data_i->min_ngb_time_bin, tj);
  }
}

/**
 * @brief Timestep limiter loop
 */
__attribute__((always_inline)) INLINE static void runner_iact_limiter(
    const float r2, const float dx[3], const float hi, const float hj,
    struct part *restrict pi, struct part *restrict pj, const float a,
    const float H) {

  /* Nothing to do here if both particles are active */

#ifdef SWIFT_HYDRO_DENSITY_CHECKS

  float wi, wj;
  const float r = sqrtf(r2);

  const float hi_inv = 1.f / hi;
  const float ui = r * hi_inv;
  kernel_eval(ui, &wi);

  const float hj_inv = 1.f / hj;
  const float uj = r * hj_inv;
  kernel_eval(uj, &wj);

  accumulate_add_f(&pi->limiter_data.n_limiter, wi);
  accumulate_add_f(&pj->limiter_data.n_limiter, wj);
  accumulate_inc_i(&pi->limiter_data.N_limiter);
  accumulate_inc_i(&pj->limiter_data.N_limiter);
#endif
}

/**
 * @brief Timestep limiter loop (non-symmetric version)
 */
__attribute__((always_inline)) INLINE static void runner_iact_nonsym_limiter(
    const float r2, const float dx[3], const float hi, const float hj,
    struct part *restrict pi, struct part *restrict pj, const float a,
    const float H) {

  const timebin_t ti = part_get_time_bin(pi);
  const timebin_t tj = part_get_time_bin(pj);

  /* Wake up the neighbour? */
  if (tj > ti + time_bin_neighbour_max_delta_bin) {

    /* Store the smallest time bin that woke up this particle */
    struct timestep_limiter_data* limiter_data_j = part_get_limiter_data(pj);
    accumulate_max_c(&limiter_data_j->wakeup, -ti);
  }

#ifdef SWIFT_HYDRO_DENSITY_CHECKS
  float wi;

  const float r = sqrtf(r2);
  const float hi_inv = 1.f / hi;
  const float ui = r * hi_inv;
  kernel_eval(ui, &wi);

  accumulate_add_f(&pi->limiter_data.n_limiter, wi);
  accumulate_inc_i(&pi->limiter_data.N_limiter);
#endif
}

#endif /* SWIFT_TIMESTEP_LIMITER_IACT_H */
