/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2025 Mladen Ivkovic (mladen.ivkovic@durham.ac.uk)
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
#ifndef SWIFT_HYDRO_PART_H
#define SWIFT_HYDRO_PART_H

/**
 * @file src/hydro_part.h
 * @brief Top level file for hydro particle structs. Imports the correct
 * hydro_part definition and Contains getters and setters which are identical
 * among all particles. In particular those for accessing fields of structs
 * withing the part struct.
 */

/* Config parameters. */
#include <config.h>

/* Import the right hydro particle definition */
#if defined(NONE_SPH)
#include "./hydro/None/hydro_part.h"
#define hydro_need_extra_init_loop 0
#elif defined(MINIMAL_SPH)
#include "./hydro/Minimal/hydro_part.h"
#define hydro_need_extra_init_loop 0
#elif defined(GADGET2_SPH)
#include "./hydro/Gadget2/hydro_part.h"
#define hydro_need_extra_init_loop 0
#elif defined(HOPKINS_PE_SPH)
#include "./hydro/PressureEntropy/hydro_part.h"
#define hydro_need_extra_init_loop 1
#elif defined(HOPKINS_PU_SPH)
#include "./hydro/PressureEnergy/hydro_part.h"
#define hydro_need_extra_init_loop 0
#elif defined(HOPKINS_PU_SPH_MONAGHAN)
#include "./hydro/PressureEnergyMorrisMonaghanAV/hydro_part.h"
#define hydro_need_extra_init_loop 0
#elif defined(PHANTOM_SPH)
#include "./hydro/Phantom/hydro_part.h"
#define EXTRA_HYDRO_LOOP
#define hydro_need_extra_init_loop 0
#elif defined(GIZMO_MFV_SPH) || defined(GIZMO_MFM_SPH)
#include "./hydro/Gizmo/hydro_part.h"
#define hydro_need_extra_init_loop 0
#define EXTRA_HYDRO_LOOP
#define MPI_SYMMETRIC_FORCE_INTERACTION
#elif defined(SHADOWSWIFT)
#include "./hydro/Shadowswift/hydro_part.h"
#define hydro_need_extra_init_loop 0
#define EXTRA_HYDRO_LOOP
#elif defined(PLANETARY_SPH)
#include "./hydro/Planetary/hydro_part.h"
#define hydro_need_extra_init_loop 0
#elif defined(REMIX_SPH)
#include "./hydro/REMIX/hydro_part.h"
#define hydro_need_extra_init_loop 0
#define EXTRA_HYDRO_LOOP
#define EXTRA_HYDRO_LOOP_TYPE2
#elif defined(SPHENIX_SPH)
#include "./hydro/SPHENIX/hydro_part.h"
#define hydro_need_extra_init_loop 0
#define EXTRA_HYDRO_LOOP
#elif defined(GASOLINE_SPH)
#include "./hydro/Gasoline/hydro_part.h"
#define hydro_need_extra_init_loop 0
#define EXTRA_HYDRO_LOOP
#elif defined(ANARCHY_PU_SPH)
#include "./hydro/AnarchyPU/hydro_part.h"
#define hydro_need_extra_init_loop 0
#define EXTRA_HYDRO_LOOP
#else
#error "Invalid choice of SPH variant"
#endif

#include "timestep_limiter_struct.h"

/* ----------------------------------------
 * Timestep limiter data getters/setters
 * ---------------------------------------- */

static __attribute__((always_inline)) INLINE timebin_t
part_get_timestep_limiter_wakeup(const struct part* restrict p) {
  const struct timestep_limiter_data* d = part_get_const_limiter_data_p(p);
  return timestep_limiter_get_wakeup(d);
}

static __attribute__((always_inline)) INLINE timebin_t*
part_get_timestep_limiter_wakeup_p(struct part* restrict p) {
  struct timestep_limiter_data* d = part_get_limiter_data_p(p);
  return timestep_limiter_get_wakeup_p(d);
}

static __attribute__((always_inline)) INLINE void
part_set_timestep_limiter_wakeup(struct part* restrict p,
                                 const timebin_t wakeup) {
  struct timestep_limiter_data* d = part_get_limiter_data_p(p);
  timestep_limiter_set_wakeup(d, wakeup);
}

static __attribute__((always_inline)) INLINE timebin_t
part_get_timestep_limiter_min_ngb_time_bin(const struct part* restrict p) {
  const struct timestep_limiter_data* d = part_get_const_limiter_data_p(p);
  return timestep_limiter_get_min_ngb_time_bin(d);
}

static __attribute__((always_inline)) INLINE void
part_set_timestep_limiter_min_ngb_time_bin(struct part* restrict p,
                                           const timebin_t min_ngb_time_bin) {
  struct timestep_limiter_data* d = part_get_limiter_data_p(p);
  timestep_limiter_set_min_ngb_time_bin(d, min_ngb_time_bin);
}

static __attribute__((always_inline)) INLINE char
part_get_timestep_limiter_to_be_synchronized(const struct part* restrict p) {
  const struct timestep_limiter_data* d = part_get_const_limiter_data_p(p);
  return timestep_limiter_get_to_be_synchronized(d);
}

static __attribute__((always_inline)) INLINE void
part_set_timestep_limiter_to_be_synchronized(struct part* restrict p,
                                             const char to_be_synchronized) {
  struct timestep_limiter_data* d = part_get_limiter_data_p(p);
  timestep_limiter_set_to_be_synchronized(d, to_be_synchronized);
}

/* ----------------------------------------------
 * Radiative Transfer time data getters/setters
 * ---------------------------------------------- */

static __attribute__((always_inline)) INLINE timebin_t
part_get_rt_time_bin(const struct part* restrict p) {
  const struct rt_timestepping_data* const rtd =
      part_get_const_rt_time_data_p(p);
  return rt_timestepping_data_get_time_bin(rtd);
}

static __attribute__((always_inline)) INLINE void part_set_rt_time_bin(
    struct part* restrict p, const timebin_t time_bin) {
  struct rt_timestepping_data* rtd = part_get_rt_time_data_p(p);
  rt_timestepping_data_set_time_bin(rtd, time_bin);
}

static __attribute__((always_inline)) INLINE timebin_t
part_get_rt_min_ngb_time_bin(const struct part* restrict p) {
  const struct rt_timestepping_data* const rtd =
      part_get_const_rt_time_data_p(p);
  return rt_timestepping_data_get_min_ngb_time_bin(rtd);
}

static __attribute__((always_inline)) INLINE void part_set_rt_min_ngb_time_bin(
    struct part* restrict p, const timebin_t min_ngb_time_bin) {
  struct rt_timestepping_data* rtd = part_get_rt_time_data_p(p);
  rt_timestepping_data_set_min_ngb_time_bin(rtd, min_ngb_time_bin);
}

#endif
