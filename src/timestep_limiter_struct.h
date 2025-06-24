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
#ifndef SWIFT_TIMESTEP_LIMITER_STRUCT_H
#define SWIFT_TIMESTEP_LIMITER_STRUCT_H

/**
 * @file src/chemistry_struct.h
 * @brief Branches between the different chemistry functions.
 */

/* Config parameters. */
#include <config.h>

/* Local includes */
#include "timeline.h"

/**
 * @brief #part-carried quantities for the time-step limiter
 */
struct timestep_limiter_data {

  /* Need waking-up ? */
  timebin_t _wakeup;

  /*! Minimal time-bin across all neighbours */
  timebin_t _min_ngb_time_bin;

  /* Do we want this particle to be synched back on the time-line? */
  char _to_be_synchronized;

#ifdef SWIFT_HYDRO_DENSITY_CHECKS
  /* Weighted number of neighbours in the limiter loop */
  float _n_limiter;

  /* Exact weighted number of neighbours in the limiter loop */
  float _n_limiter_exact;

  /* Integer number of neighbours in the limiter loop */
  int _N_limiter;

  /* Exact integer number of neighbours in the limiter loop */
  int _N_limiter_exact;
#endif
};

static __attribute__((always_inline)) INLINE timebin_t
timestep_limiter_get_wakeup(const struct timestep_limiter_data* d) {
  return d->_wakeup;
}

static __attribute__((always_inline)) INLINE void timestep_limiter_set_wakeup(
    struct timestep_limiter_data* d, const timebin_t wakeup) {
  d->_wakeup = wakeup;
}

static __attribute__((always_inline)) INLINE timebin_t
timestep_limiter_get_min_ngb_time_bin(const struct timestep_limiter_data* d) {
  return d->_min_ngb_time_bin;
}

static __attribute__((always_inline)) INLINE void
timestep_limiter_set_min_ngb_time_bin(struct timestep_limiter_data* d,
                                      const timebin_t min_ngb_time_bin) {
  d->_min_ngb_time_bin = min_ngb_time_bin;
}

static __attribute__((always_inline)) INLINE char
timestep_limiter_get_to_be_synchronized(const struct timestep_limiter_data* d) {
  return d->_to_be_synchronized;
}

static __attribute__((always_inline)) INLINE void
timestep_limiter_set_to_be_synchronized(struct timestep_limiter_data* d,
                                        const char to_be_synchronized) {
  d->_to_be_synchronized = to_be_synchronized;
}

#endif /* SWIFT_TIMESTEP_LIMITER_STRUCT_H */
