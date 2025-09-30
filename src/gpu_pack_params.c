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

#include "gpu_pack_params.h"

#include "error.h"

#include <math.h>

/**
 * @file gpu_pack_params.c
 * @brief functions related to global GPU packing parameters struct
 */

/**
 * Get global packing parameters from the scheduler and fill out the
 * gpu_pack_params struct
 *
 * @params pars (return): the gpu global pack parameter struct to be filled out
 * @params sched: the @scheduler
 * @params eta_neighours: Neighbour resolution eta.
 */
void gpu_set_pack_params(struct gpu_global_pack_params* pars,
                         const size_t pack_size, const size_t pack_size_pair,
                         const size_t bundle_size,
                         const size_t bundle_size_pair,
                         const float eta_neighbours) {

  pars->pack_size = pack_size;
  pars->pack_size_pair = pack_size_pair;
  pars->bundle_size = bundle_size;
  pars->bundle_size_pair = bundle_size_pair;

  swift_assert(pars->pack_size > 0);
  swift_assert(pars->pack_size_pair > 0);
  swift_assert(pars->bundle_size > 0);
  swift_assert(pars->bundle_size_pair > 0);

  /* A. Nasar: n_bundles is the number of task bundles each thread has. Used to
   * loop through bundles */
  pars->n_bundles =
      (pars->pack_size + pars->bundle_size - 1) / pars->bundle_size;
  pars->n_bundles_pair = (pars->pack_size_pair + pars->bundle_size_pair - 1) /
                         pars->bundle_size_pair;

  swift_assert(pars->n_bundles > 0);
  swift_assert(pars->n_bundles_pair > 0);

  /* Now try to estimate average number of particles per leaf-cell. */
  /* First get smoothing length/particle spacing */
  /* Multiplication by 2 is also to ensure we do not over-run the allocated
   * memory on buffers and GPU. This can happen if calculated h is larger than
   * cell width and splitting makes bigger than target cells */
  size_t np_per_cell = 2 * ceil(2.0 * eta_neighbours);

  /* Apply appropriate dimensional multiplication */
#if defined(HYDRO_DIMENSION_2D)
  np_per_cell *= np_per_cell;
#elif defined(HYDRO_DIMENSION_3D)
  np_per_cell *= np_per_cell * np_per_cell;
#elif defined(HYDRO_DIMENSION_1D)
#else
#pragma error("We shouldn't be here.")
#endif

  /* Increase parts per recursed task-level cell by buffer to
     ensure we allocate enough memory. */
  const size_t buff = ceil(0.25 * np_per_cell);

  /* Leave this until we implement recursive self tasks -> Exaggerated as we
   * will off-load really big cells since we don't recurse */
  pars->count_max_parts = 64ul * pars->pack_size * (np_per_cell + buff);

  swift_assert(pars->count_max_parts > 0);
}
