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
void gpu_get_pack_params(struct gpu_global_pack_params* pars,
                         const float eta_neighbours) {

  pars->target_n_tasks = sched->pack_size;
  pars->target_n_tasks_pair = sched->pack_size_pair;
  pars->bundle_size = N_TASKS_BUNDLE_SELF;
  pars->bundle_size_pair = N_TASKS_BUNDLE_PAIR;

  /* A. Nasar: n_bundles is the number of task bundles each thread has. Used to
   * loop through bundles */
  pars->n_bundles =
      (pars->target_n_tasks + pars->bundle_size - 1) / pars->bundle_size;
  pars->n_bundles_pair =
      (pars->target_n_tasks_pair + pars->bundle_size_pair - 1) /
      pars->bundle_size_pair;

  /* A. Nasar: Try to estimate average number of particles per leaf-cell */

  /* Get smoothing length/particle spacing */
  int np_per_cell = ceil(2.0 * eta_neighbours);

  /* Apply appropriate dimensional multiplication */
#if defined(HYDRO_DIMENSION_2D)
  np_per_cell *= np_per_cell;
#elif defined(HYDRO_DIMENSION_3D)
  np_per_cell *= np_per_cell * np_per_cell;
#elif defined(HYDRO_DIMENSION_1D)
#else
  /* TODO mladen: temporary debug check, remove before MR */
#pragma error("THIS IS NOT SUPPOSED TO HAPPEN.")
#endif

  /* A. Nasar: Increase parts per recursed task-level cell by buffer to
    ensure we allocate enough memory */
  const int buff = ceil(0.5 * np_per_cell);

  /* A. Nasar: Multiplication by 2 is also to ensure we do not over-run
   * the allocated memory on buffers and GPU. This can happen if calculated h
   * is larger than cell width and splitting makes bigger than target cells */

  /* Leave this until we implement recursive self tasks -> Exaggerated as we
   * will off-load really big cells since we don't recurse */
  pars->count_max_parts =
      64ul * 8ul * pars->target_n_tasks * (np_per_cell + buff);
}


