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

#ifndef GPU_PACK_PARAMS_H
#define GPU_PACK_PARAMS_H

/**
 * @file gpu_pack_params.h
 * @brief struct containing global GPU packing parameters
 */

#include <stddef.h>

/*! Struct holding global packing parameters. */
struct gpu_global_pack_params {

  size_t target_n_tasks;
  size_t target_n_tasks_pair;
  size_t bundle_size;
  size_t bundle_size_pair;
  size_t n_bundles;
  size_t n_bundles_pair;
  int count_max_parts;
};


void gpu_get_pack_params(struct gpu_global_pack_params *pars,
                         const float eta_neighbours);

#endif
