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

  /*! How many tasks we target to offload in total, per offload cycle */
  int pack_size;

  /*! An offload cycle offloads `pack_size` tasks in total over several bundles
   * of `bundle_size` tasks per bundle. */
  int bundle_size;

  /*! How many bundles we expect to offload each cycle. */
  int n_bundles;

  /*! Size of the particle data buffers for offloading */
  int part_buffer_size;

  /*! Size of the cell list buffers for offloading. */
  int leaf_buffer_size;
};

void gpu_pack_params_set(struct gpu_global_pack_params *pars,
                         const int pack_size, const int bundle_size,
                         const int gpu_recursion_max_depth,
                         const int part_buffer_size, const float eta_neighbours,
                         const int nparts_hydro, const int n_top_level_cells,
                         const int nthreads);
void gpu_pack_params_copy(const struct gpu_global_pack_params *src,
                          struct gpu_global_pack_params *dest);

#endif
