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
#include "minmax.h"

#include <math.h>

/**
 * @file gpu_pack_params.c
 * @brief functions related to global GPU packing parameters struct
 */

/**
 * @brief Get the global packing parameters and fill out the
 * gpu_global_pack_params struct with meaningful data.
 *
 * @param pars (return): the gpu global pack parameter struct to be filled out
 * @param pack_size: Packing size (nr of leaf cells) for self tasks
 * @param bundle_size: Size of a bundle (nr of leaf cells) for self tasks
 * @param gpu_recursion_max_depth Max depth we expect to reach while recursing
 * down from task's super level to reach leaf cells
 * @param part_buffer_size Size of particle buffer arrays. If -1, will be
 * guessed.
 * @param eta_neighours Neighbour resolution eta.
 * @param nparts_hydro how many hydro particles are on this rank
 * @param n_top_level_cells how many top level cells we have
 * @param nthreads how many threads we're running on
 */
void gpu_pack_params_set(struct gpu_global_pack_params *pars,
                         const int pack_size, const int bundle_size,
                         const int gpu_recursion_max_depth,
                         const int part_buffer_size, const float eta_neighbours,
                         const int nparts_hydro, const int n_top_level_cells,
                         const int nthreads) {

  /* Store quantities we'll need directly */
  pars->pack_size = pack_size;
  pars->bundle_size = bundle_size;

  swift_assert(pars->pack_size > 0);
  swift_assert(pars->bundle_size > 0);

  /* bundles need to be smaller than packs */
  swift_assert(pars->pack_size >= pars->bundle_size);

  /* n_bundles is the number of task bundles each thread has. Used to
   * loop through bundles */
  pars->n_bundles =
      (pars->pack_size + pars->bundle_size - 1) / pars->bundle_size;

  swift_assert(pars->n_bundles > 0);

  /* Try to estimate array sizes containing leaf cells */
  int leafcount = 1;
  for (int i = 0; i < gpu_recursion_max_depth; i++) leafcount *= 8;
  /* Add some extra buffer space */
  leafcount = ceil(leafcount * 1.2);
  pars->leaf_buffer_size = leafcount;
  swift_assert(pars->leaf_buffer_size > 0);

  int partbuff = part_buffer_size;
  if (part_buffer_size == -1) {

    /* Guess the particle array size. First try to estimate average number of
     * particles per leaf-cell. */

    /* Get smoothing length/particle spacing */
    int np_per_cell = 2 * ceil(2.0 * eta_neighbours);

    /* Apply appropriate dimensional multiplication */
#if defined(HYDRO_DIMENSION_2D)
    np_per_cell *= np_per_cell;
#elif defined(HYDRO_DIMENSION_3D)
    np_per_cell *= np_per_cell * np_per_cell;
#elif defined(HYDRO_DIMENSION_1D)
#endif

    /* Increase parts per recursed task-level cell by buffer to
       ensure we allocate enough memory. */
    partbuff = np_per_cell * leafcount;
    partbuff = ceil(1.2 * partbuff);

    /* Also guess a reasonable upper limit here based on how many particles we
     * have in total. */
    int partbuff_upper =
        ceil(2. * (float)nparts_hydro / (float)(nthreads * n_top_level_cells));
    partbuff = min(partbuff, partbuff_upper);

    /* ... but have at least space to store 2 recursion levels... */
    int partbuff_lower = ceil(1.2 * 64 * np_per_cell);
    partbuff = max(partbuff, partbuff_lower);
  }

  pars->part_buffer_size = partbuff;
  swift_assert(pars->part_buffer_size > 0);
}

/**
 * @brief Copy the global packing parameters from src to dest
 */
void gpu_pack_params_copy(const struct gpu_global_pack_params *src,
                          struct gpu_global_pack_params *dest) {

  dest->pack_size = src->pack_size;
  dest->bundle_size = src->bundle_size;
  dest->n_bundles = src->n_bundles;
  dest->leaf_buffer_size = src->leaf_buffer_size;
  dest->part_buffer_size = src->part_buffer_size;
}
