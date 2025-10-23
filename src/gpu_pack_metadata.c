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
#ifdef __cplusplus
extern "C" {
#endif

#include "gpu_pack_metadata.h"

/**
 * @file gpu_pack_metadata.c
 * @brief functions related to GPU packing data and meta-data
 */

/**
 * @brief Initialise empty gpu_pack_metadata struct
 *
 * @param md medata struct to be initialised
 * @param params gpu_global_pack_params struct containing valid parameters
 */
void gpu_pack_metadata_init(struct gpu_pack_metadata* md,
                            const struct gpu_global_pack_params* params) {

  md->task_list = NULL;
  md->tasks_in_list = 0;

  md->ci_list = NULL;

  md->count_parts = 0;
  md->self_tasks_packed = 0;
  md->leaf_pairs_packed = 0;

  md->launch = 0;
  md->launch_leftovers = 0;

  md->bundle_first_part = NULL;
  md->bundle_last_part = NULL;
  md->bundle_first_leaf = NULL;

  md->n_bundles_unpack = 0;
  md->n_leaves = 0;
  md->task_n_leaves = 0;

  md->ci_leaves = NULL;
  md->cj_leaves = NULL;
  md->task_first_last_packed_leaf_pair = NULL;
  md->ci_super = NULL;
  md->cj_super = NULL;

  gpu_pack_params_copy(params, &md->params);

#ifdef SWIFT_DEBUG_CHECKS
  md->is_pair_task = 0;
  md->send_struct_size = 0;
  md->recv_struct_size = 0;
#endif
}

/**
 * @brief perform the initialisations required at the start of each step
 */
void gpu_pack_metadata_init_step(struct gpu_pack_metadata* md) {

  gpu_pack_metadata_reset(md, /*reset_leaves_lists=*/1);
}

/**
 * @brief reset the meta-data after a completed launch and unpack to prepare
 * for the next pack operation
 *
 * @param reset_leaves_lists if 1, also reset lists containing leaves.
 */
void gpu_pack_metadata_reset(struct gpu_pack_metadata* md,
                             int reset_leaves_lists) {

  md->launch = 0;
  md->launch_leftovers = 0;

  /* Self tasks */
  md->count_parts = 0;
  md->self_tasks_packed = 0;
  md->n_bundles_unpack = 0;

  /* Pair tasks */
  md->n_leaves = 0;
  md->tasks_in_list = 0;
  md->leaf_pairs_packed = 0;

#ifdef SWIFT_DEBUG_CHECKS
  const struct gpu_global_pack_params pars = md->params;
  size_t n_bundles = pars.n_bundles;
  size_t pack_size = pars.pack_size;
  if (md->is_pair_task) {
    n_bundles = pars.n_bundles_pair;
    pack_size = pars.pack_size_pair;
  }

  for (size_t i = 0; i < n_bundles; i++) md->bundle_first_part[i] = 0;
  for (size_t i = 0; i < n_bundles; i++) md->bundle_last_part[i] = 0;
  for (size_t i = 0; i < n_bundles; i++) md->bundle_first_leaf[i] = 0;

  for (size_t i = 0; i < pack_size; i++) md->task_list[i] = NULL;

  if (md->is_pair_task) {
    for (size_t i = 0; i < pack_size; i++) {
      md->task_first_last_packed_leaf_pair[i][0] = 0;
      md->task_first_last_packed_leaf_pair[i][1] = 0;
    }

    for (size_t i = 0; i < pack_size; i++) {
      md->ci_super[i] = NULL;
    }
    for (size_t i = 0; i < pack_size; i++) {
      md->cj_super[i] = NULL;
    }

    for (size_t i = 0; i < pack_size; i++) md->task_first_part[i] = 0;

    if (reset_leaves_lists) {
      for (int i = 0; i < pars.leaf_buffer_size; i++) {
        md->ci_leaves[i] = NULL;
      }
      for (int i = 0; i < pars.leaf_buffer_size; i++) {
        md->cj_leaves[i] = NULL;
      }
    }

  } else { /* self task */

    for (size_t i = 0; i < pack_size; i++) md->ci_list[i] = NULL;
  }

#endif
}

#ifdef __cplusplus
}
#endif
