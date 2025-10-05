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
    const struct gpu_global_pack_params *params) {

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
  md->task_first_last_leaf_pair = NULL;
  md->ci_super = NULL;
  md->cj_super = NULL;

  gpu_pack_params_copy(params, &md->params);

#ifdef SWIFT_DEBUG_CHECKS
  md->task_list_size = 0;
  md->ci_list_size = 0;
  md->bundle_first_part_size = 0;
  md->bundle_last_part_size = 0;
  md->bundle_first_leaf_size = 0;
  md->ci_leaves_size = 0;
  md->cj_leaves_size = 0;
  md->task_first_last_leaf_pair_size = 0;
  md->ci_super_size = 0;
  md->cj_super_size = 0;
#endif

}

/**
 * @brief perform the initialisations required at the start of each step
 */
void gpu_pack_metadata_init_step(struct gpu_pack_metadata* md) {

  /* Initialise packing counters */
  gpu_pack_metadata_reset(md);
}

/**
 * @brief reset the meta-data after a completed launch and unpack to prepare
 * for the next pack operation
 */
void gpu_pack_metadata_reset(struct gpu_pack_metadata* md){

  md->launch = 0;
  md->launch_leftovers = 0;

  /* Self tasks */
  md->count_parts = 0;
  md->self_tasks_packed = 0;
  md->n_bundles_unpack = 0;

#ifdef SWIFT_DEBUG_CHECKS
  for (int i = 0; i < md->task_list_size; i++)
    md->task_list[i] = NULL;
  for (int i = 0; i < md->ci_list_size; i++)
    md->ci_list[i] = NULL;
  for (int i = 0; i < md->bundle_first_part_size; i++)
    md->bundle_first_part[i] = -123;
  for (int i = 0; i < md->bundle_last_part_size; i++)
    md->bundle_last_part[i] = -123;
  for (int i = 0; i < md->bundle_first_leaf_size; i++)
    md->bundle_first_leaf[i] = -123;
#endif

  /* Pair tasks */
  md->n_leaves = 0;
  md->tasks_in_list = 0;
  md->leaf_pairs_packed = 0;
  /* md->count_parts = 0; */ /* Already covered in self task section above */

#ifdef SWIFT_DEBUG_CHECKS
  for (int i = 0; i < md->task_first_last_leaf_pair_size; i++){
    md->task_first_last_leaf_pair[i][0] = -123;
    md->task_first_last_leaf_pair[i][1] = -123;
  }
  for (int i = 0; i < md->ci_super_size; i++)
    md->ci_super[i] = NULL;
  for (int i = 0; i < md->cj_super_size; i++)
    md->cj_super[i] = NULL;
  for (int i = 0; i < md->bundle_first_part_size; i++)
    md->bundle_first_part[i] = -123;
  for (int i = 0; i < md->bundle_first_leaf_size; i++)
    md->bundle_first_leaf[i] = -123;
#endif
}

#ifdef __cplusplus
}
#endif
