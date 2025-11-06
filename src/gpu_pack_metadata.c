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

/**
 * @file gpu_pack_metadata.c
 * @brief functions related to GPU packing data and meta-data
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "gpu_pack_metadata.h"

#include <stdlib.h>

/**
 * @brief Initialise empty gpu_pack_metadata struct
 *
 * @param md medata struct to be initialised
 * @param params gpu_global_pack_params struct containing valid parameters
 */
void gpu_pack_metadata_init(struct gpu_pack_metadata *md,
                            const struct gpu_global_pack_params *params) {

  /* Grab some handles */
  const size_t pack_size = params->pack_size;
  const size_t n_bundles = params->n_bundles;
  const size_t leaf_buffer_size = params->leaf_buffer_size;

  md->ci_leaves =
      (struct cell **)malloc(leaf_buffer_size * sizeof(struct cell *));
  for (size_t i = 0; i < leaf_buffer_size; i++) md->ci_leaves[i] = 0;

  md->cj_leaves =
      (struct cell **)malloc(leaf_buffer_size * sizeof(struct cell *));
  for (size_t i = 0; i < leaf_buffer_size; i++) md->cj_leaves[i] = 0;

  md->task_list = (struct task **)malloc(pack_size * sizeof(struct task *));
  for (size_t i = 0; i < pack_size; i++) md->task_list[i] = NULL;

  md->task_first_packed_leaf = (int *)malloc(pack_size * sizeof(int));
  for (size_t i = 0; i < pack_size; i++) md->task_first_packed_leaf[i] = 0;

  md->task_last_packed_leaf = (int *)malloc(pack_size * sizeof(int));
  for (size_t i = 0; i < pack_size; i++) md->task_last_packed_leaf[i] = 0;

  md->task_first_packed_part = malloc(pack_size * sizeof(int));
  for (size_t i = 0; i < pack_size; i++) md->task_first_packed_part[i] = 0;

  md->bundle_first_part = (int *)malloc(n_bundles * sizeof(int));
  for (size_t i = 0; i < n_bundles; i++) md->bundle_first_part[i] = 0;

  md->task_n_leaves = 0;
  md->tasks_in_list = 0;
  md->count_parts = 0;
  md->n_leaves_packed = 0;
  md->n_leaves = 0;
  md->launch = 0;
  md->launch_leftovers = 0;

  gpu_pack_params_copy(params, &md->params);

#ifdef SWIFT_DEBUG_CHECKS
  md->send_struct_size = 0;
  md->recv_struct_size = 0;
#endif
}

/**
 * @brief perform the initialisations required at the start of each step
 */
void gpu_pack_metadata_init_step(struct gpu_pack_metadata *md) {

  gpu_pack_metadata_reset(md, /*reset_leaves_lists=*/1);
}

/**
 * @brief reset the meta-data after a completed launch and unpack to prepare
 * for the next pack operation
 *
 * @param reset_leaves_lists if 1, also reset lists containing leaves.
 */
void gpu_pack_metadata_reset(struct gpu_pack_metadata *md,
                             int reset_leaves_lists) {

  /* md->task_n_leaves = 0;  */ /* Don't reset this! */
  md->tasks_in_list = 0;
  md->count_parts = 0;
  md->n_leaves_packed = 0;
  md->n_leaves = 0;
  md->launch = 0;
  md->launch_leftovers = 0;

#ifdef SWIFT_DEBUG_CHECKS
  const struct gpu_global_pack_params pars = md->params;
  int n_bundles = pars.n_bundles;
  int pack_size = pars.pack_size;

  if (reset_leaves_lists) {
    for (int i = 0; i < pars.leaf_buffer_size; i++) md->ci_leaves[i] = NULL;
    for (int i = 0; i < pars.leaf_buffer_size; i++) md->cj_leaves[i] = NULL;
  }
  for (int i = 0; i < pack_size; i++) md->task_list[i] = NULL;
  for (int i = 0; i < pack_size; i++) md->task_first_packed_leaf[i] = 0;
  for (int i = 0; i < pack_size; i++) md->task_last_packed_leaf[i] = 0;
  for (int i = 0; i < pack_size; i++) md->task_first_packed_part[i] = 0;
  for (int i = 0; i < n_bundles; i++) md->bundle_first_part[i] = 0;

#endif
}

/**
 * @brief Free the allocations.
 */
void gpu_pack_metadata_free(struct gpu_pack_metadata *md) {

  free((void *)md->ci_leaves);
  free((void *)md->cj_leaves);
  free((void *)md->task_list);
  free(md->task_first_packed_leaf);
  free(md->task_last_packed_leaf);
  free(md->task_first_packed_part);
  free(md->bundle_first_part);
}

#ifdef __cplusplus
}
#endif
