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

#include "gpu_pack_vars.h"

#include "cuda/cuda_config.h"

/**
 * @file gpu_pack_vars.c
 * @brief functions related to GPU packing data and meta-data
 */

/**
 * Initialise empty gpu_pack_vars struct
 */
void gpu_init_pack_vars(struct gpu_pack_vars* pv) {

  pv->task_list = NULL;
  pv->top_task_list = NULL;
  pv->ci_list = NULL;
  pv->cellx = NULL;
  pv->celly = NULL;
  pv->cellz = NULL;

  pv->bundle_size = 0;
  pv->count_parts = 0;
  pv->tasks_packed = 0;

  pv->bundle_first_part = NULL;
  pv->bundle_last_part = NULL;
  pv->bundle_first_task_list = NULL;

  pv->count_max_parts = 0;
  pv->launch = 0;
  pv->launch_leftovers = 0;

  pv->target_n_tasks = 0;
  pv->n_bundles = 0;
  pv->tasksperbundle = 0;
  pv->n_daughters_total = 0;
  pv->n_daughters_packed_index = 0;
  pv->n_leaves_found = 0;

#ifdef SWIFT_DEBUG_CHECKS
  pv->task_list_size = 0;
  pv->top_task_list_size = 0;
  pv->ci_list_size = 0;
  pv->bundle_first_part_size = 0;
  pv->bundle_last_part_size = 0;
  pv->bundle_first_task_list_size = 0;
#endif
}

/**
 * @brief perform the initialisations required at the start of each step
 */
void gpu_init_pack_vars_step(struct gpu_pack_vars* pv) {

  /* Initialise packing counters */
  pv->tasks_packed = 0;
  pv->top_tasks_packed = 0;
  pv->count_parts = 0;
  pv->n_daughters_total = 0;
  pv->n_leaves_found = 0;
}

#ifdef __cplusplus
}
#endif
