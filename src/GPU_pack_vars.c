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

#include "GPU_pack_vars.h"
#include "cuda/cuda_config.h"


/**
 * Get global packing parameters from the scheduler and fill out the
 * gpu_pack_params struct
 *
 * @params pars (return): the gpu global pack parameter struct to be filled out
 * @params sched: the @scheduler
 * @params eta_neighours: Neighbour resolution eta.
 */
void gpu_get_pack_params(struct gpu_global_pack_params* pars, const struct scheduler* sched, const float eta_neighbours){

  pars->target_n_tasks = sched->pack_size;
  pars->target_n_tasks_pair = sched->pack_size_pair;
  pars->bundle_size = N_TASKS_BUNDLE_SELF;
  pars->bundle_size_pair = N_TASKS_BUNDLE_PAIR;

  /* A. Nasar: n_bundles is the number of task bundles each thread has. Used to loop through bundles */
  pars->n_bundles = (pars->target_n_tasks + pars->bundle_size - 1) / pars->bundle_size;
  pars->n_bundles_pair = (pars->target_n_tasks_pair + pars->bundle_size_pair - 1) / pars->bundle_size_pair;

  /* A. Nasar: Try to estimate average number of particles per leaf-level cell */
  /* Get smoothing length/particle spacing */
  int np_per_cell = ceil(2.0 * eta_neighbours);

  /* Cube to find average number of particles in 3D */
  np_per_cell *= np_per_cell * np_per_cell;

  /* A. Nasar: Increase parts per recursed task-level cell by buffer to
    ensure we allocate enough memory */
  const int buff = ceil(0.5 * np_per_cell);

  /*A. Nasar: Multiplication by 2 is also to ensure we do not over-run
   *  the allocated memory on buffers and GPU. This can happen if calculated h
   * is larger than cell width and splitting makes bigger than target cells */

  /* Leave this until we implement recursive self tasks -> Exaggerated as we will
   * off-load really big cells since we don't recurse */
  pars->count_max_parts = 64ul * 8ul * pars->target_n_tasks * (np_per_cell + buff);

}



/**
 * Initialise empty gpu_pack_vars struct
 */
void gpu_init_pack_vars(struct gpu_pack_vars* pv){

  pv->task_list = NULL;
  pv->top_task_list = NULL;
  pv->ci_list = NULL;
  pv->cj_list = NULL;
  pv->cellx = NULL;
  pv->celly = NULL;
  pv->cellz = NULL;
  pv->d_cellx = NULL;
  pv->d_celly = NULL;
  pv->d_cellz = NULL;

  pv->bundle_size = 0;
  pv->count_parts = 0;
  pv->tasks_packed = 0;

  pv->task_first_part = NULL;
  pv->task_last_part = NULL;
  pv->d_task_first_part = NULL;
  pv->d_task_last_part = NULL;
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
  pv->n_leaves_total = 0;
}


/**
 * @brief perform the initialisations required at the start of each step
 */
void gpu_init_pack_vars_step(struct gpu_pack_vars* pv){

  // Initialise packing counters
  pv->tasks_packed = 0;
  pv->count_parts = 0;
  pv->top_tasks_packed = 0;
  pv->n_daughters_total = 0;
  pv->n_leaves_found = 0;
  pv->n_leaves_total = 0;
}

#ifdef __cplusplus
}
#endif


