/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2025 Abouzied M. A. Nasar (abouzied.nasar@manchester.ac.uk)
 *                    Mladen Ivkovic
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
#ifndef GPU_PACK_VARS_H
#define GPU_PACK_VARS_H


#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <vector_types.h>

#include "scheduler.h"


/*! Struct holding global packing parameters. */
struct gpu_global_pack_params{

  size_t target_n_tasks;
  size_t target_n_tasks_pair;
  size_t bundle_size;
  size_t bundle_size_pair;
  size_t n_bundles;
  size_t n_bundles_pair;
  int count_max_parts;
};


/**
 * TODO Abouzeid: documentation
 */
struct gpu_pack_vars {

  /*! List of tasks and respective cells to be packed */
  struct task **task_list;

  /*! TODO: documentation */
  struct task **top_task_list;

  /*! TODO: documentation */
  struct cell **ci_list;

  /*! TODO: documentation */
  struct cell **cj_list;

  /* List of cell shifts and positions. Shifts are used for pair tasks,
   * while cell positions are used for self tasks.*/
  union {
    double *cellx;
    double *shiftx;
  };
  union {
    double *celly;
    double *shifty;
  };
  union {
    double *cellz;
    double *shiftz;
  };
  /* List of cell shifts and positions ON DEVICE. Shifts are used for
   * pair tasks, while cell positions are used for self tasks.*/
  union {
    double *d_cellx;
    double *d_shiftx;
  };
  union {
    double *d_celly;
    double *d_shifty;
  };
  union {
    double *d_cellz;
    double *d_shiftz;
  };

  /*! TODO: documentation */
  size_t bundle_size;

  /*! How many particles in a bundle*/
  size_t count_parts;
  /**/
  size_t tasks_packed;

  /*! TODO: documentation */
  size_t top_tasks_packed;

  /*! TODO: documentation */
  int *task_first_part;

  /*! TODO: documentation */
  int *task_last_part;

  /*! TODO: documentation */
  int *d_task_first_part;

  /*! TODO: documentation */
  int *d_task_last_part;

  /*! TODO: documentation */
  int *bundle_first_part;

  /*! TODO: documentation */
  int *bundle_last_part;

  /*! TODO: documentation */
  int *bundle_first_task_list;

  /*! TODO: documentation */
  size_t count_max_parts;

  /*! TODO: documentation */
  char launch;

  /*! TODO: documentation */
  char launch_leftovers;

  /*! TODO: documentation */
  size_t target_n_tasks;

  /*! TODO: documentation */
  size_t n_bundles;

  /*! number of bundles to use for unpacking. May differ from n_bundles if
   * we're launching leftovers. */
  size_t n_bundles_unpack;

  /*! TODO: documentation */
  size_t tasksperbundle;

  /*! TODO: documentation */
  size_t n_daughters_total;

  /*! TODO: documentation */
  size_t n_daughters_packed_index;

  /*! TODO: documentation */
  size_t n_leaves_found;

  /*! TODO: documentation */
  size_t n_leaves_total;
};


void gpu_init_pack_vars(struct gpu_pack_vars* pv);
void gpu_get_pack_params(struct gpu_global_pack_params* pars, const struct scheduler* sched, const float eta_neighbours);

#ifdef __cplusplus
}
#endif

#endif
