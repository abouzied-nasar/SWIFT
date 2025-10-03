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
#ifndef GPU_PACK_VARS_H
#define GPU_PACK_VARS_H

/**
 * @file gpu_pack_vars.h
 * @brief struct and corresponding functions related to GPU packing data and
 * meta-data
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "config.h"

#include <stddef.h>

/*! struct holding bookkeeping meta-data required for offloading;
 * does not depend on cuda/hip et al. */
 struct gpu_pack_vars {

  /* TODO: We only need 1 of these. task_list is used in self tasks,
   * top_task_list is used in pair tasks. They never interact. Keeping
   * this until we've figured out current bugs so as to not modify the
   * codebase beyond recognition yet. */
  union {
    /*! List of tasks and respective cells to be packed */
    struct task **task_list;

    /*! List of "top" tasks (tasks at the super level) we've packed */
    struct task **top_task_list;
  };

#ifdef SWIFT_DEBUG_CHECKS
  /*! Keep track of allocated array size for boundary checks. */
  union {
    int task_list_size;
    int top_task_list_size;
  };
#endif

  /*! TODO: documentation */
  struct cell **ci_list;

#ifdef SWIFT_DEBUG_CHECKS
  /*! Keep track of allocated array size for boundary checks. */
  int ci_list_size;
#endif

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

  /*! TODO: documentation */
  int bundle_size;

  /*! How many particles have been packed */
  int count_parts;

  /*! TODO: DOCUMENTATION */
  int tasks_packed;

  /*! TODO: documentation */
  int top_tasks_packed;

  /*! TODO: documentation */
  int *bundle_first_part;

#ifdef SWIFT_DEBUG_CHECKS
  /*! Keep track of allocated array size for boundary checks. */
  int bundle_first_part_size;
#endif

  /*! TODO: documentation */
  int *bundle_last_part;

#ifdef SWIFT_DEBUG_CHECKS
  /*! Keep track of allocated array size for boundary checks. */
  int bundle_last_part_size;
#endif

  /*! TODO: documentation */
  int *bundle_first_task_list;

#ifdef SWIFT_DEBUG_CHECKS
  /*! Keep track of allocated array size for boundary checks. */
  int bundle_first_task_list_size;
#endif

  /*! TODO: documentation */
  int count_max_parts;

  /*! TODO: documentation */
  char launch;

  /*! TODO: documentation */
  char launch_leftovers;

  /*! TODO: documentation */
  int target_n_tasks;

  /*! TODO: documentation */
  int n_bundles;

  /*! number of bundles to use for unpacking. May differ from n_bundles if
   * we're launching leftovers. */
  int n_bundles_unpack;

  /*! TODO: documentation */
  int tasksperbundle;

  /*! TODO: documentation */
  int n_daughters_total;

  /*! TODO: documentation */
  int n_daughters_packed_index;

  /*! Number of leaf cells which require interactions found during a recursive
   * search */
  int n_leaves_found;





  /*! TODO: Documentation */
  struct cell **ci_d;
  struct cell **cj_d;

#ifdef SWIFT_DEBUG_CHECKS
  /*! Keep track of allocated array size for boundary checks. */
  int ci_d_size;
  int cj_d_size;
#endif

  /*! The indexes of the first and last leaf cell packed into the particle
   * buffer per super-level pair task. The first index of this array
   * corresponds to the super-level task stored in */
   int **first_and_last_daughters;

#ifdef SWIFT_DEBUG_CHECKS
  /*! Keep track of allocated array size for boundary checks. */
  int first_and_last_daughters_size;
#endif

  /*! TODO: Documentation */
  struct cell **ci_top;
  struct cell **cj_top;

#ifdef SWIFT_DEBUG_CHECKS
  /*! Keep track of allocated array size for boundary checks. */
  int ci_top_size;
  int cj_top_size;
#endif


};

void gpu_init_pack_vars(struct gpu_pack_vars *pv);
void gpu_init_pack_vars_step(struct gpu_pack_vars *pv);

#ifdef __cplusplus
}
#endif

#endif
