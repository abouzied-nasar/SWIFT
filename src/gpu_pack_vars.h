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
#include <vector_types.h>

/**
 * TODO Abouzeid: documentation
 */
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
    size_t task_list_size;
    size_t top_task_list_size;
  };
#endif

  /*! TODO: documentation */
  struct cell **ci_list;

#ifdef SWIFT_DEBUG_CHECKS
  /*! Keep track of allocated array size for boundary checks. */
  size_t ci_list_size;
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
  size_t bundle_size;

  /*! How many particles in a bundle*/
  size_t count_parts;

  /*! TODO: DOCUMENTATION */
  size_t tasks_packed;

  /*! TODO: documentation */
  size_t top_tasks_packed;

  /*! TODO: documentation */
  int *bundle_first_part;

#ifdef SWIFT_DEBUG_CHECKS
  /*! Keep track of allocated array size for boundary checks. */
  size_t bundle_first_part_size;
#endif

  /*! TODO: documentation */
  int *bundle_last_part;

#ifdef SWIFT_DEBUG_CHECKS
  /*! Keep track of allocated array size for boundary checks. */
  size_t bundle_last_part_size;
#endif

  /*! TODO: documentation */
  int *bundle_first_task_list;

#ifdef SWIFT_DEBUG_CHECKS
  /*! Keep track of allocated array size for boundary checks. */
  size_t bundle_first_task_list_size;
#endif

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

  /*! Number of leaf cells which require interactions found during a recursive
   * search */
  size_t n_leaves_found;
};

void gpu_init_pack_vars(struct gpu_pack_vars *pv);
void gpu_init_pack_vars_step(struct gpu_pack_vars *pv);

#ifdef __cplusplus
}
#endif

#endif
