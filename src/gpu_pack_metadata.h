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
#ifndef GPU_PACK_METADATA_H
#define GPU_PACK_METADATA_H

/**
 * @file gpu_pack_metadata.h
 * @brief struct and corresponding functions related to GPU packing data and
 * meta-data
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "config.h"
#include "gpu_pack_params.h"

#include <stddef.h>

/*! struct holding bookkeeping meta-data required for offloading;
 * does not depend on cuda/hip et al. */
struct gpu_pack_metadata {

  /*! List of tasks cells to be packed */
  struct task **task_list;

  /*! Count how many (super-level) tasks we've identified for packing */
  int tasks_in_list;

  /*! List of super level cells to be packed in self tasks */
  /* TODO: Use ci_super instead, as for pair tasks. We don't need this
   * to be separate. */
  struct cell **ci_list;

  /*! How many particles have been packed */
  int count_parts;

  /*! How many self tasks we have already packed (their particle data copied)
   * into the buffers */
  /* TODO: to be replaced by leaf_pairs_packed */
  int self_tasks_packed;

  /*! How man pairs of leaf cells for pair interactions we have alread packed
   * (their particle data copied) into the buffers */
  int leaf_pairs_packed;

  /*! Are these buffers ready to trigger launch on GPU? */
  char launch;

  /*! Are we launching leftovers (fewer than target offload size)? */
  char launch_leftovers;

  /*! Index of the first particle of a bundle in the buffer arrays */
  int *bundle_first_part;

  /*! Index of the last particle of a bundle in the buffer arrays */
  int *bundle_last_part;

  /*! For self-tasks: The index of the first leaf cell (pair) of a bundle as it
   * is stored in the task_list. For pair-tasks: The index of the first leaf
   * cell of a bundle in the ci_leaves, cj_leaves arrays.*/
  int *bundle_first_leaf;

  /*! Number of bundles to use unpack. May differ from target_n_bundles if
   * we're launching leftovers. */
  int n_bundles_unpack;

  /*! Number of leaf cells which require interactions found during a recursive
   * search */
  int n_leaves;

  /*! How many leaves have been identified for a single (pair) task during
   * recursion */
  int task_n_leaves;

  /*! Lists of leaf cell pairs (ci, cj) which are to be interacted. May contain
   * entries of multiple tasks' leaf cells. */
  struct cell **ci_leaves;
  struct cell **cj_leaves;

  /*! The indexes of the first and last leaf cell pairs packed into the
   * particle buffer per super-level (pair) task. The first index of this array
   * corresponds to the super-level task stored in `task_list`.*/
  int **task_first_last_packed_leaf_pair;

  /*! Cells ci and cj at the super level of the associated pair task */
  /* TODO: From what I see, we only use them to lock and unlock cell trees.
   * We can access them through t->ci, t->cj since we keep track of tasks
   * in md->task_list. So these need purging. */
  struct cell **ci_super;
  struct cell **cj_super;

  /*! Global (fixed) packing parameters */
  struct gpu_global_pack_params params;
};

void gpu_pack_metadata_init(struct gpu_pack_metadata *md,
                            const struct gpu_global_pack_params *params);
void gpu_pack_metadata_init_step(struct gpu_pack_metadata *md);
void gpu_pack_metadata_reset(struct gpu_pack_metadata *md,
                             int reset_leaves_lists);

#ifdef __cplusplus
}
#endif

#endif
