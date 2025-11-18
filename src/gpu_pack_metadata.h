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

#include <cuda_runtime.h>

#include <stddef.h>

/* Hash table entry. TODO: Move declaration to a separate file maybe?
 * Used to construct a hash table to find unique cells */
struct hash_entry {
	/*We use the cell pointer as the key for the hash*/
	struct cell *c;
	/*This cell's index in the array of unique_cells*/
	int index;
	int occupied;
};

/*! struct holding bookkeeping meta-data required for offloading;
 * does not depend on cuda/hip et al. */
struct gpu_pack_metadata {

  /*! Lists of leaf cell pairs (ci, cj) which are to be interacted. May contain
   * entries of multiple tasks' leaf cells. */
  struct cell **ci_leaves;
  struct cell **cj_leaves;

  /*Data required for unique sorting*/
  /*list of unique cells we find in the leaves lists above*/
  struct cell **unique_cells;

  /*Is this cell unique? If so let the CPU know to pack it*/
  int2 *pack_flags;

  /*Hash table used to find unique cells*/
  struct hash_table{
    struct hash_entry * entry;
    int capacity;
    int count;
  } hash_table;

  /*Array to guide each leaf computation to it's cell's/s' index in the array of unique cells/particles*/
  int2 *my_index;

  /*number of unique cells we find*/
  int n_unique;

  /*! Number of leaf cells which require interactions found during recursive
   * search for a single task */
  int task_n_leaves;

  /*! List of tasks cells to be packed */
  struct task **task_list;

  /*! Index of the first (pair of) leaf cell(s) packed into the particle buffer
   * of tasks stored in task_list. */
  int *task_first_packed_leaf;

  /*! Index of the last (pair of) leaf cell(s) packed into the particle buffer
   * of tasks stored in task_list. */
  int *task_last_packed_leaf;

  /*! The index of the first particle in the buffer arrays of tasks stored in
   * task_list. */
  int *task_first_packed_part;

  /*! Index of the first particle of a bundle in the buffer arrays */
  int *bundle_first_part;

  /*! Index of the first particle of a bundle in the buffer arrays */
  int *bundle_first_cell;

  /*! Count how many (super-level) tasks we've identified for packing */
  int tasks_in_list;

  /*! How many particles have been packed */
  int count_parts;

  /*! How many particles have been packed */
  int count_parts_unique;

  /*! How many (pairs of) leaf cells we have alread packed (their particle data
   * copied) into the buffers */
  int n_leaves_packed;

  /*! Total number of leaf cells which require interactions found during
   * recursive searches since last offload cycle */
  int n_leaves;

  /*! Are these buffers ready to trigger launch on GPU? */
  char launch;

  /*! Are we launching leftovers (fewer than target offload size)? */
  char launch_leftovers;

  /*! Global (fixed) packing parameters */
  struct gpu_global_pack_params params;

  /*! Is this metadata for a pair task? */
  char is_pair_task;

  int hash_size;

#ifdef SWIFT_DEBUG_CHECKS
  /*! Size of the send_part struct used */
  size_t send_struct_size;

  /*! Size of the recv_part struct used */
  size_t recv_struct_size;

#endif
};

void gpu_pack_metadata_init(struct gpu_pack_metadata *md,
                            const struct gpu_global_pack_params *params,
                            const char is_pair_task);
void gpu_pack_metadata_init_step(struct gpu_pack_metadata *md);
void gpu_pack_metadata_reset(struct gpu_pack_metadata *md,
                             int reset_leaves_lists);
void gpu_pack_metadata_free(struct gpu_pack_metadata *md);

#ifdef __cplusplus
}
#endif

#endif
