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
#ifndef RUNNER_DOIACT_FUNCTIONS_HYDRO_GPU_H
#define RUNNER_DOIACT_FUNCTIONS_HYDRO_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

#include "active.h"
#include "error.h"
#include "inline.h"
#include "runner.h"
#include "runner_gpu_pack_functions.h"
#include "scheduler.h"
#include "space_getsid.h"
#include "task.h"
#include "timers.h"

#ifdef WITH_CUDA
#include "cuda/cuda_config.h"
#include "cuda/gpu_launch.h"
#include "cuda/gpu_offload_data.h"

#include <cuda.h>
#include <cuda_runtime.h>
#else
#endif

#ifdef WITH_HIP
#include "hip/gpu_runner_functions.h"
#include "hip/hip_config.h"
#endif

/**
 * @brief recurse into a pair of cells and recusrively identify all cell-cell
 * interactions.
 *
 * @param r The #runner
 * @param s The #scheduler
 * @param buf the data buffers
 * @param ci the first #cell to be interacted recursively
 * @param cj the second #cell to be interacted recursively
 * @param depth current recursion depth
 * @param timer are we timing this?
 */
static void runner_dopair_gpu_recurse(const struct runner *r,
                                      const struct scheduler *s,
                                      struct gpu_offload_data *restrict buf,
                                      struct cell *ci, struct cell *cj,
                                      const int depth, const char timer) {

  /* Note: Can't inline a recursive function... */

  TIMER_TIC;

  /* Should we even bother? */
  const struct engine *e = r->e;
  if (!cell_is_active_hydro(ci, e) && !cell_is_active_hydro(cj, e)) return;
  if (ci->hydro.count == 0 || cj->hydro.count == 0) return;

  /* Grab some handles. */
  /* packing data and metadata */
  struct gpu_pack_metadata *md = &buf->md;

  /* Arrays for leaf cells */
  struct cell **ci_leaves = md->ci_leaves;
  struct cell **cj_leaves = md->cj_leaves;

  /* Get the type of pair and flip ci/cj if needed. */
  double shift[3];
  const int sid = space_getsid_and_swap_cells(e->s, &ci, &cj, shift);

  /* Recurse? */
  if (cell_can_recurse_in_pair_hydro_task(ci) &&
      cell_can_recurse_in_pair_hydro_task(cj)) {

    struct cell_split_pair *csp = &cell_split_pairs[sid];

    for (int k = 0; k < csp->count; k++) {
      const int pid = csp->pairs[k].pid;
      const int pjd = csp->pairs[k].pjd;
      if (ci->progeny[pid] != NULL && cj->progeny[pjd] != NULL) {
        runner_dopair_gpu_recurse(r, s, buf, ci->progeny[pid], cj->progeny[pjd],
                                  depth + 1, /*timer=*/0);
      }
    }
  } else {

    /* At this point, we found leaves with work to do. Add them to list. */
    /* Note: We leave md->n_leaves unmodified during the recursion. So the
     * correct cell index will be md->n_leaves + how many new leaf cells we've
     * found for this task's recursion, which is stored in md->task_n_leaves. */
    const int ind = md->n_leaves + md->task_n_leaves;

    if (ind >= md->params.leaf_buffer_size) {
      error(
          "Found more leaf cells (%d) than expected (%d), depth=%i;\n"
          "Increase array size through Scheduler:gpu_recursion_max_depth",
          ind, md->params.leaf_buffer_size, depth);
    }

    ci_leaves[ind] = ci;
    cj_leaves[ind] = cj;

    /* Increment the counter. */
    md->task_n_leaves++;
  }

  if (timer) TIMER_TOC(timer_gpu_pair_recurse);
}

/**
 * @brief recurse into a cell and recursively identify all leaf cell
 * interactions needed in this step.
 *
 * @param r The #runner
 * @param s The #scheduler
 * @param buf the data buffers
 * @param ci the #cell to be interacted recursively
 * @param depth current recursion depth
 * @param timer are we timing this?
 */
static void runner_doself_gpu_recurse(const struct runner *r,
                                      const struct scheduler *s,
                                      struct gpu_offload_data *restrict buf,
                                      struct cell *ci, const int depth,
                                      const char timer) {

  /* Note: Can't inline a recursive function... */

  TIMER_TIC;

  /* Should we even bother? */
  const struct engine *e = r->e;
  if (!cell_is_active_hydro(ci, e)) return;
  if (ci->hydro.count == 0) return;

  /* Grab some handles. */
  /* packing data and metadata */
  struct gpu_pack_metadata *md = &buf->md;

  /* Arrays for leaf cells */
  struct cell **ci_leaves = md->ci_leaves;
  struct cell **cj_leaves = md->cj_leaves;

  /* Recurse? */
  if (cell_can_recurse_in_self_hydro_task(ci)) {

    for (int k = 0; k < 8; k++) {
      if (ci->progeny[k] != NULL) {
        runner_doself_gpu_recurse(r, s, buf, ci->progeny[k], depth + 1,
                                  /*timer=*/0);
        for (int j = k + 1; j < 8; j++) {
          if (ci->progeny[j] != NULL) {
            runner_dopair_gpu_recurse(r, s, buf, ci->progeny[k], ci->progeny[j],
                                      depth + 1, /*timer=*/0);
          }
        }
      }
    }

  } else {

    /* At this point, we found a leaf with work to do. Add it to list. */
    /* Note: We leave md->n_leaves unmodified during the recursion. So the
     * correct cell index will be md->n_leaves + how many new leaf cells we've
     * found for this task's recursion, which is stored in md->task_n_leaves. */
    const int ind = md->n_leaves + md->task_n_leaves;

    if (ind >= md->params.leaf_buffer_size) {
      error(
          "Found more leaf cells (%d) than expected (%d), depth=%i;\n"
          "Increase array size through Scheduler:gpu_recursion_max_depth",
          ind, md->params.leaf_buffer_size, depth);
    }

    /* Not a typo: Store same cell in ci and cj leaf arrays */
    ci_leaves[ind] = ci;
    cj_leaves[ind] = ci;

    /* Increment the counter. */
    md->task_n_leaves++;
  }

  if (timer) TIMER_TOC(timer_gpu_self_recurse);
}

/**
 * @brief recurse into a pair of cells found from self recursion and count
 * the number of cell-cell interactions needed for self tasks in this step.
 *
 * @param r The #runner
 * @param s The #scheduler
 * @param buf the data buffers
 * @param ci the first #cell to be interacted recursively
 * @param cj the second #cell to be interacted recursively
 * @param depth current recursion depth
 * @param timer are we timing this?
 */
static void runner_pair_recurse_and_test_active(const struct runner *r,
                                      const struct scheduler *s,
                                      struct gpu_offload_data *restrict buf,
                                      struct cell *ci, struct cell *cj,
                                      const int depth, const char timer) {

  /* Note: Can't inline a recursive function... */

  TIMER_TIC;

  /* Should we even bother? */
  const struct engine *e = r->e;
  if (!cell_is_active_hydro(ci, e) && !cell_is_active_hydro(cj, e)) return;
  if (ci->hydro.count == 0 || cj->hydro.count == 0) return;

  /* Grab some handles. */
  /* packing data and metadata */
  struct gpu_pack_metadata *md = &buf->md;

  /* Get the type of pair and flip ci/cj if needed. */
  double shift[3];
  const int sid = space_getsid_and_swap_cells(e->s, &ci, &cj, shift);

  /* Recurse? */
  if (cell_can_recurse_in_pair_hydro_task(ci) &&
      cell_can_recurse_in_pair_hydro_task(cj)) {

    struct cell_split_pair *csp = &cell_split_pairs[sid];

    for (int k = 0; k < csp->count; k++) {
      const int pid = csp->pairs[k].pid;
      const int pjd = csp->pairs[k].pjd;
      if (ci->progeny[pid] != NULL && cj->progeny[pjd] != NULL) {
        runner_pair_recurse_and_test_active(r, s, buf, ci->progeny[pid], cj->progeny[pjd],
                                  depth + 1, /*timer=*/0);
      }
    }
  } else {

    /* At this point, we found leaves with work to do. Add them to list. */
    /* Note: We leave md->n_leaves unmodified during the recursion. So the
     * correct cell index will be md->n_leaves + how many new leaf cells we've
     * found for this task's recursion, which is stored in md->task_n_leaves. */
    /* Increment the counter. */
    md->n_active_leaves++;
  }

  if (timer) TIMER_TOC(timer_gpu_pair_recurse);
}

/**
 * @brief recurse into a cell and recursively identify how many leaf cell
 * interactions needed in this step.
 *
 * @param r The #runner
 * @param s The #scheduler
 * @param buf the data buffers
 * @param ci the #cell to be interacted recursively
 * @param depth current recursion depth
 * @param timer are we timing this?
 */
static void runner_self_recurse_and_test_active(const struct runner *r,
                                      const struct scheduler *s,
                                      struct gpu_offload_data *restrict buf,
                                      struct cell *ci, const int depth,
                                      const char timer) {

  /* Note: Can't inline a recursive function... */

  TIMER_TIC;

  /* Should we even bother? */
  const struct engine *e = r->e;
  if (!cell_is_active_hydro_inc_tops(ci, e)) return;
  if (ci->hydro.count == 0) return;

  /* Grab some handles. */
  /* packing data and metadata */
  struct gpu_pack_metadata *md = &buf->md;

  /* Recurse? */
  if (cell_can_recurse_in_self_hydro_task(ci)) {

    for (int k = 0; k < 8; k++) {
      if (ci->progeny[k] != NULL) {
        runner_self_recurse_and_test_active(r, s, buf, ci->progeny[k], depth + 1,
                                  /*timer=*/0);
        for (int j = k + 1; j < 8; j++) {
          if (ci->progeny[j] != NULL) {
            runner_pair_recurse_and_test_active(r, s, buf, ci->progeny[k], ci->progeny[j],
                                      depth + 1, /*timer=*/0);
          }
        }
      }
    }

  } else {

    /* Increment the counter. */
    md->n_active_leaves++;
  }

  if (timer) TIMER_TOC(timer_gpu_self_recurse);
}

/**
 * @brief Decide if we will offload this time step.
 * Recurse through all local top level cells and see if we
 * have enough leaf level computations to offload.
 *
 * @param r The #runner
 * @param s The #scheduler
 * @param e The #engine
 * @param buf the data buffers
 * @param timer are we timing this?
 */
__attribute__((always_inline)) INLINE static int runner_GPU_offload_switch(const struct runner *r,
                                      const struct scheduler *s,
									  struct engine *e,
                                      struct gpu_offload_data *restrict buf,
                                      const char timer) {

//  TIMER_TIC;

  buf->md.n_active_leaves = 0;
  /*TODO: Need to time this separately to ensure we are not wasting too much time here*/
  for(int i = 0; i < e->s->nr_local_cells; i++){
	runner_self_recurse_and_test_active(r, s, buf, &e->s->cells_top[i], /*depth=*/0, /*timer=*/1);
  }
  /* We want to have at least one full pack of leaf computations per thread
   * If we do not have enough run on the CPU.
   * TODO: Need to check that this is a good estimate*/
  return buf->md.n_active_leaves > e->gpu_pack_params.pack_size * e->nr_threads;

//  if (timer) TIMER_TOC(timer_gpu_self_recurse);

}
/*TODO: Move all hash_table code into hash_cell_pointers.c or something*/
/* Simple hash function for pointers */
__attribute__((always_inline)) INLINE static int hash_func(const struct cell *ptr, const int hash_size) {
    return ((uintptr_t)ptr) % hash_size;
}

/* Insert into hash table. No need for probing as we will
 * only store one cell in each index of hash table*/
__attribute__((always_inline)) INLINE static void hash_insert(const struct cell *restrict c, const int unique_count, const int h_id, struct hash_entry * ht) {
    ht[h_id].c = (struct cell*)c;
    /*This is where the cell will be located in the unique_cells array*/
    ht[h_id].index = unique_count;
    ht[h_id].occupied = 1;
}

/* Lookup in hash table */
__attribute__((always_inline)) INLINE static void hash_lookup_and_pack(const struct cell *restrict c, const int hash_size,
        struct hash_entry *restrict ht, struct gpu_offload_data *restrict buf, const int ij,
        const enum task_subtypes task_subtype) {

  /*Get the hash using the cell's pointer address*/
  struct gpu_pack_metadata *md = &buf->md;
  int h_id = hash_func(c, hash_size);
  int start = h_id;
  const int n_leaves_packed = md->n_leaves_packed;
  int unique_count = md->n_unique;
  /*Do a linear probe of hash table
   * TODO: If this becomes a large overhead look into
   * optimising the hashing*/
  while(ht[h_id].occupied){
    /*If we already have a cell hashed to h_id.
     * Return it's index in the array of
     * unique cells*/
    if (ht[h_id].c == c){
      /*We found this cell's hash value exists -> Not unique.
       * The hash_lookup returns it's position in the sorted list
       * (not in the hash table)*/
      /*Check if this is ci*/
      if(ij == 0){
        md->my_index[n_leaves_packed].x = ht[h_id].index;
      }
      /*cell is cj*/
      else{
        md->my_index[n_leaves_packed].y = ht[h_id].index;
      }
      return;
    }
    //Add one to the cells pointer after converting to int and hash again
    h_id = (h_id + 1) % hash_size;
    if(h_id == start)
        error("hash table full");
  }
  if(h_id > hash_size)
      error("Ran over hash table");

  /*unique_cells is different from hash table.
   * This is just an array to keep track of
   * unique cells. Used for debugging but no longer necessary
   * TODO: Remove unique_cells if no longer needed*/
  md->unique_cells[unique_count] = (struct cell *)c;
  md->hash_table.count++;
  int c_count = c->hydro.count;
  /*Store where ci starts*/
  md->unique_start_end[unique_count].x = md->count_parts_unique;
  /*Store where ci ends*/
  md->unique_start_end[unique_count].y = md->count_parts_unique + c_count + 1;

  if(ij == 0){ /*This is ci and it is unique*/
    /*This cell has not been found yet.
     * Add to unique_cells and store it's index ascending
     * from index where we last inserted a unique cell*/
    md->my_index[n_leaves_packed].x = unique_count;
    /*Now pack the particles since this cell is unique*/
    if(task_subtype == task_subtype_gpu_density)
      gpu_pack_part_density(c, buf->parts_send_d, md->count_parts_unique);
    else if(task_subtype == task_subtype_gpu_gradient)
      gpu_pack_part_gradient(c, buf->parts_send_g, md->count_parts_unique);
    else if(task_subtype == task_subtype_gpu_force)
      gpu_pack_part_force(c, buf->parts_send_f, md->count_parts_unique);
    /*Add one as we have packed the cells position in index count_parts_unique + cii_count*/
    md->count_parts_unique += c_count + 1;
  }
  else{ /*This is cj and it is unique*/
    /*This cell has not been found yet.
     * Add to unique_cells and store it's index ascending
     * from index where we last inserted a unique cell*/
    md->my_index[n_leaves_packed].y = unique_count;
    /*Now pack the particles since this cell is unique*/
    if(task_subtype == task_subtype_gpu_density)
      gpu_pack_part_density(c, buf->parts_send_d, md->count_parts_unique);
    else if(task_subtype == task_subtype_gpu_gradient)
      gpu_pack_part_gradient(c, buf->parts_send_g, md->count_parts_unique);
    else if(task_subtype == task_subtype_gpu_force)
      gpu_pack_part_force(c, buf->parts_send_f, md->count_parts_unique);
    /*Add one as we have packed the cells position in index count_parts_unique + cii_count*/
    md->count_parts_unique += c_count + 1;
  }

  /*Store pointers for this unique cell, update it's unique index in array of unique cells*/
  hash_insert(c, unique_count, h_id, ht);
  md->n_unique++;

}

__attribute__((always_inline)) INLINE static void pack_cell_particles_in_unique_list(const struct runner *r,
                                      const struct scheduler *s,
                                      struct gpu_offload_data *restrict buf,
                                      const char timer, const struct task * t, const struct cell *restrict cii,
                                      const struct cell *restrict cjj, const enum task_subtypes task_subtype) {

#ifdef SWIFT_DEBUG_CHECKS
  if (cii == NULL) error("Got NULL cell ci?");
  if (cjj == NULL) error("Got NULL cell cj?");
#endif
  /* Grab some handles. */
  /* packing data and metadata */
  struct gpu_pack_metadata *md = &buf->md;
  const struct gpu_md *gpu_md = &buf->gpu_md;
  const int cii_count = cii->hydro.count;
  const int cjj_count = cjj->hydro.count;

#ifdef SWIFT_DEBUG_CHECKS
  /* Anything to do here? */
  if (cii_count == 0 || cjj_count == 0)
    error("Empty cells should've been excluded during the recursion.");
#endif

  /* Get how many particles we've packed until now */
  int pack_ind = md->count_parts_unique;

  int last_ind = pack_ind + cii_count;
  if (cii != cjj) last_ind += cjj_count; /* packing pair interaction */
  if (last_ind >= md->params.part_buffer_size) {
    error(
        "Exceeded particle buffer size. Increase "
        "Scheduler:gpu_part_buffer_size."
        "ind=%d, counts=%d %d, buffer_size=%ld, task_subtype=%s, is self "
        "task?=%d",
        pack_ind, cii_count, cjj_count, md->params.part_buffer_size,
        subtaskID_names[task_subtype], cii == cjj);
  }
  /*Figure out where cells start for controlling GPU computations*/
  /*How many blocks have we packed so far?
   * Each cell is split into count/BS chunks so that
   * multiple cuda blocks work on particles in each cell if cell is big enough*/
  const int n_blocks_packed = md->n_blocks_packed;
  /*How many blocks will the current cell be split into*/
  int n_blocks_current;
  if(cii == cjj){
	  n_blocks_current = (cii_count + GPU_THREAD_BLOCK_SIZE - 1)/GPU_THREAD_BLOCK_SIZE;
  }else{/*This is a pair task need to take the max count of ci and cj*/
	  n_blocks_current = (max(cii_count, cjj_count) + GPU_THREAD_BLOCK_SIZE - 1)/GPU_THREAD_BLOCK_SIZE;
  }

  /*Let the CUDA blocks know which parts of the data we send they need to work on*/
  for(int b = 0; b < n_blocks_current; b++){
	  /*Which leaf computation will this block (n_blocks_packed + b) work on?*/
	  gpu_md->block_leaf_id[n_blocks_packed + b].x = md->n_leaves_packed;
	  /*Save the id of the first block acting on this leaf comp.
	   * Needed for indexing in kernel*/
	  gpu_md->block_leaf_id[n_blocks_packed + b].y = n_blocks_packed;
  }

  /* Check to see we've not somehow gone over the number of blocks we allocated */
  /* TODO: Put in debug checks ifdef. Leave for now while dev'ing */
  /* TODO: Implement a way to check for this before we offload. The condition should be:
   * If: The next pack will exceed n_blocks_max, which also means that we exceed gpu_part_buffer_size
   * Then: Offload what we have packed and re-start packing.*/
  ///////////////////////////////////////////////////////////////////////
  int n_blocks_max = (md->params.part_buffer_size + GPU_THREAD_BLOCK_SIZE - 1)/GPU_THREAD_BLOCK_SIZE;
  md->n_blocks_packed += n_blocks_current;
  if(md->n_blocks_packed > n_blocks_max)
	  error("Exceeded n_block_max (gpu_part_buffer_size/GPU_THREAD_BLOCK_SIZE = %i). "
			  "n_blocks_current %i. n_blocks_packed %i. "
			  "Possibly due to cell(s) much larger than anticipated in a  deep heirarchy. "
			  "Increasing gpu_part_buffer_size in *.yml file could help", n_blocks_max,
			  n_blocks_current, md->n_blocks_packed);
  ///////////////////////////////////////////////////////////////////////

  /*Get a pointer to the full hash table and it's size
   * TODO: Make this a dynamically sized hash table
   * to use load factor to resize so that it is only ever 50% full*/
  struct hash_entry * ht = md->hash_table.entry;
  const int hash_size = md->hash_size;

  /*Check if ci has already been found.
   * If so, return where it's unique copy
   * is found in the hash table
   * Otherwise, add cell to hash table*/
  /*Flag that we're testing ci*/
  int ij = 0;
  hash_lookup_and_pack(cii, hash_size, ht, buf, ij, task_subtype);
  /*Same for cj. For self tasks this will point to ci's location*/
  /*Flag that we're testing cj*/
  ij = 1;
  hash_lookup_and_pack(cjj, hash_size, ht, buf, ij, task_subtype);

  /* Now finish up bookkeeping*/
  /* Update incremented pack length accordingly */
  if (cii == cjj) {
	  /* We packed a self interaction */
	  md->count_parts += cii_count;
  } else {
	  /* We packed a pair interaction */
	  md->count_parts += cii_count + cjj_count;
  }
  /* Record that we have now packed a new leaf cell (pair) & increment number
   * of leaf cells to offload */
  md->n_leaves_packed++;

}

/**
 * @brief Generic function to launch GPU computations: Copies CPU buffer data
 * asynchronously over to the GPU, calls the solver, then copies data back.
 *
 * @param r the #runner
 * @param buf struct holding buffer arrays
 * @param stream array of streams to use during offloading
 * @param d_a the current expansion scale factor
 * @param d_H the current Hubble constant
 * @param task_subtype the current task's subtype
 */
__attribute__((always_inline)) INLINE static void runner_gpu_launch(
    const struct runner *r, struct gpu_offload_data *restrict buf,
    cudaStream_t *stream, const float d_a, const float d_H,
    const enum task_subtypes task_subtype) {

  /* Grab handles */
  struct gpu_pack_metadata *md = &buf->md;

  /* How many leaves have we packed? */
  const int leaves_packed = md->n_leaves_packed;

  /*Send to GPU to have an idea of space size for periodics*/
  double3 space_dim;
  space_dim.x = r->e->s->dim[0];
  space_dim.y = r->e->s->dim[1];
  space_dim.z = r->e->s->dim[2];

  /*Grab a pointer to GPU destined metadata*/
  const struct gpu_md *gpu_md = &buf->gpu_md;

  /* initialise to just some meaningless value to silence the compiler */
  cudaError_t cu_error = cudaErrorMemoryAllocation;

  /*Copy the tasks' metadata to the GPU. Send it using stream[0] for now. N.B. this is not default stream ;).
   * TODO: Make this one asynchronous copy via events to stop kernel launch before this happens
   * NOTE: This could be removed outside conditionals since it's name, type and size are the same
   *  for all task subtypes*/
  cu_error =
      cudaMemcpyAsync(&buf->gpu_md.d_cell_i_j_start_end[0],
          &buf->gpu_md.cell_i_j_start_end[0],
          leaves_packed * sizeof(int4),
          cudaMemcpyHostToDevice, stream[0]);

  /*Copy the metadata to GPU telling each cuda block what sections of the unique particle data to work on.*/
  cu_error =
      cudaMemcpyAsync(&buf->gpu_md.d_block_leaf_id[0],
          &buf->gpu_md.block_leaf_id[0],
          md->n_blocks_packed * sizeof(int2),
          cudaMemcpyHostToDevice, stream[0]);

  /*Get the number of cuda blocks we need to launch. No need for calculation here as
   * this is calculated while we pack*/
  const int n_blocks = md->n_blocks_packed;

  /*TODO: Refactor this if at all possible*/
  if (task_subtype == task_subtype_gpu_density){

    /*"What's gone and what's past help. Should be past grief"
     *Re-set sums to zero on GPU before launching kernel*/
    cu_error =
        cudaMemsetAsync(&buf->d_parts_recv_d[0],
        0, md->count_parts_unique * sizeof(struct gpu_part_recv_d),
        stream[0]);

    if (cu_error != cudaSuccess) {
      /* If we're here, assume something's messed up with our code, not with
       * CUDA. */
      error(
          "CUDA memset: CUDA error '%s' for task_subtype %s: cpuid=%i ",
          cudaGetErrorString(cu_error), subtaskID_names[task_subtype], r->cpuid);
    }

    /*"Give to a gracious message a host of tongues"
     * Copy the unique particle data to the GPU*/
    cu_error =
        cudaMemcpyAsync(&buf->d_parts_send_d[0],
            &buf->parts_send_d[0],
            md->count_parts_unique * sizeof(struct gpu_part_send_d),
            cudaMemcpyHostToDevice, stream[0]);

    if (cu_error != cudaSuccess) {
      /* If we're here, assume something's messed up with our code, not with
       * CUDA. */
      error(
          "H2D memcpy: CUDA error '%s' for task_subtype %s: cpuid=%i ",
          cudaGetErrorString(cu_error), subtaskID_names[task_subtype], r->cpuid);
    }

    /*"Once more unto the breach dear friends, once more!"
     *Issue instruction to launch GPU computations*/
    gpu_launch_density(buf->d_parts_send_d, buf->d_parts_recv_d, d_a, d_H,
        n_blocks,
        gpu_md->d_cell_i_j_start_end,
        gpu_md->d_block_leaf_id, space_dim, stream[0]);

    /*"The wheel is come full circle; I am here"
     *  Results are ready to copy back to CPU BUFFERS */
    cu_error =
        cudaMemcpyAsync(&buf->parts_recv_d[0],
            &buf->d_parts_recv_d[0],
            md->count_parts_unique * sizeof(struct gpu_part_recv_d),
            cudaMemcpyDeviceToHost, stream[0]);
  }
  else if (task_subtype == task_subtype_gpu_gradient){

    /*"What's gone and what's past help. Should be past grief"
     *Re-set sums to zero on GPU before launching kernel*/
    cu_error =
        cudaMemsetAsync(&buf->d_parts_recv_g[0],
        0, md->count_parts_unique * sizeof(struct gpu_part_recv_g),
        stream[0]);
    if (cu_error != cudaSuccess) {
      /* If we're here, assume something's messed up with our code, not with
       * CUDA. */
      error(
          "CUDA memset: CUDA error '%s' for task_subtype %s: cpuid=%i ",
          cudaGetErrorString(cu_error), subtaskID_names[task_subtype], r->cpuid);
    }
      /*"Give to a gracious message a host of tongues"
       * Copy the unique particle data to the GPU*/
      cu_error =
          cudaMemcpyAsync(&buf->d_parts_send_g[0],
            &buf->parts_send_g[0],
            md->count_parts_unique * sizeof(struct gpu_part_send_g),
            cudaMemcpyHostToDevice, stream[0]);

      if (cu_error != cudaSuccess) {
        /* If we're here, assume something's messed up with our code, not with
         * CUDA. */
        error(
            "H2D memcpy pair: CUDA error '%s' for task_subtype %s: cpuid=%i ",
            cudaGetErrorString(cu_error), subtaskID_names[task_subtype], r->cpuid);
      }

      /*"Once more unto the breach dear friends, once more!"
       *Issue instruction to launch GPU computations*/
      gpu_launch_gradient(buf->d_parts_send_g, buf->d_parts_recv_g, d_a, d_H,
              n_blocks,
              gpu_md->d_cell_i_j_start_end,
              gpu_md->d_block_leaf_id, space_dim, stream[0]);

      /*"The wheel is come full circle; I am here"
       *  Results are ready to copy back to CPU BUFFERS */
      cu_error =
          cudaMemcpyAsync(&buf->parts_recv_g[0],
                          &buf->d_parts_recv_g[0],
                          md->count_parts_unique * sizeof(struct gpu_part_recv_g),
                          cudaMemcpyDeviceToHost, stream[0]);

  }
  else if (task_subtype == task_subtype_gpu_force){

    /*What's gone and what's past help. Should be past grief
     *Re-set sums to zero on GPU before launching kernel*/
    cu_error =
        cudaMemsetAsync(&buf->d_parts_recv_f[0],
        0, md->count_parts_unique * sizeof(struct gpu_part_recv_f),
        stream[0]);
    if (cu_error != cudaSuccess) {
      /* If we're here, assume something's messed up with our code, not with
       * CUDA. */
      error(
          "CUDA memset: CUDA error '%s' for task_subtype %s: cpuid=%i ",
          cudaGetErrorString(cu_error), subtaskID_names[task_subtype], r->cpuid);
    }

    /*"Give to a gracious message a host of tongues"
     *Copy the unique particle data to the GPU*/
    cu_error =
        cudaMemcpyAsync(&buf->d_parts_send_f[0],
            &buf->parts_send_f[0],
            md->count_parts_unique * sizeof(struct gpu_part_send_f),
            cudaMemcpyHostToDevice, stream[0]);

    if (cu_error != cudaSuccess) {
      /* If we're here, assume something's messed up with our code, not with
       * CUDA. */
      error(
          "H2D memcpy pair: CUDA error '%s' for task_subtype %s: cpuid=%i ",
          cudaGetErrorString(cu_error), subtaskID_names[task_subtype], r->cpuid);
    }

    /*"Once more unto the breach dear friends, once more!"
     *Issue instruction to launch GPU computations*/
    gpu_launch_force(buf->d_parts_send_f, buf->d_parts_recv_f, d_a, d_H,
        n_blocks,
        gpu_md->d_cell_i_j_start_end,
        gpu_md->d_block_leaf_id, space_dim, stream[0]);

    /*"The wheel is come full circle; I am here."
     * Copy results back to CPU BUFFERS */
    cu_error =
        cudaMemcpyAsync(&buf->parts_recv_f[0],
            &buf->d_parts_recv_f[0],
            md->count_parts_unique * sizeof(struct gpu_part_recv_f),
            cudaMemcpyDeviceToHost, stream[0]);
    if (cu_error != cudaSuccess) {
      /* If we're here, assume something's messed up with our code, not with
       * CUDA. */
      error(
          "D2H memcpy: CUDA error '%s' for task_subtype %s: cpuid=%i ",
          cudaGetErrorString(cu_error), subtaskID_names[task_subtype], r->cpuid);
    }
  }
#ifdef SWIFT_DEBUG_CHECKS
  else {
    error("Unknown GPU task subtype %s", subtaskID_names[task_subtype]);
  }
#endif
  /*"All things are ready, if our mind be so..."
   * Make sure CPU has synchronised with GPU before moving on*/
  cu_error =
      cudaStreamSynchronize(stream[0]);

  if (cu_error != cudaSuccess) {
    /* If we're here, assume something's messed up with our code, not with
     * CUDA. */
    error(
        "Stream synchronize: CUDA error '%s' for task_subtype %s: cpuid=%i ",
        cudaGetErrorString(cu_error), subtaskID_names[task_subtype], r->cpuid);
  }

  /*Check to see if the kernel returned any errors.
   * If we get here without crashing due to "error" call
   * Then the error is not related to memcpys or memsets*/
  cu_error = cudaGetLastError();
  if (cu_error != cudaSuccess) {
    /* If we're here, assume something's messed up with our code, not with
     * CUDA. */
    error(
        "kernel launch: CUDA error '%s' for task_subtype %s: cpuid=%i "
        "n_blocks=%i",
        cudaGetErrorString(cu_error), subtaskID_names[task_subtype], r->cpuid,
        n_blocks);
  }

}

__attribute__((always_inline)) INLINE static void finalise_gpu_metadata(
    struct gpu_offload_data *restrict buf) {

  const struct gpu_pack_metadata *md = &buf->md;
  struct gpu_md *gpu_md = &buf->gpu_md;
  /*Use tic for packing and tic2 for launch timing*/
  TIMER_TIC;
  for(int i = 0; i < md->n_leaves_packed; i++){
	  int index_i = md->my_index[i].x;
	  int index_j = md->my_index[i].y;
	  gpu_md->cell_i_j_start_end[i].x = md->unique_start_end[index_i].x;
	  gpu_md->cell_i_j_start_end[i].y = md->unique_start_end[index_i].y;
	  gpu_md->cell_i_j_start_end[i].z = md->unique_start_end[index_j].x;
	  gpu_md->cell_i_j_start_end[i].w = md->unique_start_end[index_j].y;
  }
  TIMER_TOC(timer_gpu_pack_f);
}

/**
 * @brief Wrapper to launch density tasks on the GPU.
 *
 * @param r the #runner
 * @param buf struct holding buffer arrays
 * @param stream array of streams to use during offloading
 * @param d_a the current expansion scale factor
 * @param d_H the current Hubble constant
 */
__attribute__((always_inline)) INLINE static void runner_gpu_launch_density(
    const struct runner *r, struct gpu_offload_data *restrict buf,
    cudaStream_t *stream, const float d_a, const float d_H) {

  /*Expand the start/end list from only the unique_start_end array into
   * the full metadata required by GPU threads.
   * This counts as packing but can only be done once we're ready to launch*/
  finalise_gpu_metadata(buf);

  TIMER_TIC;

  runner_gpu_launch(r, buf, stream, d_a, d_H, task_subtype_gpu_density);

  TIMER_TOC(timer_gpu_launch_d);
}

/**
 * @brief Wrapper to launch gradient tasks on the GPU.
 *
 * @param r the #runner
 * @param buf struct holding buffer arrays
 * @param stream array of streams to use during offloading
 * @param d_a the current expansion scale factor
 * @param d_H the current Hubble constant
 */
__attribute__((always_inline)) INLINE static void runner_gpu_launch_gradient(
    const struct runner *r, struct gpu_offload_data *restrict buf,
    cudaStream_t *stream, const float d_a, const float d_H) {

  /*Expand the start/end list from only the unique_start_end array into
    * the full metadata required by GPU threads.
	* This counts as packing but can only be done once we're ready to launch*/
  finalise_gpu_metadata(buf);

  TIMER_TIC;

  runner_gpu_launch(r, buf, stream, d_a, d_H, task_subtype_gpu_gradient);

  TIMER_TOC(timer_gpu_launch_g);
}

/**
 * @brief Wrapper to launch force tasks on the GPU.
 *
 * @param r the #runner
 * @param buf struct holding buffer arrays
 * @param stream array of streams to use during offloading
 * @param d_a the current expansion scale factor
 * @param d_H the current Hubble constant
 */
__attribute__((always_inline)) INLINE static void runner_gpu_launch_force(
    const struct runner *r, struct gpu_offload_data *restrict buf,
    cudaStream_t *stream, const float d_a, const float d_H) {

  /*Expand the start/end list from only the unique_start_end array into
    * the full metadata required by GPU threads.
	* This counts as packing but can only be done once we're ready to launch*/
  finalise_gpu_metadata(buf);

  TIMER_TIC;

  runner_gpu_launch(r, buf, stream, d_a, d_H, task_subtype_gpu_force);

  TIMER_TOC(timer_gpu_launch_f);
}

/**
 * @brief Generic function to pack tasks's data and launch them on the device
 * depending on the task subtype.
 *
 * @param r the #runner
 * @param s the #scheduler
 * @param buf the buffer to use for offloading
 * @param t the #task
 * @param stream array of cuda streams to use for offloading
 * @param d_a current expansion scale factor
 * @param d_H current Hubble constant
 */
__attribute__((always_inline)) INLINE static void runner_gpu_pack_and_launch(
    const struct runner *r, struct scheduler *s,
    struct gpu_offload_data *restrict buf, struct task *t, cudaStream_t *stream,
    const float d_a, const float d_H) {

  /* Grab handles */
  struct gpu_pack_metadata *md = &buf->md;
  int *task_first_packed_leaf = md->task_first_packed_leaf;
  int *task_last_packed_leaf = md->task_last_packed_leaf;
  t->signalled = 0;
  /* Should we early exit? */
  if ((md->task_n_leaves == 0) &&
      (!md->launch_leftovers ||
       (md->launch_leftovers && md->n_leaves_packed == 0))) {
    /* Exception-handle the case where we found an active task with no cells
     * that need interacting. This can happen for a pair task when either one
     * or both cells involved are active, but their respective leaf cells are
     * too far apart to actually interact with each other.
     * If we're not launching leftovers, we can just skip doing any work on
     * this task. So just mark it as completed. However, if we're unlucky and
     * have to launch leftovers currently stored in the buffer, they may never
     * get launched otherwise. So only early-exit if we're not launching
     * leftovers too.
     * The third option to skip launching is if we're launching a single
     * leftover task that has no interacting cells and we have no previously
     * packed tasks. Not exception-handling this scenario leads to errors down
     * the line (e.g. bundle_size = 0). */

#ifdef SWIFT_DEBUG_CHECKS
    /* This could be benign if I failed to account for some weird scenario, but
     * an active self task should have at least one active self cell to do work
     * on. */
    if (t->type == task_type_self)
      error("Found active self task with zero interactions.");
#endif

    /* Unlock this task's resources. */
    task_unlock(t);

    /* No work left for this task, close it up. */
    scheduler_enqueue_dependencies(s, t);

    /* Tell the scheduler's bookkeeping that this task is done */
    pthread_mutex_lock(&s->sleep_mutex);
    atomic_dec(&s->waiting);
    pthread_cond_broadcast(&s->sleep_cond);
    pthread_mutex_unlock(&s->sleep_mutex);

//    t->signalled = 1;

//    /* Mark the task as done. */
//    t->skip = 1;

    /* We're done here. */
    return;
  }

  /* Nr of super-level tasks we've accounted for in the meda-data arrays. */
  int tind = md->tasks_in_list;

#ifdef SWIFT_DEBUG_CHECKS
  /* Check whether we'll be writing out of bounds. Allow for the special case
   * where we find a task with zero leaf cell pair interactions */
  if (tind > md->params.pack_size ||
      ((tind == md->params.pack_size && md->task_n_leaves != 0)))
    error(
        "Runner %d: Writing out of task_list array bounds: "
        "%d/%d, n_leaves_packed=%d, task_n_leaves=%d",
        r->id, tind, md->params.pack_size, md->n_leaves_packed,
        md->task_n_leaves);
#endif

  /*TODO: Note: This was md->n_leaves not md->n_leaves_packed, come back to this after
   * editing the function to follow unique sorting algo*/
  /* Keep track of index of first leaf cell pairs in lists per super-level pair
   * task in case we are packing more than one super-level task into this
   * buffer. */
  task_first_packed_leaf[tind] = md->n_leaves_packed;
  /* Same for the first particle index in particle buffers */
  md->task_first_packed_part[tind] = md->count_parts;

  /* Get pointer to task. Needed to enqueue dependencies after we're done. */
  md->task_list[tind] = t;

  /* Increment how many tasks we've accounted for */
  md->tasks_in_list++;

#ifdef SWIFT_DEBUG_CHECKS
  /* At this point, md->n_leaves_packed and md->n_leaves should be identical.
   * They will diverge during the main offloading loop below, but if they
   * aren't equal here, then something's wrong with our bookkeeping. */
  if (md->n_leaves_packed != md->n_leaves)
    error("Found leaf_pairs_packed=%d and n_leaves=%d", md->n_leaves_packed,
          md->n_leaves);
#endif

  /* Update the total number of leaf interactions we found through the
   * recursion. */
  md->n_leaves += md->task_n_leaves;

  /* How many leaf cell interactions do we want to offload at once? */
  const int target_n_leaves = md->params.pack_size;

  /* Counter for how many leaf cells of this task we've packed */
  int npacked = 0;

  /* Is this task's cell data currently unlocked? It comes in locked, but we
   * may unlock it during the offloading. */
  char unlocked = 0;

  /* Do we have a task with no active cells, but are launching leftovers? */
  char launch_empty_task_leftovers =
      (md->task_n_leaves == 0) && md->launch_leftovers;

  /* Now we go on to pack the particle data into the buffers. If we find enough
   * data (leaf cell pairs) for an offload, we launch. If there are leaf cell
   * pairs to pack after the launch, we pack those too after the launch and
   * unpacking is complete. By the end, all data will have been packed and some
   * of it (possibly all of it) will have been solved on the GPU already. */

//  md->ive_been_robbed = 0;
  while ((npacked < md->task_n_leaves) || launch_empty_task_leftovers){// || md->ive_been_robbed) {

    /* We only need this for the first entry into the main loop. */
    launch_empty_task_leftovers = 0;

    /* Inside this loop, we're always working on the last task in our list. But
     * if we launch within this loop, we will shift data back to index 0
     * afterwards, so read the correct up-to-date task index here each time. */
    tind = md->tasks_in_list - 1;

    /* Lock this task's cell data. */
    if (unlocked) {
      /* Spin until you get the lock. */
      while (task_lock(t) != 1) {
      };
      unlocked = 0;
    }

#ifdef SWIFT_DEBUG_CHECKS
    if (md->n_leaves_packed >= md->params.leaf_buffer_size)
      error("Writing out of ci_leaves array bounds: %d/%d", md->n_leaves_packed,
            md->params.leaf_buffer_size);
#endif

    /* Grab handles. */
    struct cell *cii = md->ci_leaves[md->n_leaves_packed];
    struct cell *cjj = md->cj_leaves[md->n_leaves_packed];

    char launch_before_over_filling = 0;
    if (md->task_n_leaves > 0) {
#ifdef SWIFT_DEBUG_CHECKS
      if (cii->hydro.count == 0)
        error(
            "Found cell cii with particle count=0 during packing. "
            "It should have been excluded during the recursion.");
      if (cjj->hydro.count == 0)
        error(
            "Found cell cjj with particle count=0 during packing. "
            "It should have been excluded during the recursion.");
#endif
      TIMER_TIC;

      /* Check to see if we will go over packing limits in the next step
       * Necessary if cell heirarchy gets very deep and we have very large cells*/
      uint count_next = 0;
      /* Note we will have checked the task_n_leaves'th leaf computation to see
       * if it will overfill buffers before we start to pack so no issues here.
       * Also, we only want to check if we are at the penultimate leaf computation
       * before offloading pack_size computations*/
      if(npacked < md->task_n_leaves - 1 && npacked < md->params.pack_size - 1){
        struct cell *ci_next = md->ci_leaves[npacked + 1];
        struct cell *cj_next = md->cj_leaves[npacked + 1];
        count_next = max(ci_next->hydro.count, cj_next->hydro.count);
      }
      /*TODO: Replcae this with a member of struct md rather than calculate repeatedly*/
      int n_blocks_max = (md->params.part_buffer_size + GPU_THREAD_BLOCK_SIZE - 1)/GPU_THREAD_BLOCK_SIZE;
      /* Get how many blocks we have now */
      int n_blocks_current = md->n_blocks_packed + (max(cii->hydro.count, cjj->hydro.count)+ GPU_THREAD_BLOCK_SIZE - 1)/GPU_THREAD_BLOCK_SIZE;
      /* Get how many blocks we will end up with in the next packing cycle*/
      int n_blocks_next = n_blocks_current + (count_next + GPU_THREAD_BLOCK_SIZE - 1)/GPU_THREAD_BLOCK_SIZE;
      /* If we need to, raise the flag*/
      launch_before_over_filling = (n_blocks_next >= n_blocks_max) ? 1 : 0;
      /* Test to see if cells i and j have already been packed
       * cells i and j are the same cell for self tasks but use the same
       * function as for the pairs.
       * If cells are already packed, keep track of where
       * they're packed (index). If not, pack and store their index as unique*/
      /* Note that this increments md->count_parts, md->count_parts_unique and md->n_leaves_packed */
      pack_cell_particles_in_unique_list(r, s, buf, /*timer=*/1, t, cii, cjj, t->subtype);

      /*Record packing time*/
      if(t->subtype == task_subtype_gpu_density){
    	TIMER_TOC(timer_gpu_pack_d);
      }
      else if(t->subtype == task_subtype_gpu_gradient){
    	TIMER_TOC(timer_gpu_pack_g);
      }
      else if(t->subtype == task_subtype_gpu_force){
    	TIMER_TOC(timer_gpu_pack_f);
      }
#ifdef SWIFT_DEBUG_CHECKS
      else {
        error("Unknown task subtype %s", subtaskID_names[t->subtype]);
      }
#endif

      /* record how many leaves we've packed in total during this while loop */
      npacked++;
    }

    /* Update the current last leaf cell pair index of this task. */
    /* md->leaf_pairs_packed was incremented in runner_dopair_gpu_pack_<*>. */
    task_last_packed_leaf[tind] = md->n_leaves_packed;

    /* Can we launch? */
    if (md->n_leaves_packed == target_n_leaves) md->launch = 1;

//    const int qid = r->qid;
//    if(t->subtype == task_subtype_gpu_density){
//      if(s->queues[qid].gpu_tasks_left[gpu_task_type_hydro_density] == 0){
//        md->launch_leftovers = 1;
//      }
//    }
//    if(t->subtype == task_subtype_gpu_gradient){
//      if(s->queues[qid].gpu_tasks_left[gpu_task_type_hydro_gradient] == 0){
//        md->launch_leftovers = 1;
//      }
//    }
//    if(t->subtype == task_subtype_gpu_force){
//      if(s->queues[qid].gpu_tasks_left[gpu_task_type_hydro_force] == 0){
//        md->launch_leftovers = 1;
//      }
//    }

    /* Are we launching, or are we launching leftovers AND have packed all
     * remaining leaves or are we launching before exceeding buffer array size? */
    if (md->launch ||
        (md->launch_leftovers && (npacked == md->task_n_leaves)) || launch_before_over_filling) {

      /* Let others touch data while we're doing GPU computations.
       * We need this task's data released for the unpacking procedure: If
       * there is another task in the list we're offloading which requires data
       * of a cell that this current task is locking, then we'll deadlock. */
      task_unlock(t);
      /* Take note that we unlocked this task. */
      unlocked = 1;

      if (t->subtype == task_subtype_gpu_density) {

        /* Launch the GPU offload */
        runner_gpu_launch_density(r, buf, stream, d_a, d_H);

        /* Unpack the results into CPU memory */
        runner_gpu_unpack_density(r, s, buf, npacked);

      } else if (t->subtype == task_subtype_gpu_gradient) {

        /* Launch the GPU offload */
        runner_gpu_launch_gradient(r, buf, stream, d_a, d_H);

        /* Unpack the results into CPU memory */
        runner_gpu_unpack_gradient(r, s, buf, npacked);

      } else if (t->subtype == task_subtype_gpu_force) {

        /* Launch the GPU offload */
        runner_gpu_launch_force(r, buf, stream, d_a, d_H);

        /* Unpack the results into CPU memory */
        runner_dopair_gpu_unpack_force(r, s, buf, npacked);

      }
#ifdef SWIFT_DEBUG_CHECKS
      else {
        error("Unknown task subtype %s", subtaskID_names[t->subtype]);
      }
#endif

      if (npacked == md->task_n_leaves) {

        /* We have launched, finished all leaf cell pairs, and are done. */
        /* Reset all buffers and counters. */
        gpu_pack_metadata_reset(md, /*reset_leaves_lists=*/1);
        gpu_data_buffers_reset(buf);

      } else {

        /* We'll continue with this packing and offloading loop. Clear out the
         * buffers and the metadata, then fill in whatever's necessary for us
         * to continue. After this operation, there will be no packed data in
         * the buffers. The only thing we need is the list of leaf cells that
         * are yet to be packed and offloaded, and associated counters.*/

        /* Store this before it's gone. */
        /* How many leaves do we still need to go through? */
        int n_leaves_new = md->n_leaves - md->n_leaves_packed;
        /* How many leaves does this task have in total? */
        int task_n_leaves = md->task_n_leaves;
        /* Store launch_leftovers in case we still need to do that after we
         * finish packing all leaf cell pairs */
        char launch_leftovers = md->launch_leftovers;

        /* Shift the leaf cells down to index 0 in their arrays. */
        /* Reminder: md->n_leaves_packed is the current number of leaf pairs
         * in the buffers. md->n_leaves is the total number of leaf pairs we
         * have identified for offloading, including all of this task's leaves.
         */
        for (int i = md->n_leaves_packed; i < md->n_leaves; i++) {
          const int shift_ind = i - md->n_leaves_packed;
          md->ci_leaves[shift_ind] = md->ci_leaves[i];
          md->cj_leaves[shift_ind] = md->cj_leaves[i];
#ifdef SWIFT_DEBUG_CHECKS
          md->ci_leaves[i] = NULL;
          md->cj_leaves[i] = NULL;
#endif
        }

        /* Reset all the buffers and metadata. We still need the rest of the
         * leaf cells we found during the recursion but haven't packed yet, so
         * don't reset those. */
        gpu_pack_metadata_reset(md, /*reset_leaves_lists=*/0);
        gpu_data_buffers_reset(buf);

        /* Now fill in relevant data. At this point, nothing is packed. */
        md->launch = 0;
        md->launch_leftovers = launch_leftovers;
        md->task_list[0] = t;
        md->n_leaves = n_leaves_new;
        md->task_n_leaves = task_n_leaves;
        md->tasks_in_list = 1;

        /* Whatever the remaining leaf cells are, they will belong to the
         * current task. Any previous task will already have been offloaded. */
        task_first_packed_leaf[0] = 0; /* First index is now 0 */
        task_last_packed_leaf[0] = 0;  /* Nothing's packed yet. */
        /* Same for first particle index of task */
        md->task_first_packed_part[0] = 0;

      } /* Launched, but not finished packing */
    } /* if launch or launch_leftovers */
//    if(t->subtype == task_subtype_gpu_density && s->queues[qid].gpu_tasks_left[gpu_task_type_hydro_density] == 0
//        && npacked < md->task_n_leaves){
//      md->launch_leftovers = 1;
////      md->ive_been_robbed = 1;
//    }
////    else{
////      md->ive_been_robbed = 0;
////    }
//    if(t->subtype == task_subtype_gpu_gradient && s->queues[qid].gpu_tasks_left[gpu_task_type_hydro_gradient] == 0
//        && npacked < md->task_n_leaves){
//      md->launch_leftovers = 1;
////      md->ive_been_robbed = 1;
//    }
////    else{
////      md->ive_been_robbed = 0;
////    }
//    if(t->subtype == task_subtype_gpu_force && s->queues[qid].gpu_tasks_left[gpu_task_type_hydro_force] == 0
//        && npacked < md->task_n_leaves){
//      md->launch_leftovers = 1;
////      md->ive_been_robbed = 1;
//    }
//    else{
//      md->ive_been_robbed = 0;
//    }
  } /* while npacked < md->task_n_leaves */

  /* We're done with this task's data: Everything we'll need has been copied
   * into buffers. So we can release the cell locks now. */
  if (!unlocked) task_unlock(t);

  /* Tell the scheduler's bookkeeping that this task is done */
//  pthread_mutex_lock(&s->sleep_mutex);
//  atomic_dec(&s->waiting);
//  pthread_cond_broadcast(&s->sleep_cond);
//  pthread_mutex_unlock(&s->sleep_mutex);
//
//  if(s->waiting < 0)
//    error("s->waiting less than zero");

  /* Reset flags too. */
  md->ive_been_robbed = 0;
  md->launch_leftovers = 0;
  md->launch = 0;
}

/**
 * @brief Run the hydro density self tasks on GPU
 *
 * @param r the #runner
 * @param s the #scheduler
 * @param buf the buffer to use for offloading
 * @param t the #task
 * @param stream array of cuda streams to use for offloading
 * @param d_a current expansion scale factor
 * @param d_H current Hubble constant
 */
static void runner_doself_gpu_density(const struct runner *r,
                                      struct scheduler *s,
                                      struct gpu_offload_data *buf,
                                      struct task *t, cudaStream_t *stream,
                                      const float d_a, const float d_H) {

  /* Reset leaf cell counter for this task before we recurse down */
  buf->md.task_n_leaves = 0;

  /* Collect cell interaction data recursively*/
  runner_doself_gpu_recurse(r, s, buf, t->ci, /*depth=*/0, /*timer=*/1);

  /* Check to see if this is the last task in the queue. If so, set
   * launch_leftovers to 1 and pack and launch on GPU */
  unsigned int qid = r->qid;
  /* atomic_dec returns previously held value; So subtract 1 from it again */
//  while(lock_trylock(&s->queues[qid].lock) == 0);
  int count =
      atomic_dec(&s->queues[qid].gpu_tasks_left[gpu_task_type_hydro_density]) -
      1;
//  if (lock_unlock(&s->queues[qid].lock) != 0) error("Unlocking our queue failed");
  if (count < 1) buf->md.launch_leftovers = 1;

  /* pack the data and run, if enough data has been gathered */
  runner_gpu_pack_and_launch(r, s, buf, t, stream, d_a, d_H);
}

/**
 * @brief Run the hydro gradient self tasks on GPU
 *
 * @param r the #runner
 * @param s the #scheduler
 * @param buf the buffer to use for offloading
 * @param t the #task
 * @param stream array of cuda streams to use for offloading
 * @param d_a current expansion scale factor
 * @param d_H current Hubble constant
 */
static void runner_doself_gpu_gradient(const struct runner *r,
                                       struct scheduler *s,
                                       struct gpu_offload_data *buf,
                                       struct task *t, cudaStream_t *stream,
                                       const float d_a, const float d_H) {

  /* Reset leaf cell counter for this task before we recurse down */
  buf->md.task_n_leaves = 0;

  /* Pack the data. */
  runner_doself_gpu_recurse(r, s, buf, t->ci, /*depth=*/0, /*timer=*/1);

  /* Check to see if this is the last task in the queue. If so, set
   * launch_leftovers to 1 and pack and launch on GPU */
  unsigned int qid = r->qid;
  /* atomic_dec returns previously held value; So subtract 1 from it again */
//  while(lock_trylock(&s->queues[qid].lock) == 0);
  int count =
      atomic_dec(&s->queues[qid].gpu_tasks_left[gpu_task_type_hydro_gradient]) -
      1;
//  if (lock_unlock(&s->queues[qid].lock) != 0) error("Unlocking our queue failed");
  if (count < 1) buf->md.launch_leftovers = 1;

  /* pack the data and run, if enough data has been gathered */
  runner_gpu_pack_and_launch(r, s, buf, t, stream, d_a, d_H);
}

/**
 * @brief Run the hydro force self tasks on GPU
 *
 * @param r the #runner
 * @param s the #scheduler
 * @param buf the buffer to use for offloading
 * @param t the #task
 * @param stream array of cuda streams to use for offloading
 * @param d_a current expansion scale factor
 * @param d_H current Hubble constant
 */
static void runner_doself_gpu_force(const struct runner *r, struct scheduler *s,
                                    struct gpu_offload_data *buf,
                                    struct task *t, cudaStream_t *stream,
                                    const float d_a, const float d_H) {

  /* Reset leaf cell counter for this task before we recurse down */
  buf->md.task_n_leaves = 0;

  /* Collect cell interaction data recursively*/
  runner_doself_gpu_recurse(r, s, buf, t->ci, /*depth=*/0, /*timer=*/1);

  /* Check to see if this is the last task in the queue. If so, set
   * launch_leftovers to 1 and pack and launch on GPU */
  unsigned int qid = r->qid;
  /* atomic_dec returns previously held value; So subtract 1 from it again */
//  while(lock_trylock(&s->queues[qid].lock) == 0);
  int count =
      atomic_dec(&s->queues[qid].gpu_tasks_left[gpu_task_type_hydro_force]) - 1;
//  if (lock_unlock(&s->queues[qid].lock) != 0) error("Unlocking our queue failed");
  if (count < 1) buf->md.launch_leftovers = 1;
  /* pack the data and run, if enough data has been gathered */
  runner_gpu_pack_and_launch(r, s, buf, t, stream, d_a, d_H);

}

/**
 * @brief Top level runner function to solve hydro density pair tasks on GPU.
 *
 * @param r the #runner
 * @param s the #scheduler
 * @param ci first #cell to interact
 * @param cj second #cell to interact
 * @param buf struct holding buffer arrays
 * @param t the current task
 * @param stream array of streams to use during offloading
 * @param d_a the current expansion scale factor
 * @param d_H the current Hubble constant
 */
static void runner_dopair_gpu_density(const struct runner *r,
                                      struct scheduler *s, struct cell *ci,
                                      struct cell *cj,
                                      struct gpu_offload_data *restrict buf,
                                      struct task *t, cudaStream_t *stream,
                                      const float d_a, const float d_H) {

  /* Reset leaf cell counter for this task before we recurse down */
  buf->md.task_n_leaves = 0;

  /* Collect cell interaction data recursively*/
  runner_dopair_gpu_recurse(r, s, buf, ci, cj, /*depth=*/0, /*timer=*/1);

  /* Check to see if this is the last task in the queue. If so, set
   * launch_leftovers to 1 to pack and launch on GPU */
  unsigned int qid = r->qid;
  /* atomic_dec returns previously held value; So subtract 1 from it again */
//  while(lock_trylock(&s->queues[qid].lock) == 0);
  int count =
      atomic_dec(&s->queues[qid].gpu_tasks_left[gpu_task_type_hydro_density]) -
      1;
//  if (lock_unlock(&s->queues[qid].lock) != 0) error("Unlocking our queue failed");
  if (count < 1) buf->md.launch_leftovers = 1;

  /* pack the data and run, if enough data has been gathered */
  runner_gpu_pack_and_launch(r, s, buf, t, stream, d_a, d_H);
}

/**
 * @brief Top level runner function to solve hydro gradient pair tasks on GPU.
 *
 * @param r the #runner
 * @param s the #scheduler
 * @param ci first #cell to interact
 * @param cj second #cell to interact
 * @param buf struct holding buffer arrays
 * @param t the current task
 * @param stream array of streams to use during offloading
 * @param d_a the current expansion scale factor
 * @param d_H the current Hubble constant
 */
static void runner_dopair_gpu_gradient(const struct runner *r,
                                       struct scheduler *s, struct cell *ci,
                                       struct cell *cj,
                                       struct gpu_offload_data *restrict buf,
                                       struct task *t, cudaStream_t *stream,
                                       const float d_a, const float d_H) {

  /* Reset leaf cell counter for this task before we recurse down */
  buf->md.task_n_leaves = 0;

  /* Collect cell interaction data recursively */
  runner_dopair_gpu_recurse(r, s, buf, ci, cj, /*depth=*/0, /*timer=*/1);

  /* Check to see if this is the last task in the queue. If so, set
   * launch_leftovers to 1 to pack and launch on GPU */
  unsigned int qid = r->qid;
  /* atomic_dec returns previously held value; So subtract 1 from it again */
//  while(lock_trylock(&s->queues[qid].lock) == 0);
  int count =
      atomic_dec(&s->queues[qid].gpu_tasks_left[gpu_task_type_hydro_gradient]) -
      1;
//  if (lock_unlock(&s->queues[qid].lock) != 0) error("Unlocking our queue failed");
  if (count < 1) buf->md.launch_leftovers = 1;

  /* pack the data and run, if enough data has been gathered */
  runner_gpu_pack_and_launch(r, s, buf, t, stream, d_a, d_H);
}

/**
 * @brief Top level runner function to solve hydro force pair tasks on GPU.
 *
 * @param r the #runner
 * @param s the #scheduler
 * @param ci first #cell to interact
 * @param cj second #cell to interact
 * @param buf struct holding buffer arrays
 * @param t the current task
 * @param stream array of streams to use during offloading
 * @param d_a the current expansion scale factor
 * @param d_H the current Hubble constant
 */
static void runner_dopair_gpu_force(const struct runner *r, struct scheduler *s,
                                    struct cell *ci, struct cell *cj,
                                    struct gpu_offload_data *restrict buf,
                                    struct task *t, cudaStream_t *stream,
                                    const float d_a, const float d_H) {

  /* Reset leaf cell counter for this task before we recurse down */
  buf->md.task_n_leaves = 0;

  /* Collect cell interaction data recursively*/
  runner_dopair_gpu_recurse(r, s, buf, ci, cj, /*depth=*/0, /*timer=*/1);

  /* Check to see if this is the last task in the queue. If so, set
   * launch_leftovers to 1 to pack and launch on GPU */
  unsigned int qid = r->qid;
  /* atomic_dec returns previously held value; So subtract 1 from it again */
//  while(lock_trylock(&s->queues[qid].lock) == 0);
  int count =
      atomic_dec(&s->queues[qid].gpu_tasks_left[gpu_task_type_hydro_force]) - 1;
//  if (lock_unlock(&s->queues[qid].lock) != 0) error("Unlocking our queue failed");
  if (count < 1) buf->md.launch_leftovers = 1;

  /* pack the data and run, if enough data has been gathered */
  runner_gpu_pack_and_launch(r, s, buf, t, stream, d_a, d_H);
}

#ifdef __cplusplus
}
#endif

#endif /* RUNNER_GPU_PACK_FUNCTIONS_H */
