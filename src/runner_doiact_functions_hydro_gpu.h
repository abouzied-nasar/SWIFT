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

  if (timer) TIMER_TOC(timer_dopair_gpu_recurse);
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

  if (timer) TIMER_TOC(timer_doself_gpu_recurse);
}

bool is_unique(const struct cell *cii, struct cell *unique[], int unique_count) {
    for (int i = 0; i < unique_count; i++) {
        if (unique[i] == cii) {
            return false;
        }
    }
    return true;
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
static void runner_gpu_filter_data(const struct runner *r,
                                      const struct scheduler *s,
                                      struct gpu_offload_data *buf,
                                      const char timer) {

  /* Note: Can't inline a recursive function... */

  TIMER_TIC;

  /* Grab some handles. */
  /* packing data and metadata */
  struct gpu_pack_metadata *md = &buf->md;
  /* Arrays for leaf cells */
  struct cell **ci_leaves = md->ci_leaves;
  struct cell **cj_leaves = md->cj_leaves;

  int unique_count = 0;


  if(md->n_leaves == 0)
    return;

  /*TODO: Crude loop for now just to quickly bash something in to work.
   * Come back to this and optimise so we only loop to
   * uniqe_count not n_leaves in inner loop*/
  for(int i = 0; i < md->n_leaves; i++) {
    struct cell *cii = ci_leaves[i];
    struct cell *cjj = cj_leaves[i];
    md->pack_flags[i].x = 0;
    md->pack_flags[i].y = 0;
    if (cii == NULL || cjj == NULL)
      error("working on NULL cells");

    /*Test for ci*/
    int unique = 1;
    for (int j = 0; j < i; j++) {
      if (md->unique_cells[j] == cii) {
        buf->my_index[i].x = j;
        unique = 0;
        break;
      }
    }
    if (unique) {
      md->unique_cells[unique_count] = cii;
      buf->my_index[i].x = unique_count;
      md->pack_flags[i].x = 1;
      unique_count++;
    }
    /*Test for cj*/
    unique = 1;
    for (int j = 0; j < i; j++) {
      if (md->unique_cells[j] == cjj) {
        buf->my_index[i].y = j;
        unique = 0;
        break;
      }
    }
    if (unique) {
      md->unique_cells[unique_count] = cjj;
      buf->my_index[i].y = unique_count;
      md->pack_flags[i].y = 1;
      unique_count++;
    }
  }
//  for(int i = 0; i < md->n_leaves; i++){
//    message("cell i %i unique position is %i", i, buf->my_index[i].x);
//  }

//TODO: Add a debug check that the list is unique

  md->n_unique = unique_count;
//  message("found %i unique cells in %i leaf computations", unique_count, md->n_leaves);

  if (timer) TIMER_TOC(timer_doself_gpu_recurse);
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
  cudaEvent_t *event_end = buf->event_end;

  /* How many leaves have we packed? */
  const int leaves_packed = md->n_leaves_packed;

  /* How many leaves should be in a bundle? */
  const int bundle_size =
      md->is_pair_task ? md->params.bundle_size_pair : md->params.bundle_size;

  /* Identify the number of GPU bundles to run in ideal case */
  int n_bundles =
      md->is_pair_task ? md->params.n_bundles_pair : md->params.n_bundles;

  /* Special case for incomplete bundles (when having not enough leftover leafs
   * to fill a bundle) */
  if (md->launch_leftovers) {

    n_bundles = (leaves_packed + bundle_size - 1) / bundle_size;

#ifdef SWIFT_DEBUG_CHECKS
    int n_bundles_max =
        md->is_pair_task ? md->params.n_bundles_pair : md->params.n_bundles;

    if (n_bundles > n_bundles_max) {
      error("Launching leftovers with too many bundles? Target size=%d, got=%d",
            n_bundles_max, n_bundles);
    }
    if (n_bundles == 0) {
      error("Got 0 bundles. leaves_packed=%d, bundle_size=%d", leaves_packed,
            bundle_size);
    }
#endif
  }

  cudaError_t cu_error = cudaSuccess;
//  cudaEvent_t metadata_copied;
  /*Copy the tasks cell start/end metadata to the GPU. Send it using regular streams for now.
   * TODO: Make this one asynchronous copy via events to stop kernel launch before this happens
   * instead of n_bundle copies */
//  cu_error =
//      cudaMemcpyAsync(&buf->d_cell_i_j_start_end[0],
//                      &buf->cell_i_j_start_end[0],
//                      leaves_packed * sizeof(int4),
//                      cudaMemcpyHostToDevice, stream[0]);
//  if (task_subtype == task_subtype_gpu_density){
//    /*Create an event to say we have issue a send of this data to GPU*/
//    cudaEventCreateWithFlags(&metadata_copied, cudaEventDisableTiming);
//    cudaEventRecord(metadata_copied, stream[0]);
//  }
  /* Launch the copies for each bundle and run the GPU kernel. Each bundle gets
   * its own stream. */
  const struct gpu_md_and_cell_positions *gpu_md_cp = &buf->gpu_md_and_cell_positions;
  for (int bid = 0; bid < n_bundles; bid++) {

//    /* Get the particle count for this bundle */
    const int bundle_first_part = md->bundle_first_part[bid];
    const int bundle_last_part = bid < (n_bundles - 1)
                                     ? md->bundle_first_part[bid + 1]
                                     : md->count_parts;
    const int bundle_n_parts = bundle_last_part - bundle_first_part;
    /* Get the particle count for this bundle */
    const int bundle_first_cell = md->bundle_first_cell[bid];
    const int bundle_last_cell = bid < (n_bundles - 1)
                                     ? md->bundle_first_cell[bid + 1]
                                     : leaves_packed;
    const int bundle_n_cells = bundle_last_cell - bundle_first_cell;

    if (task_subtype == task_subtype_gpu_density) {
      /*Copy the tasks cell start/end metadata to the GPU. Send it using regular streams for now.
       * TODO: Make this one asynchronous copy via events to stop kernel launch before this happens
       * instead of n_bundle copies */
      cu_error =
          cudaMemcpyAsync(&gpu_md_cp->d_cell_i_j_start_end_non_compact[bundle_first_cell],
                          &gpu_md_cp->cell_i_j_start_end_non_compact[bundle_first_cell],
                          bundle_n_cells * sizeof(int4),
                          cudaMemcpyHostToDevice, stream[bid]);
      cu_error =
          cudaMemcpyAsync(&gpu_md_cp->d_cell_i_j_start_end[bundle_first_cell],
                          &gpu_md_cp->cell_i_j_start_end[bundle_first_cell],
                          bundle_n_cells * sizeof(int4),
                          cudaMemcpyHostToDevice, stream[bid]);

      /* Transfer particle data to device */
      cu_error =
          cudaMemcpyAsync(&buf->d_parts_send_d[bundle_first_part],
                          &buf->parts_send_d[bundle_first_part],
                          bundle_n_parts * sizeof(struct gpu_part_send_d),
                          cudaMemcpyHostToDevice, stream[bid]);

    } else if (task_subtype == task_subtype_gpu_gradient) {

      cu_error =
          cudaMemcpyAsync(&buf->d_parts_send_g[bundle_first_part],
                          &buf->parts_send_g[bundle_first_part],
                          bundle_n_parts * sizeof(struct gpu_part_send_g),
                          cudaMemcpyHostToDevice, stream[bid]);

    } else if (task_subtype == task_subtype_gpu_force) {

      cu_error =
          cudaMemcpyAsync(&buf->d_parts_send_f[bundle_first_part],
                          &buf->parts_send_f[bundle_first_part],
                          bundle_n_parts * sizeof(struct gpu_part_send_f),
                          cudaMemcpyHostToDevice, stream[bid]);
    }
#ifdef SWIFT_DEBUG_CHECKS
    else {
      error("Unknown task subtype %s", subtaskID_names[task_subtype]);
    }
#endif

    if (cu_error != cudaSuccess) {
      /* If we're here, assume something's messed up with our code, not with
       * CUDA. */
      error(
          "H2D memcpy pair: CUDA error '%s' for task_subtype %s: cpuid=%i "
          "first_part=%d bundle_n_parts=%d",
          cudaGetErrorString(cu_error), subtaskID_names[task_subtype], r->cpuid,
          bundle_first_part, bundle_n_parts);
    }
    /* Launch the GPU kernels for ci & cj as a 1D grid */
    /* TODO: num_blocks_y is not used anymore. Purge it. */
    const int num_blocks_x =
        (bundle_n_parts + GPU_THREAD_BLOCK_SIZE - 1) / GPU_THREAD_BLOCK_SIZE;
    const int num_blocks_x_cells =
        (bundle_n_cells + GPU_THREAD_BLOCK_SIZE - 1) / GPU_THREAD_BLOCK_SIZE;
    const int num_blocks_y = 0;

    /* Launch the kernel for ci using data for ci and cj */
    if (task_subtype == task_subtype_gpu_density) {
      /*Tell the kernel to wait untim metadata copied*/
//      cudaStreamWaitEvent(stream[bid], metadata_copied, cudaEventWaitDefault);
      gpu_launch_density(buf->d_parts_send_d, buf->d_parts_recv_d, d_a, d_H,
                         stream[bid], num_blocks_x_cells, num_blocks_y,
                         bundle_first_part, bundle_n_parts, gpu_md_cp->d_cell_i_j_start_end,
                         gpu_md_cp->d_cell_i_j_start_end, bundle_first_cell, bundle_n_cells);
//      cudaEventDestroy(metadata_copied);

    } else if (task_subtype == task_subtype_gpu_gradient) {

      gpu_launch_gradient(buf->d_parts_send_g, buf->d_parts_recv_g, d_a, d_H,
                          stream[bid], num_blocks_x, num_blocks_y,
                          bundle_first_part, bundle_n_parts);

    } else if (task_subtype == task_subtype_gpu_force) {

      gpu_launch_force(buf->d_parts_send_f, buf->d_parts_recv_f, d_a, d_H,
                       stream[bid], num_blocks_x, num_blocks_y,
                       bundle_first_part, bundle_n_parts);
    }
#ifdef SWIFT_DEBUG_CHECKS
    else {
      error("Unknown task subtype %s", subtaskID_names[task_subtype]);
    }
#endif

    cu_error = cudaGetLastError();
    if (cu_error != cudaSuccess) {
      /* If we're here, assume something's messed up with our code, not with
       * CUDA. */
      error(
          "kernel launch pair: CUDA error '%s' for task_subtype %s: cpuid=%i "
          "nbx=%i nby=%i",
          cudaGetErrorString(cu_error), subtaskID_names[task_subtype], r->cpuid,
          num_blocks_x, num_blocks_y);
    }

    /* Copy results back to CPU BUFFERS */
    if (task_subtype == task_subtype_gpu_density) {

      cu_error =
          cudaMemcpyAsync(&buf->parts_recv_d[bundle_first_part],
                          &buf->d_parts_recv_d[bundle_first_part],
                          bundle_n_parts * sizeof(struct gpu_part_recv_d),
                          cudaMemcpyDeviceToHost, stream[bid]);

    } else if (task_subtype == task_subtype_gpu_gradient) {

      cu_error =
          cudaMemcpyAsync(&buf->parts_recv_g[bundle_first_part],
                          &buf->d_parts_recv_g[bundle_first_part],
                          bundle_n_parts * sizeof(struct gpu_part_recv_g),
                          cudaMemcpyDeviceToHost, stream[bid]);

    } else if (task_subtype == task_subtype_gpu_force) {

      cu_error =
          cudaMemcpyAsync(&buf->parts_recv_f[bundle_first_part],
                          &buf->d_parts_recv_f[bundle_first_part],
                          bundle_n_parts * sizeof(struct gpu_part_recv_f),
                          cudaMemcpyDeviceToHost, stream[bid]);
    }
#ifdef SWIFT_DEBUG_CHECKS
    else {
      error("Unknown task subtype %s", subtaskID_names[task_subtype]);
    }
#endif

    if (cu_error != cudaSuccess) {
      /* If we're here, something's messed up with our code, not with CUDA. */
      error("D2H async memcpy: CUDA error '%s' for task_subtype %s: cpuid=%i",
            cudaGetErrorString(cu_error), subtaskID_names[task_subtype],
            r->cpuid);
    }

    /* Issue event to be recorded by GPU after copy back to CPU */
    cu_error = cudaEventRecord(event_end[bid], stream[bid]);
    swift_assert(cu_error == cudaSuccess);

  } /* End of looping over bundles to launch in streams */

  /* Issue synchronisation commands for all events recorded by GPU
   * Should swap with one cuda Device Synchronise really if we decide to go
   * this way with unpacking done separately */
  /* TODO Abouzied: Is the comment above still appropriate? */
  for (int bid = 0; bid < n_bundles; bid++) {
    cu_error = cudaEventSynchronize(event_end[bid]);
    if (cu_error != cudaSuccess) {
      error(
          "cudaEventSynchronize failed: '%s' for task subtype %s,"
          " cpuid=%d, bundle=%d",
          cudaGetErrorString(cu_error), subtaskID_names[task_subtype], r->cpuid,
          bid);
    }
  }
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

  TIMER_TIC;

  runner_gpu_launch(r, buf, stream, d_a, d_H, task_subtype_gpu_density);

  if (buf->md.is_pair_task)
    TIMER_TOC(timer_dopair_gpu_launch_d);
  else
    TIMER_TOC(timer_doself_gpu_launch_d);
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

  TIMER_TIC;

  runner_gpu_launch(r, buf, stream, d_a, d_H, task_subtype_gpu_gradient);

  if (buf->md.is_pair_task)
    TIMER_TOC(timer_dopair_gpu_launch_g);
  else
    TIMER_TOC(timer_doself_gpu_launch_g);
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

  TIMER_TIC;

  runner_gpu_launch(r, buf, stream, d_a, d_H, task_subtype_gpu_force);

  if (buf->md.is_pair_task)
    TIMER_TOC(timer_dopair_gpu_launch_f);
  else
    TIMER_TOC(timer_doself_gpu_launch_f);
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

  /* Grab handles*/
  struct gpu_pack_metadata *md = &buf->md;
  int *task_first_packed_leaf = md->task_first_packed_leaf;
  int *task_last_packed_leaf = md->task_last_packed_leaf;

  /* Nr of super-level tasks we've accounted for in the meda-data arrays. */
  int tind = md->tasks_in_list;

#ifdef SWIFT_DEBUG_CHECKS
  int pack_size =
      md->is_pair_task ? md->params.pack_size_pair : md->params.pack_size;
  if (tind >= pack_size)
    error("Writing out of task_list array bounds: %d/%d, is pair task?=%d",
          tind, pack_size, md->is_pair_task);
#endif

  /* Keep track of index of first leaf cell pairs in lists per super-level pair
   * task in case we are packing more than one super-level task into this
   * buffer. Note that md->n_leaves has not been yet updated to contain this
   * task's leaf cell pair count.*/
  task_first_packed_leaf[tind] = md->n_leaves;
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
  const int target_n_leaves =
      md->is_pair_task ? md->params.pack_size_pair : md->params.pack_size;

  /* Counter for how many leaf cells  of this task we've packed */
  int npacked = 0;

  /* We do not need the lock on this cell as we are packing read-only data
   * unlock here but lock when unpacking ;)*/
  cell_unlocktree(t->ci);
  if (t->cj != NULL) cell_unlocktree(t->cj);

  /* Now we go on to pack the particle data into the buffers. If we find enough
   * data (leaf cell pairs) for an offload, we launch. If there are leaf cell
   * pairs to pack after the launch, we pack those too after the launch and
   * unpacking is complete. By the end, all data will have been packed and some
   * of it (possibly all of it) will have been solved on the GPU already. */
  //TODO: Look into getting rid of the while and replacing with a for loop
  while (npacked < md->task_n_leaves) {

    /* Inside this loop, we're always working on the last task in our list. But
     * if we launch within this loop, we will shift data back to index 0
     * afterwards, so read the correct up-to-date task index here each time. */
    tind = md->tasks_in_list - 1;

#ifdef SWIFT_DEBUG_CHECKS
    if (md->n_leaves_packed >= md->params.leaf_buffer_size)
      error("Writing out of ci_leaves array bounds: %d/%d", md->n_leaves_packed,
            md->params.leaf_buffer_size);
#endif

    /* Grab handles. */
    struct cell *cii = md->ci_leaves[md->n_leaves_packed];
    struct cell *cjj = md->cj_leaves[md->n_leaves_packed];

    int cii_count = cii->hydro.count;
    int cjj_count = cjj->hydro.count;

    /*Figure out where cells start for controlling GPU computations*/
    if(t->subtype == task_subtype_gpu_density){
      if(md->is_pair_task){
        /*Get indices for where we unpack to*/
        buf->cell_i_j_start_end_non_compact[md->n_leaves_packed].x = md->count_parts;
        buf->cell_i_j_start_end_non_compact[md->n_leaves_packed].y = md->count_parts + cii_count;
        buf->cell_i_j_start_end_non_compact[md->n_leaves_packed].z = md->count_parts + cii_count;
        buf->cell_i_j_start_end_non_compact[md->n_leaves_packed].w = md->count_parts + cii_count + cjj_count;

        /*Now figure out where to start from in the unique particle buffer*/
        /*Don't count my count. this is the start pos*/
        if(md->pack_flags[md->n_leaves_packed].x == 1){
          /*Store where ci starts*/
          buf->cell_i_j_start_end[md->n_leaves_packed].x = md->count_parts_unique;
          /*Store where ci ends*/
          buf->cell_i_j_start_end[md->n_leaves_packed].y = md->count_parts_unique + cii_count;
          gpu_pack_part_density(cii, buf->parts_send_d, md->count_parts_unique);
          md->count_parts_unique += cii_count;
        }
        else{
          /*Get the cell's index in the unique cell list*/
          int my_index_i = buf->my_index[md->n_leaves_packed].x;
          /*Store where ci starts in unique list*/
          buf->cell_i_j_start_end[md->n_leaves_packed].x = buf->cell_i_j_start_end[my_index_i].x;
          /*Store where ci ends in unique list*/
          buf->cell_i_j_start_end[md->n_leaves_packed].y = buf->cell_i_j_start_end[my_index_i].y;
        }
        if(md->pack_flags[md->n_leaves_packed].y == 1){
          /*Store where cj starts*/
          buf->cell_i_j_start_end[md->n_leaves_packed].z = md->count_parts_unique;
          /*Store where cj ends*/
          buf->cell_i_j_start_end[md->n_leaves_packed].w = md->count_parts_unique + cjj_count;
          gpu_pack_part_density(cjj, buf->parts_send_d, md->count_parts_unique);
          md->count_parts_unique += cjj_count;
        }
        else{
          /*Get the cell's index in the unique cell list*/
          int my_index_j = buf->my_index[md->n_leaves_packed].y;
          /*Store where cj starts*/
          buf->cell_i_j_start_end[md->n_leaves_packed].z = buf->cell_i_j_start_end[my_index_j].z;
          /*Store where ci starts*/
          buf->cell_i_j_start_end[md->n_leaves_packed].w = buf->cell_i_j_start_end[my_index_j].w;
        }
      }
      /*This is a self task but need to check that it is density*/
      else{
        buf->cell_i_j_start_end_non_compact[md->n_leaves_packed].x = md->count_parts;
        buf->cell_i_j_start_end_non_compact[md->n_leaves_packed].y = md->count_parts + cii_count;
        //TODO: Add a debug check in unpacking to make sure we never touch this!
        buf->cell_i_j_start_end_non_compact[md->n_leaves_packed].z = -1;
        buf->cell_i_j_start_end_non_compact[md->n_leaves_packed].w = -1;
        /*Now figure out where to start from in the unique particle buffer*/
        /*Don't count my count. this is the start pos*/
        if(md->pack_flags[md->n_leaves_packed].x == 1){
          /*Store where ci starts*/
          buf->cell_i_j_start_end[md->n_leaves_packed].x = md->count_parts_unique;
          /*Store where ci ends*/
          buf->cell_i_j_start_end[md->n_leaves_packed].y = md->count_parts_unique + cii_count;
          /*Store where ci starts*/
          buf->cell_i_j_start_end[md->n_leaves_packed].z = md->count_parts_unique;
          /*Store where ci ends*/
          buf->cell_i_j_start_end[md->n_leaves_packed].w = md->count_parts_unique + cii_count;
          gpu_pack_part_density(cii, buf->parts_send_d, md->count_parts_unique);
          md->count_parts_unique += cii_count;
        }
        else{
          /*Get the cell's index in the unique cell list*/
          int my_index_i = buf->my_index[md->n_leaves_packed].x;
          /*Store where ci starts in unique list*/
          buf->cell_i_j_start_end[md->n_leaves_packed].x = buf->cell_i_j_start_end[my_index_i].x;
          /*Store where ci ends in unique list*/
          buf->cell_i_j_start_end[md->n_leaves_packed].y = buf->cell_i_j_start_end[my_index_i].y;
        }
      }
      /* Now finish up the bookkeeping. */

      /* Get the index for the leaf cell */
      const int lid = md->n_leaves_packed;

      /*TODO: Do we still need this? We're now working with cells not particles*/
      /* Identify first particle for each bundle of tasks */
      const int bundle_size =
          md->is_pair_task ? md->params.bundle_size_pair : md->params.bundle_size;
      if (lid % bundle_size == 0) {
        int bid = lid / bundle_size;
        /* Store this before we increment md->count_parts */
        md->bundle_first_part[bid] = md->count_parts;
        /* Store this before we increment md->count_parts */
        md->bundle_first_cell[bid] = lid;
      }

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

    /* Pack the particle data */
    /* Note that this increments md->count_parts and md->n_leaves_packed */
    if (t->subtype == task_subtype_gpu_gradient) {
      runner_gpu_pack_gradient(r, buf, cii, cjj);
    } else if (t->subtype == task_subtype_gpu_force) {
      runner_gpu_pack_force(r, buf, cii, cjj);
    }

#ifdef SWIFT_DEBUG_CHECKS
//    else {
//      error("Unknown task subtype %s", subtaskID_names[t->subtype]);
//    }
#endif

    /* record how many leaves we've packed in total during this while loop */
    npacked++;

    /* Update the current last leaf cell pair index of this task. */
    /* md->leaf_pairs_packed was incremented in runner_dopair_gpu_pack_<*>. */
    task_last_packed_leaf[tind] = md->n_leaves_packed;

    /* Can we launch? */
    if (md->n_leaves_packed == target_n_leaves) md->launch = 1;

    /* Are we launching, or are we launching leftovers AND have packed all
     * remaining leaves? */
    if (md->launch ||
        (md->launch_leftovers && (npacked == md->task_n_leaves))) {

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
  } /* while npacked < md->task_n_leaves */

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
static void runner_doself_gpu_density(struct runner *r, struct scheduler *s,
                                      struct gpu_offload_data *buf,
                                      struct task *t, cudaStream_t *stream,
                                      const float d_a, const float d_H) {

  /* Reset leaf cell counter for this task before we recurse down */
  buf->md.task_n_leaves = 0;

  /* Collect cell interaction data recursively*/
  runner_doself_gpu_recurse(r, s, buf, t->ci, /*depth=*/0, /*timer=*/1);

  /* Find unique cells*/
  runner_gpu_filter_data(r, s, buf, /*timer=*/1);

  /* Check to see if this is the last task in the queue. If so, set
   * launch_leftovers to 1 and pack and launch on GPU */
  unsigned int qid = r->qid;
  lock_lock(&s->queues[qid].lock);
  s->queues[qid].n_packs_self_left_d--;
  if (s->queues[qid].n_packs_self_left_d < 1) buf->md.launch_leftovers = 1;
  (void)lock_unlock(&s->queues[qid].lock);

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
static void runner_doself_gpu_gradient(struct runner *r, struct scheduler *s,
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
  lock_lock(&s->queues[qid].lock);
  s->queues[qid].n_packs_self_left_g--;
  if (s->queues[qid].n_packs_self_left_g < 1) buf->md.launch_leftovers = 1;
  (void)lock_unlock(&s->queues[qid].lock);

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
static void runner_doself_gpu_force(struct runner *r, struct scheduler *s,
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
  lock_lock(&s->queues[qid].lock);
  s->queues[qid].n_packs_self_left_f--;
  if (s->queues[qid].n_packs_self_left_f < 1) buf->md.launch_leftovers = 1;
  (void)lock_unlock(&s->queues[qid].lock);

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
  lock_lock(&s->queues[qid].lock);
  s->queues[qid].n_packs_pair_left_d--;
  if (s->queues[qid].n_packs_pair_left_d < 1) buf->md.launch_leftovers = 1;
  (void)lock_unlock(&s->queues[qid].lock);

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
  lock_lock(&s->queues[qid].lock);
  s->queues[qid].n_packs_pair_left_g--;
  if (s->queues[qid].n_packs_pair_left_g < 1) buf->md.launch_leftovers = 1;
  (void)lock_unlock(&s->queues[qid].lock);

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
  lock_lock(&s->queues[qid].lock);
  s->queues[qid].n_packs_pair_left_f--;
  if (s->queues[qid].n_packs_pair_left_f < 1) buf->md.launch_leftovers = 1;
  (void)lock_unlock(&s->queues[qid].lock);

  /* pack the data and run, if enough data has been gathered */
  runner_gpu_pack_and_launch(r, s, buf, t, stream, d_a, d_H);
}

#ifdef __cplusplus
}
#endif

#endif /* RUNNER_GPU_PACK_FUNCTIONS_H */
