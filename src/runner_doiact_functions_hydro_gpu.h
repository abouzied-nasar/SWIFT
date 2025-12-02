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
 * @brief Generic function to pack self tasks.
 *
 * @param r The runner
 * @param s The scheduler
 * @param buf the gpu data buffers
 * @param ci the associated cell
 * @param t the associated task to pack
 * @param task_subtype this task's subtype
 */
__attribute__((always_inline)) INLINE static void runner_doself_gpu_pack(
    struct cell *ci, struct task *t, struct gpu_offload_data *restrict buf,
    const enum task_subtypes task_subtype) {

  /* Grab a hold of the packing buffers */
  struct gpu_pack_metadata *md = &(buf->md);

#ifdef SWIFT_DEBUG_CHECKS
  if (md->params.pack_size <= md->self_tasks_packed)
    error("Trying to write outside of array bounds");
  if (md->params.pack_size <= md->self_tasks_packed)
    error("Trying to write outside of array bounds");
  if (md->params.pack_size <= md->self_tasks_packed)
    error("Trying to write outside of array bounds");
#endif

  /* Place pointers to the task and cells packed in an array for use later
   * when unpacking after the GPU offload */
  int tasks_packed = md->self_tasks_packed;
  md->task_list[tasks_packed] = t;
  md->ci_list[tasks_packed] = ci;

  /* Identify index in particle arrays where this task starts */
  buf->self_task_first_last_part[tasks_packed].x = md->count_parts;

  /* Anything to do here? */
  int count = ci->hydro.count;
  if (count > 0) {
    /* TODO: we already did some bookkeeping above. Doing stuff conditionally
     * at this poin logic for empty or inactive cells. needs checking and
     * testing. */

    if (md->count_parts + count >= md->params.part_buffer_size) {
      error(
          "Exceeded part_buffer_size. count=%d buffer=%d;\n"
          "Make arrays bigger through Scheduler:gpu_part_buffer_size ",
          md->count_parts + count, md->params.part_buffer_size);
    }

    /* This re-arranges the particle data from cell->hydro->parts into a long
     * array of part structs */
    if (task_subtype == task_subtype_gpu_density) {
      gpu_pack_self_density(ci, buf);
    } else if (task_subtype == task_subtype_gpu_gradient) {
      gpu_pack_self_gradient(ci, buf);
    } else if (task_subtype == task_subtype_gpu_force) {
      gpu_pack_self_force(ci, buf);
    }
#ifdef SWIFT_DEBUG_CHECKS
    else {
      error("Unknown task subtype %s", subtaskID_names[task_subtype]);
    }
#endif

    /* Increment pack length accordingly */
    md->count_parts += count;
  }

  /* Identify the row in the array where this task ends (row id of its
     last particle)*/
  buf->self_task_first_last_part[tasks_packed].y = md->count_parts;

  /* Identify first particle index and first task index for each bundle of
   * tasks */
  const int bundle_size = md->params.bundle_size;
  if (tasks_packed % bundle_size == 0) {
    int bid = tasks_packed / bundle_size;
    md->bundle_first_part[bid] = buf->self_task_first_last_part[tasks_packed].x;
    md->bundle_first_leaf[bid] = tasks_packed;
  }

  /* Record that we have now done a packing (self) */
  t->done = 1;
  md->self_tasks_packed++;
  md->launch = 0;
  md->launch_leftovers = 0;

  /* Have we packed enough tasks to offload to GPU? */
  if (md->self_tasks_packed == md->params.pack_size) {
    md->launch = 1;
  }

  /* Release the cell. */
  /* TODO: We should be able to do this earlier, directly after the packing. */
  cell_unlocktree(ci);
}

/**
 * @brief packs the data required for the self density tasks.
 *
 * @param r The #runner
 * @param s The #scheduler
 * @param buf the gpu data buffers
 * @param ci the #cell to pack
 * @param t the #task to pack
 */
__attribute__((always_inline)) INLINE static void
runner_doself_gpu_pack_density(const struct runner *r, struct scheduler *s,
                               struct gpu_offload_data *restrict buf,
                               struct cell *ci, struct task *t) {

  TIMER_TIC;

  runner_doself_gpu_pack(ci, t, buf, t->subtype);

  /* Get a lock to the queue so we can safely decrement counter and check for
   * launch leftover condition*/
  int qid = r->qid;
  lock_lock(&s->queues[qid].lock);
  s->queues[qid].n_packs_self_left_d--;
  if (s->queues[qid].n_packs_self_left_d < 1) {
    buf->md.launch_leftovers = 1;
  }
  (void)lock_unlock(&s->queues[qid].lock);

  TIMER_TOC(timer_doself_gpu_pack_d);
}

/**
 * @brief packs the data required for the self gradient tasks.
 *
 * @param r The #runner
 * @param s The #scheduler
 * @param buf the gpu data buffers
 * @param ci the #cell to pack
 * @param t the #task to pack
 */
__attribute__((always_inline)) INLINE static void
runner_doself_gpu_pack_gradient(const struct runner *r, struct scheduler *s,
                                struct gpu_offload_data *restrict buf,
                                struct cell *ci, struct task *t) {

  TIMER_TIC;

  runner_doself_gpu_pack(ci, t, buf, t->subtype);

  /* Get a lock to the queue so we can safely decrement counter and check for
   * launch leftover condition*/
  int qid = r->qid;
  lock_lock(&s->queues[qid].lock);
  s->queues[qid].n_packs_self_left_g--;
  if (s->queues[qid].n_packs_self_left_g < 1) {
    buf->md.launch_leftovers = 1;
  }
  (void)lock_unlock(&s->queues[qid].lock);

  TIMER_TOC(timer_doself_gpu_pack_g);
}

/**
 * @brief packs the data required for the self force tasks.
 *
 * @param r The #runner
 * @param s The #scheduler
 * @param buf the gpu data buffers
 * @param ci the #cell to pack
 * @param t the #task to pack
 */
__attribute__((always_inline)) INLINE static void runner_doself_gpu_pack_force(
    const struct runner *r, struct scheduler *s,
    struct gpu_offload_data *restrict buf, struct cell *ci, struct task *t) {

  TIMER_TIC;

  runner_doself_gpu_pack(ci, t, buf, t->subtype);

  /* Get a lock to the queue so we can safely decrement counter and check for
   * launch leftover condition*/
  int qid = r->qid;
  lock_lock(&s->queues[qid].lock);
  s->queues[qid].n_packs_self_left_f--;
  if (s->queues[qid].n_packs_self_left_f < 1) {
    buf->md.launch_leftovers = 1;
  }
  (void)lock_unlock(&s->queues[qid].lock);

  TIMER_TOC(timer_doself_gpu_pack_f);
}

/**
 * @brief recurse into a (sub-)pair task and identify all cell-cell
 * interactions.
 *
 * @param r The #runner
 * @param s The #scheduler
 * @param buf the data buffers
 * @param ci the first #cell associated with this pair task
 * @param cj the second #cell associated with this pair task
 * @param t the pair #task to pack
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

  /* Arrays for daughter cells */
  struct cell **ci_leaves = md->ci_leaves;
  struct cell **cj_leaves = md->cj_leaves;

  if (depth == 0) {
    /* Reset leaf cell pair counter for this task before we recurse down */
    md->task_n_leaves = 0;
  }

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
  } else if (cell_is_active_hydro(ci, e) || cell_is_active_hydro(cj, e)) {

    /* if any cell empty: skip */
    if (ci->hydro.count == 0 || cj->hydro.count == 0) return;

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
 * @brief Generic function to pack GPU pair tasks' leaf cell pairs
 *
 * @param r the #runner
 * @param buf the CPU-side buffer to copy into
 * @param ci first leaf #cell
 * @param cj second leaf #cell
 * @param task_subtype this task's subtype
 */
__attribute__((always_inline)) INLINE static void runner_dopair_gpu_pack(
    const struct runner *r, struct gpu_offload_data *restrict buf,
    const struct cell *ci, const struct cell *cj,
    const enum task_subtypes task_subtype) {

  const int count_ci = ci->hydro.count;
  const int count_cj = cj->hydro.count;

#ifdef SWIFT_DEBUG_CHECKS
  /* Anything to do here? */
  if (count_ci == 0 || count_cj == 0)
    error("Empty cells should've been excluded during the recursion.");
#endif

  /* Grab handles */
  const struct engine *e = r->e;
  struct gpu_pack_metadata *md = &buf->md;

  /* Get the relative distance between the pairs and apply wrapping in case of
   * periodic boundary conditions */
  double shift[3] = {0.0, 0.0, 0.0};
  for (int k = 0; k < 3; k++) {
    if (cj->loc[k] - ci->loc[k] < -e->s->dim[k] * 0.5) {
      shift[k] = e->s->dim[k];
    } else if (cj->loc[k] - ci->loc[k] > e->s->dim[k] * 0.5) {
      shift[k] = -e->s->dim[k];
    }
  }

  /* Get the index for the leaf cell pair */
  const int lid = md->leaf_pairs_packed;

  /* Pack the data into the CPU-side buffers for offloading. */
  if (task_subtype == task_subtype_gpu_density) {
    gpu_pack_pair_density(buf, ci, cj, shift);
  } else if (task_subtype == task_subtype_gpu_gradient) {
    gpu_pack_pair_gradient(buf, ci, cj, shift);
  } else if (task_subtype == task_subtype_gpu_force) {
    gpu_pack_pair_force(buf, ci, cj, shift);
  }
#ifdef SWIFT_DEBUG_CHECKS
  else {
    error("Unknown task subtype %s", subtaskID_names[task_subtype]);
  }
#endif

  /* Identify first particle for each bundle of tasks */
  const int bundle_size = md->params.bundle_size_pair;
  if (lid % bundle_size == 0) {
    int bid = lid / bundle_size;
    /* Store this before we increment md->count_parts */
    md->bundle_first_part[bid] = md->count_parts;

    md->bundle_first_leaf[bid] = lid;
  }

  /* Update incremented pack length accordingly */
  md->count_parts += count_ci + count_cj;

  /* Record that we have now done a pair pack leaf cell pair & increment number
   * of leaf cell pairs to offload */
  md->leaf_pairs_packed++;
};

/**
 * @brief Wrapper to pack data for density pair tasks on the GPU.
 */
__attribute__((always_inline)) INLINE static void
runner_dopair_gpu_pack_density(const struct runner *r,
                               struct gpu_offload_data *restrict buf,
                               const struct cell *ci, const struct cell *cj) {

  TIMER_TIC;
  runner_dopair_gpu_pack(r, buf, ci, cj, task_subtype_gpu_density);
  TIMER_TOC(timer_dopair_gpu_pack_d);
}

/**
 * Wrapper to pack data for density pair tasks on the GPU.
 */
__attribute__((always_inline)) INLINE static void
runner_dopair_gpu_pack_gradient(const struct runner *r,
                                struct gpu_offload_data *restrict buf,
                                const struct cell *ci, const struct cell *cj) {

  TIMER_TIC;
  runner_dopair_gpu_pack(r, buf, ci, cj, task_subtype_gpu_gradient);
  TIMER_TOC(timer_dopair_gpu_pack_g);
}

/**
 * Wrapper to pack data for density pair tasks on the GPU.
 */
__attribute__((always_inline)) INLINE static void runner_dopair_gpu_pack_force(
    const struct runner *r, struct gpu_offload_data *restrict buf,
    const struct cell *ci, const struct cell *cj) {

  TIMER_TIC;
  runner_dopair_gpu_pack(r, buf, ci, cj, task_subtype_gpu_force);
  TIMER_TOC(timer_dopair_gpu_pack_f);
}

/**
 * @brief Solves self task on GPU: Copies data over to GPU and launches
 * appropriate kernels given the task_subtype.
 */
__attribute__((always_inline)) INLINE static void runner_doself_gpu_launch(
    const struct runner *r, struct gpu_offload_data *buf,
    const enum task_subtypes task_subtype, cudaStream_t *stream,
    const float d_a, const float d_H) {

  /* Grab metadata */
  struct gpu_pack_metadata *md = &buf->md;

  /* Identify the number of GPU bundles to run in ideal case */
  int n_bundles = md->params.n_bundles;

  /* How many tasks should be in a bundle? */
  const int bundle_size = md->params.bundle_size;

  /* How many tasks have we packed? */
  const int tasks_packed = md->self_tasks_packed;

  if (md->launch_leftovers) {
    /* Special case for incomplete bundles (when not having enough leftover
     * tasks to fill a bundle) */

    /* Compute how many bundles we actually have, rounding up */
    n_bundles = (tasks_packed + bundle_size - 1) / bundle_size;

#ifdef SWIFT_DEBUG_CHECKS
    if (tasks_packed == 0)
      error("zero tasks packed but somehow got into GPU loop");
    if (n_bundles <= 0) error("Found case with n_bundles <= 0");
#endif
  }

  /* Store how many bundles we'll need to unpack */
  md->n_bundles_unpack = n_bundles;

  /* Identify the last particle for each bundle of tasks */
  for (int bid = 0; bid < n_bundles - 1; bid++) {
    md->bundle_last_part[bid] = md->bundle_first_part[bid + 1];
  }
  md->bundle_last_part[n_bundles - 1] = md->count_parts;

  /* Launch the copies for each bundle and run the GPU kernel */
  /* We don't go into this loop if tasks_left_self == 1 as
   n_bundles will be zero DUHDUHDUHDUHHHHHH!!!!!*/
  for (int bid = 0; bid < n_bundles; bid++) {

    /* First, get the max number of parts per cell in the bundle. Used for
     * determining the number of GPU CUDA blocks*/
    int max_parts = 0;
    for (int tid = bid * bundle_size; tid < (bid + 1) * bundle_size; tid++) {
      if (tid < tasks_packed) {
        int count = buf->self_task_first_last_part[tid].y -
                    buf->self_task_first_last_part[tid].x;
        max_parts = max(max_parts, count);
      }
    }

    const int first_part = md->bundle_first_part[bid];
    const int bundle_n_parts = md->bundle_last_part[bid] - first_part;
    const int tasks_left = (bid == n_bundles - 1)
                               ? tasks_packed - (n_bundles - 1) * bundle_size
                               : bundle_size;

    /* Will launch a 2d grid of GPU thread blocks (number of tasks is
       the y dimension and max_parts is the x dimension */
    const int numBlocks_y = tasks_left;
    const int numBlocks_x =
        (max_parts + GPU_THREAD_BLOCK_SIZE - 1) / GPU_THREAD_BLOCK_SIZE;

    const int first_task = md->bundle_first_leaf[bid];
#ifdef SWIFT_DEBUG_CHECKS
    const int last_task = first_task + tasks_left;
    if (last_task > md->params.pack_size)
      error("Trying to access out-of-boundary in d_self_task_first_last_part");
    if (last_task > md->params.pack_size)
      error("Trying to access out-of-boundary in task_first_last_part");
#endif

    /* Copy data over to GPU */
    /* First the meta-data */
    cudaError_t cu_error = cudaMemcpyAsync(
        &buf->d_self_task_first_last_part[first_task],
        &buf->self_task_first_last_part[first_task], tasks_left * sizeof(int2),
        cudaMemcpyHostToDevice, stream[bid]);

    if (cu_error != cudaSuccess) {
      error(
          "H2D memcpy metadata self: CUDA error '%s' for task_sybtype %s: "
          "cpuid=%i, first_task=%d, size=%lu",
          cudaGetErrorString(cu_error), subtaskID_names[task_subtype], r->cpuid,
          first_task, tasks_left * sizeof(int2));
    }

    /* Now the actual particle data */
    if (task_subtype == task_subtype_gpu_density) {

      cu_error = cudaMemcpyAsync(
          &buf->d_parts_send_d[first_part], &buf->parts_send_d[first_part],
          bundle_n_parts * sizeof(struct gpu_part_send_d),
          cudaMemcpyHostToDevice, stream[bid]);

    } else if (task_subtype == task_subtype_gpu_gradient) {

      cu_error = cudaMemcpyAsync(
          &buf->d_parts_send_g[first_part], &buf->parts_send_g[first_part],
          bundle_n_parts * sizeof(struct gpu_part_send_g),
          cudaMemcpyHostToDevice, stream[bid]);

    } else if (task_subtype == task_subtype_gpu_force) {

      cu_error = cudaMemcpyAsync(
          &buf->d_parts_send_f[first_part], &buf->parts_send_f[first_part],
          bundle_n_parts * sizeof(struct gpu_part_send_f),
          cudaMemcpyHostToDevice, stream[bid]);
    }
#ifdef SWIFT_DEBUG_CHECKS
    else {
      error("Unknown task subtype %s", subtaskID_names[task_subtype]);
    }
#endif

    if (cu_error != cudaSuccess) {
      error("H2D memcpy self: CUDA error '%s' for task_subtype %s: cpuid=%i",
            cudaGetErrorString(cu_error), subtaskID_names[task_subtype],
            r->cpuid);
    }

    /* Launch the kernel */
    if (task_subtype == task_subtype_gpu_density) {

      gpu_launch_self_density(buf->d_parts_send_d, buf->d_parts_recv_d, d_a,
                              d_H, stream[bid], numBlocks_x, numBlocks_y,
                              first_task, buf->d_self_task_first_last_part);

    } else if (task_subtype == task_subtype_gpu_gradient) {

      gpu_launch_self_gradient(buf->d_parts_send_g, buf->d_parts_recv_g, d_a,
                               d_H, stream[bid], numBlocks_x, numBlocks_y,
                               first_task, buf->d_self_task_first_last_part);

    } else if (task_subtype == task_subtype_gpu_force) {

      gpu_launch_self_force(buf->d_parts_send_f, buf->d_parts_recv_f, d_a, d_H,
                            stream[bid], numBlocks_x, numBlocks_y, first_task,
                            buf->d_self_task_first_last_part);
    }
#ifdef SWIFT_DEBUG_CHECKS
    else {
      error("Unknown task subtype %s", subtaskID_names[task_subtype]);
    }
#endif

    cu_error = cudaGetLastError();
    if (cu_error != cudaSuccess) {
      error("kernel launch self: CUDA error '%s' for task_subtype %s: cpuid=%i",
            cudaGetErrorString(cu_error), subtaskID_names[task_subtype],
            r->cpuid);
    }

    /* Copy back */
    if (task_subtype == task_subtype_gpu_density) {

      cu_error = cudaMemcpyAsync(
          &buf->parts_recv_d[first_part], &buf->d_parts_recv_d[first_part],
          bundle_n_parts * sizeof(struct gpu_part_recv_d),
          cudaMemcpyDeviceToHost, stream[bid]);

    } else if (task_subtype == task_subtype_gpu_gradient) {

      cu_error = cudaMemcpyAsync(
          &buf->parts_recv_g[first_part], &buf->d_parts_recv_g[first_part],
          bundle_n_parts * sizeof(struct gpu_part_recv_g),
          cudaMemcpyDeviceToHost, stream[bid]);

    } else if (task_subtype == task_subtype_gpu_force) {

      cu_error = cudaMemcpyAsync(
          &buf->parts_recv_f[first_part], &buf->d_parts_recv_f[first_part],
          bundle_n_parts * sizeof(struct gpu_part_recv_f),
          cudaMemcpyDeviceToHost, stream[bid]);
    }
#ifdef SWIFT_DEBUG_CHECKS
    else {
      error("Unknown task subtype %s", subtaskID_names[task_subtype]);
    }
#endif

    if (cu_error != cudaSuccess) {
      error("D2H memcpy self: CUDA error '%s' in task_subtype %s: cpuid=%i",
            cudaGetErrorString(cu_error), subtaskID_names[task_subtype],
            r->cpuid);
    }

    /* Record this event */
    cu_error = cudaEventRecord(buf->event_end[bid], stream[bid]);
    swift_assert(cu_error == cudaSuccess);

  } /*End of looping over bundles to launch in streams*/
  /* Issue synchronisation commands for all events recorded by GPU
   * Should swap with one cuda Device Synchronise really if we decide to go
   * this way with unpacking done separately */
  /* TODO Abouzied: Is the comment above still appropriate? */
  for (int bid = 0; bid < n_bundles; bid++) {
    cudaError_t cu_error = cudaEventSynchronize(buf->event_end[bid]);
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
 * @brief Unpacks the completed self tasks for the corresponding task_subtype
 */
__attribute__((always_inline)) INLINE static void runner_doself_gpu_unpack(
    const struct runner *r, struct scheduler *s, struct gpu_offload_data *buf,
    const enum task_subtypes task_subtype) {

  const struct engine *e = r->e;

  /* Grab md */
  struct gpu_pack_metadata *md = &buf->md;

  /* How many bundles do we need to unpack? */
  const int n_bundles = md->n_bundles_unpack;

  /* How many tasks have we packed? */
  const int tasks_packed = md->self_tasks_packed;

  /* How many tasks should be in a bundle? */
  const int bundle_size = md->params.bundle_size;

  char *task_unpacked = malloc(tasks_packed * sizeof(char));
  for (int i = 0; i < tasks_packed; i++) task_unpacked[i] = 0;
  int ntasks_unpacked = 0;

  /* Copy the data back from the CPU thread-local buffers to the cells, bundle
   * by bundle */
  while (ntasks_unpacked < tasks_packed) {

    for (int bid = 0; bid < n_bundles; bid++) {

      /* Loop over tasks in bundle */
      for (int tid = bid * bundle_size;
           (tid < (bid + 1) * bundle_size) && (tid < tasks_packed); tid++) {

        /* Anything to do here? */
        if (task_unpacked[tid]) continue;

        struct cell *c = md->ci_list[tid];
        struct task *t = md->task_list[tid];
        const int count = c->hydro.count;
        const int unpack_index = buf->self_task_first_last_part[tid].x;

#ifdef SWIFT_DEBUG_CHECKS
        if (!cell_is_active_hydro(c, e))
          error("We packed an inactive cell for self tasks???");
#endif
        /* Anything to do here? */
        if (count == 0) continue;

        /* Can we get the lock? */
        if (cell_locktree(c) != 0) continue;

        /* We got it! Mark that. */
        task_unpacked[tid] = 1;
        ntasks_unpacked++;

        if (unpack_index + count >= md->params.part_buffer_size) {
          error(
              "Exceeded part_buffer_size. count=%d buffer=%d;\n"
              "Make arrays bigger through Scheduler:gpu_part_buffer_size",
              unpack_index + count, md->params.part_buffer_size);
        }

        /* Do the copy */
        if (task_subtype == task_subtype_gpu_density) {
          gpu_unpack_self_density(c, buf->parts_recv_d, unpack_index, count, e);
        } else if (task_subtype == task_subtype_gpu_gradient) {
          gpu_unpack_self_gradient(c, buf->parts_recv_g, unpack_index, count,
                                   e);
        } else if (task_subtype == task_subtype_gpu_force) {
          gpu_unpack_self_force(c, buf->parts_recv_f, unpack_index, count, e);
        }
#ifdef SWIFT_DEBUG_CHECKS
        else {
          error("Unknown task subtype %s", subtaskID_names[task_subtype]);
        }
#endif

        /* Release the lock */
        cell_unlocktree(c);

        /* schedule my dependencies */
        enqueue_dependencies(s, t);

        /* Tell the scheduler's bookkeeping that this task is done */
        pthread_mutex_lock(&s->sleep_mutex);
        atomic_dec(&s->waiting);
        pthread_cond_broadcast(&s->sleep_cond);
        pthread_mutex_unlock(&s->sleep_mutex);

        t->skip = 1;
        t->done = 1;

      } /* Loop over tasks in bundle */
    } /* Loop over bundles */
  }

  /* Clean up after yourself */
  free(task_unpacked);

  /* Zero counters and buffers for the next pack operations */
  gpu_pack_metadata_reset(md, /*reset_leaves_lists=*/1);
  gpu_data_buffers_reset(buf);
}

/**
 * @brief Run the hydro density self tasks on GPU
 */
static void runner_doself_gpu_density(struct runner *r, struct scheduler *s,
                                      struct gpu_offload_data *buf,
                                      struct task *t, cudaStream_t *stream,
                                      const float d_a, const float d_H) {

  /* Pack the data. */
  runner_doself_gpu_pack_density(r, s, buf, t->ci, t);

  /* No pack tasks left in queue, flag that we want to run */
  char launch_leftovers = buf->md.launch_leftovers;

  /* Packed enough tasks. Let's go*/
  char launch = buf->md.launch;

  /* Do we have enough stuff to run the GPU ? */
  if (launch || launch_leftovers) {
    TIMER_TIC;
    runner_doself_gpu_launch(r, buf, t->subtype, stream, d_a, d_H);
    TIMER_TOC(timer_doself_gpu_launch_d);

    TIMER_TIC2;
    runner_doself_gpu_unpack(r, s, buf, t->subtype);
    TIMER_TOC2(timer_doself_gpu_unpack_d);
  }
}

/**
 * @brief Run the hydro gradient self tasks on GPU
 */
static void runner_doself_gpu_gradient(struct runner *r, struct scheduler *s,
                                       struct gpu_offload_data *buf,
                                       struct task *t, cudaStream_t *stream,
                                       const float d_a, const float d_H) {

  /* Pack the data. */
  runner_doself_gpu_pack_gradient(r, s, buf, t->ci, t);

  /* No pack tasks left in queue, flag that we want to run */
  char launch_leftovers = buf->md.launch_leftovers;

  /* Packed enough tasks. Let's go*/
  char launch = buf->md.launch;

  /* Do we have enough stuff to run the GPU ? */
  if (launch || launch_leftovers) {
    TIMER_TIC;
    runner_doself_gpu_launch(r, buf, t->subtype, stream, d_a, d_H);
    TIMER_TOC(timer_doself_gpu_launch_g);

    TIMER_TIC2;
    runner_doself_gpu_unpack(r, s, buf, t->subtype);
    TIMER_TOC2(timer_doself_gpu_unpack_g);
  }
}

/*
 * @brief Run the hydro force self tasks on GPU
 */
static void runner_doself_gpu_force(struct runner *r, struct scheduler *s,
                                    struct gpu_offload_data *buf,
                                    struct task *t, cudaStream_t *stream,
                                    const float d_a, const float d_H) {

  /* Pack the particle data */
  runner_doself_gpu_pack_force(r, s, buf, t->ci, t);

  /* No pack tasks left in queue, flag that we want to run */
  char launch_leftovers = buf->md.launch_leftovers;

  /*Packed enough tasks let's go*/
  char launch = buf->md.launch;

  /* Do we have enough stuff to run the GPU ? */
  if (launch || launch_leftovers) {
    /*Launch GPU tasks*/
    TIMER_TIC;
    runner_doself_gpu_launch(r, buf, t->subtype, stream, d_a, d_H);
    TIMER_TOC(timer_doself_gpu_launch_f);

    TIMER_TIC2;
    runner_doself_gpu_unpack(r, s, buf, t->subtype);
    TIMER_TOC2(timer_doself_gpu_unpack_f);
  }
}

/**
 * @brief Generic function to launch GPU pair tasks: Copies CPU buffer data
 * asynchronously over to the GPU, calls the solver, then copies data back.
 *
 * @param r the #runner
 * @param s the #scheduler
 * @param ci first #cell to interact
 * @param cj second #cell to interact
 * @param buf struct holding buffer arrays
 * @param stream array of streams to use during offloading
 * @param d_a the current expansion scale factor
 * @param d_H the current Hubble constant
 * @param task_subtype the current task's subtype
 */
__attribute__((always_inline)) INLINE static void runner_dopair_gpu_launch(
    const struct runner *r, struct gpu_offload_data *restrict buf,
    cudaStream_t *stream, const float d_a, const float d_H,
    const enum task_subtypes task_subtype) {

  /* Grab handles */
  struct gpu_pack_metadata *md = &buf->md;
  cudaEvent_t *pair_end = buf->event_end;

  /* How many tasks have we packed? */
  const int leaves_packed = md->leaf_pairs_packed;

  /* How many tasks should be in a bundle? */
  const int bundle_size = md->params.bundle_size_pair;

  /* Identify the number of GPU bundles to run in ideal case */
  int n_bundles = md->params.n_bundles_pair;

  /* Special case for incomplete bundles (when having not enough leftover tasks
   * to fill a bundle) */
  if (md->launch_leftovers) {

    n_bundles = (leaves_packed + bundle_size - 1) / bundle_size;

#ifdef SWIFT_DEBUG_CHECKS
    if (n_bundles > md->params.n_bundles_pair) {
      error("Launching leftovers with too many bundles? Target size=%d, got=%d",
            md->params.n_bundles_pair, n_bundles);
    }
    if (n_bundles == 0) {
      error("Got 0 bundles. leaves_packed=%d, bundle_size=%d", leaves_packed,
            bundle_size);
    }
#endif
  }

  /* Identify the last particle for each bundle of tasks */
  for (int bid = 0; bid < n_bundles - 1; bid++) {
    md->bundle_last_part[bid] = md->bundle_first_part[bid + 1];
  }
  md->bundle_last_part[n_bundles - 1] = md->count_parts;

  /* Launch the copies for each bundle and run the GPU kernel. Each bundle gets
   * its own stream. */
  for (int bid = 0; bid < n_bundles; bid++) {

    const int first_part_i = md->bundle_first_part[bid];
    const int bundle_n_parts = md->bundle_last_part[bid] - first_part_i;
    /* initialise to just some meaningless value to silence the compiler */
    cudaError_t cu_error = cudaErrorMemoryAllocation;

    /* Transfer memory to device */
    if (task_subtype == task_subtype_gpu_density) {

      cu_error = cudaMemcpyAsync(
          &buf->d_parts_send_d[first_part_i], &buf->parts_send_d[first_part_i],
          bundle_n_parts * sizeof(struct gpu_part_send_d),
          cudaMemcpyHostToDevice, stream[bid]);

    } else if (task_subtype == task_subtype_gpu_gradient) {

      cu_error = cudaMemcpyAsync(
          &buf->d_parts_send_g[first_part_i], &buf->parts_send_g[first_part_i],
          bundle_n_parts * sizeof(struct gpu_part_send_g),
          cudaMemcpyHostToDevice, stream[bid]);

    } else if (task_subtype == task_subtype_gpu_force) {

      cu_error = cudaMemcpyAsync(
          &buf->d_parts_send_f[first_part_i], &buf->parts_send_f[first_part_i],
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
          first_part_i, bundle_n_parts);
    }

    /* Launch the GPU kernels for ci & cj as a 1D grid */
    /* TODO: num_blocks_y is not used anymore. Purge it. */
    const int num_blocks_x =
        (bundle_n_parts + GPU_THREAD_BLOCK_SIZE - 1) / GPU_THREAD_BLOCK_SIZE;
    const int num_blocks_y = 0;

    /* Launch the kernel for ci using data for ci and cj */
    if (task_subtype == task_subtype_gpu_density) {

      gpu_launch_pair_density(buf->d_parts_send_d, buf->d_parts_recv_d, d_a,
                              d_H, stream[bid], num_blocks_x, num_blocks_y,
                              first_part_i, bundle_n_parts);

    } else if (task_subtype == task_subtype_gpu_gradient) {

      gpu_launch_pair_gradient(buf->d_parts_send_g, buf->d_parts_recv_g, d_a,
                               d_H, stream[bid], num_blocks_x, num_blocks_y,
                               first_part_i, bundle_n_parts);

    } else if (task_subtype == task_subtype_gpu_force) {

      gpu_launch_pair_force(buf->d_parts_send_f, buf->d_parts_recv_f, d_a, d_H,
                            stream[bid], num_blocks_x, num_blocks_y,
                            first_part_i, bundle_n_parts);
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

      cu_error = cudaMemcpyAsync(
          &buf->parts_recv_d[first_part_i], &buf->d_parts_recv_d[first_part_i],
          bundle_n_parts * sizeof(struct gpu_part_recv_d),
          cudaMemcpyDeviceToHost, stream[bid]);

    } else if (task_subtype == task_subtype_gpu_gradient) {

      cu_error = cudaMemcpyAsync(
          &buf->parts_recv_g[first_part_i], &buf->d_parts_recv_g[first_part_i],
          bundle_n_parts * sizeof(struct gpu_part_recv_g),
          cudaMemcpyDeviceToHost, stream[bid]);

    } else if (task_subtype == task_subtype_gpu_force) {

      cu_error = cudaMemcpyAsync(
          &buf->parts_recv_f[first_part_i], &buf->d_parts_recv_f[first_part_i],
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
    cu_error = cudaEventRecord(pair_end[bid], stream[bid]);
    swift_assert(cu_error == cudaSuccess);

  } /* End of looping over bundles to launch in streams */

  /* Issue synchronisation commands for all events recorded by GPU
   * Should swap with one cuda Device Synchronise really if we decide to go
   * this way with unpacking done separately */
  /* TODO Abouzied: Is the comment above still appropriate? */
  for (int bid = 0; bid < n_bundles; bid++) {
    cudaError_t cu_error = cudaEventSynchronize(pair_end[bid]);
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
 * @brief Wrapper to launch density pair tasks on the GPU.
 */
__attribute__((always_inline)) INLINE static void
runner_dopair_gpu_launch_density(const struct runner *r,
                                 struct gpu_offload_data *restrict buf,
                                 cudaStream_t *stream, const float d_a,
                                 const float d_H) {

  TIMER_TIC;

  runner_dopair_gpu_launch(r, buf, stream, d_a, d_H, task_subtype_gpu_density);

  TIMER_TOC(timer_dopair_gpu_launch_d);
}

/**
 * @brief Wrapper to launch density pair tasks on the GPU.
 */
__attribute__((always_inline)) INLINE static void
runner_dopair_gpu_launch_gradient(const struct runner *r,
                                  struct gpu_offload_data *restrict buf,
                                  cudaStream_t *stream, const float d_a,
                                  const float d_H) {

  TIMER_TIC;

  runner_dopair_gpu_launch(r, buf, stream, d_a, d_H, task_subtype_gpu_gradient);

  TIMER_TOC(timer_dopair_gpu_launch_g);
}

/**
 * Wrapper to launch density pair tasks on the GPU.
 */
__attribute__((always_inline)) INLINE static void
runner_dopair_gpu_launch_force(const struct runner *r,
                               struct gpu_offload_data *restrict buf,
                               cudaStream_t *stream, const float d_a,
                               const float d_H) {

  TIMER_TIC;

  runner_dopair_gpu_launch(r, buf, stream, d_a, d_H, task_subtype_gpu_force);

  TIMER_TOC(timer_dopair_gpu_launch_f);
}

/**
 * @brief Generic function to unpack a GPU pair task depending on the task
 * subtype.
 *
 * @param r the #runner
 * @param s the #scheduler
 * @param buf the particle data buffers
 * @param npacked how many leaf cell pairs have been packed during the current
 * pair task offloading call. May differ from the total number of packed leaf
 * cell pairs if there have been leftover leaf cell pairs from a previous task.
 * @param task_subtype this task's subtype
 */
__attribute__((always_inline)) INLINE static void runner_dopair_gpu_unpack(
    const struct runner *r, struct scheduler *s,
    struct gpu_offload_data *restrict buf, const int npacked,
    const enum task_subtypes task_subtype) {

  /* Grab handles */
  struct gpu_pack_metadata *md = &buf->md;

  struct cell **ci_leaves = md->ci_leaves;
  struct cell **cj_leaves = md->cj_leaves;
  struct cell **ci_super = md->ci_super;
  struct cell **cj_super = md->cj_super;
  int **tflplp = md->task_first_last_packed_leaf_pair;

  /* Keep track which tasks we've unpacked already */
  char *task_unpacked = malloc(md->tasks_in_list * sizeof(char));
  for (int i = 0; i < md->tasks_in_list; i++) task_unpacked[i] = 0;
  int ntasks_unpacked = 0;

  while (ntasks_unpacked < md->tasks_in_list) {

    /* Loop over all tasks that we have offloaded */
    for (int tid = 0; tid < md->tasks_in_list; tid++) {

      /* Anything to do here? */
      if (task_unpacked[tid]) continue;

      /* Can we get the locks? */
      if (cell_locktree(ci_super[tid]) != 0) continue;
      if (cell_locktree(cj_super[tid]) != 0) {
        cell_unlocktree(ci_super[tid]);
        continue;
      }

      /* We got it! Mark that. */
      task_unpacked[tid] = 1;
      ntasks_unpacked++;

      /* Get the index in the particle buffer array where to read from */
      int unpack_index = md->task_first_part[tid];

      /* Loop through leaf cell pairs of this task by index */
      for (int lid = tflplp[tid][0]; lid < tflplp[tid][1]; lid++) {

        /*Get pointers to the leaf cells*/
        struct cell *cii_l = ci_leaves[lid];
        struct cell *cjj_l = cj_leaves[lid];

        /* Not a typo: task subtype is task_subtype_pack_*. The unpacking gets
         * called at the end of packing, running, and possibly launching. */
        /* Note that these calls increment pack_length_unpack. */
        if (task_subtype == task_subtype_gpu_density) {

          gpu_unpack_pair_density(r, cii_l, cjj_l, buf->parts_recv_d,
                                  &unpack_index, md->params.part_buffer_size);

        } else if (task_subtype == task_subtype_gpu_gradient) {

          gpu_unpack_pair_gradient(r, cii_l, cjj_l, buf->parts_recv_g,
                                   &unpack_index, md->params.part_buffer_size);

        } else if (task_subtype == task_subtype_gpu_force) {

          gpu_unpack_pair_force(r, cii_l, cjj_l, buf->parts_recv_f,
                                &unpack_index, md->params.part_buffer_size);

        }
#ifdef SWIFT_DEBUG_CHECKS
        else {
          error("Unknown task subtype %s", subtaskID_names[task_subtype]);
        }
#endif
      }

      /* Release the cells */
      cell_unlocktree(ci_super[tid]);
      cell_unlocktree(cj_super[tid]);

      /* If we haven't finished packing the currently handled task's leaf cells,
       * we mustn't unlock its dependencies yet. */
      if ((tid == md->tasks_in_list - 1) && (npacked != md->task_n_leaves)) {
        continue;
      }

      /* schedule my dependencies */
      enqueue_dependencies(s, md->task_list[tid]);

      /* Tell the scheduler's bookkeeping that this task is done */
      pthread_mutex_lock(&s->sleep_mutex);
      atomic_dec(&s->waiting);
      pthread_cond_broadcast(&s->sleep_cond);
      pthread_mutex_unlock(&s->sleep_mutex);

      md->task_list[tid]->skip = 1;
      md->task_list[tid]->done = 1;

    } /* Loop over tasks in list */
  }

  /* clean up after yourself */
  free(task_unpacked);
}

/**
 * @brief Wrapper for the density unpacking of pair tasks:
 * Provide the correct subtype to use and time the runtime separately
 *
 * @param r the #runner
 * @param s the #scheduler
 * @param buf the particle data buffers
 * @param npacked how many leaf cell pairs have been packed during the current
 * pair task offloading call. May differ from the total number of packed leaf
 * cell pairs if there have been leftover leaf cell pairs from a previous task.
 */
__attribute__((always_inline)) INLINE static void
runner_dopair_gpu_unpack_density(const struct runner *r, struct scheduler *s,
                                 struct gpu_offload_data *restrict buf,
                                 const int npacked) {

  TIMER_TIC;

  runner_dopair_gpu_unpack(r, s, buf, npacked, task_subtype_gpu_density);

  TIMER_TOC(timer_dopair_gpu_unpack_d);
}

/**
 * @brief Wrapper for the gradient unpacking of pair tasks:
 * Provide the correct subtype to use and time the runtime separately
 *
 * @param r the #runner
 * @param s the #scheduler
 * @param buf the particle data buffers
 * @param npacked how many leaf cell pairs have been packed during the current
 * pair task offloading call. May differ from the total number of packed leaf
 * cell pairs if there have been leftover leaf cell pairs from a previous task.
 */
__attribute__((always_inline)) INLINE static void
runner_dopair_gpu_unpack_gradient(const struct runner *r, struct scheduler *s,
                                  struct gpu_offload_data *restrict buf,
                                  const int npacked) {

  TIMER_TIC;

  runner_dopair_gpu_unpack(r, s, buf, npacked, task_subtype_gpu_gradient);

  TIMER_TOC(timer_dopair_gpu_unpack_g);
}

/**
 * @brief Wrapper for the force unpacking of pair tasks:
 * Provide the correct subtype to use and time the runtime separately
 *
 * @param r the #runner
 * @param s the #scheduler
 * @param buf the particle data buffers
 * @param npacked how many leaf cell pairs have been packed during the current
 * pair task offloading call. May differ from the total number of packed leaf
 * cell pairs if there have been leftover leaf cell pairs from a previous task.
 */
__attribute__((always_inline)) INLINE static void
runner_dopair_gpu_unpack_force(const struct runner *r, struct scheduler *s,
                               struct gpu_offload_data *restrict buf,
                               const int npacked) {

  TIMER_TIC;

  runner_dopair_gpu_unpack(r, s, buf, npacked, task_subtype_gpu_force);

  TIMER_TOC(timer_dopair_gpu_unpack_f);
}

/**
 * @brief Generic function to pack pair tasks and launch them on the device
 * depending on the task subtype.
 *
 * @param r the #runner
 * @param s the #scheduler
 * @param buf the buffer to use for offloading
 * @param ci first #cell involved in this pair task
 * @param cj second #cell involved in this pair task
 * @param t the #task
 * @param stream array of cuda streams to use for offloading
 * @param d_a current expansion scale factor
 * @param d_H current Hubble constant
 */
__attribute__((always_inline)) INLINE static void
runner_dopair_gpu_pack_and_launch(const struct runner *r, struct scheduler *s,
                                  struct cell *ci, struct cell *cj,
                                  struct gpu_offload_data *restrict buf,
                                  struct task *t, cudaStream_t *stream,
                                  const float d_a, const float d_H) {

  /* Grab handles */
  struct gpu_pack_metadata *md = &buf->md;
  int **tflplp = md->task_first_last_packed_leaf_pair;

  /* Nr of super-level tasks we've accounted for in the meda-data arrays. */
  int tind = md->tasks_in_list;

#ifdef SWIFT_DEBUG_CHECKS
  if (tind >= md->params.pack_size_pair)
    error("Writing out of top_task_packed array bounds: %d/%d", tind,
          md->params.pack_size_pair);
  if (tind >= md->params.pack_size_pair)
    error("Writing out of ci_top array bounds: %d/%d", tind,
          md->params.pack_size_pair);
  if (tind >= md->params.pack_size_pair)
    error("Writing out of cj_top array bounds: %d/%d", tind,
          md->params.pack_size_pair);
#endif

  /* Keep track of index of first leaf cell pairs in lists per super-level pair
   * task in case we are packing more than one super-level task into this
   * buffer. Note that md->n_leaves has not been yet updated to contain this
   * task's leaf cell pair count.*/
  tflplp[tind][0] = md->n_leaves;
  /* Same for the first particle index in particle buffers */
  md->task_first_part[tind] = md->count_parts;

  /* TODO: Do we need this? We already keep track of the task, and the
   * task has access to t->ci, t->cj */
  /* Same for super-level cells */
  md->ci_super[tind] = ci;
  md->cj_super[tind] = cj;

  /* Get pointer to task. Needed to enqueue dependencies after we're done. */
  md->task_list[tind] = t;

  /* Increment how many tasks we've accounted for */
  md->tasks_in_list++;

#ifdef SWIFT_DEBUG_CHECKS
  /* At this point, md->leaf_pairs_packed and md->n_leaves should be identical.
   * They will diverge during the main offloading loop below, but if they
   * aren't equal here, then something's wrong with our bookkeeping. */
  if (md->leaf_pairs_packed != md->n_leaves)
    error("Found leaf_pairs_packed=%d and n_leaves=%d", md->leaf_pairs_packed,
          md->n_leaves);
#endif

  /* Update the total number of leaf pair interactions we found through the
   * recursion. */
  md->n_leaves += md->task_n_leaves;

  /* How many leaf cell interactions do we want to offload at once? */
  const int target_n_leaves = md->params.pack_size_pair;

  /* Counter for how many leaf cell pairs of this task we've packed */
  int npacked = 0;

  /* TODO: @Abouzied Please document what is happening here, this looks very
   * important and scary. Why does this need to happen here, and not
   * earlier/later? */
  cell_unlocktree(ci);
  cell_unlocktree(cj);

  /* Now we go on to pack the particle data into the buffers. If we find enough
   * data (leaf cell pairs) for an offload, we launch. If there are leaf cell
   * pairs to pack after the launch, we pack those too after the launch and
   * unpacking is complete. By the end, all data will have been packed and some
   * of it (possibly all of it) will have been solved on the GPU already. */
  while (npacked < md->task_n_leaves) {

    /* Inside this loop, we're always working on the last task in our list. But
     * if we launch within this loop, we will shift data back to index 0
     * afterwards, so read the correct up-to-date task index here each loop. */
    tind = md->tasks_in_list - 1;

#ifdef SWIFT_DEBUG_CHECKS
    if (md->leaf_pairs_packed >= md->params.leaf_buffer_size)
      error("Writing out of ci_d array bounds: %d/%d", md->leaf_pairs_packed,
            md->params.leaf_buffer_size);
    if (md->leaf_pairs_packed >= md->params.leaf_buffer_size)
      error("Writing out of cj_d array bounds: %d/%d", md->leaf_pairs_packed,
            md->params.leaf_buffer_size);
#endif

    /* Grab handles. */
    struct cell *cii = md->ci_leaves[md->leaf_pairs_packed];
    struct cell *cjj = md->cj_leaves[md->leaf_pairs_packed];

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
    /* Note that this increments md->count_parts and md->leaf_pairs_packed */
    if (t->subtype == task_subtype_gpu_density) {
      runner_dopair_gpu_pack_density(r, buf, cii, cjj);
    } else if (t->subtype == task_subtype_gpu_gradient) {
      runner_dopair_gpu_pack_gradient(r, buf, cii, cjj);
    } else if (t->subtype == task_subtype_gpu_force) {
      runner_dopair_gpu_pack_force(r, buf, cii, cjj);
    }
#ifdef SWIFT_DEBUG_CHECKS
    else {
      error("Unknown task subtype %s", subtaskID_names[t->subtype]);
    }
#endif

    /* record how many leaves we've packed in total during this while loop */
    npacked++;

    /* Update the current last leaf cell pair index of this task. */
    /* md->leaf_pairs_packed was incremented in runner_dopair_gpu_pack_<*>. */
    tflplp[tind][1] = md->leaf_pairs_packed;

    /* Can we launch? */
    if (md->leaf_pairs_packed == target_n_leaves) md->launch = 1;

    /* Are we launching, or are we launching leftovers AND have packed all
     * remaining leaves? */
    if (md->launch ||
        (md->launch_leftovers && (npacked == md->task_n_leaves))) {

      if (t->subtype == task_subtype_gpu_density) {

        /* Launch the GPU offload */
        runner_dopair_gpu_launch_density(r, buf, stream, d_a, d_H);

        /* Unpack the results into CPU memory */
        runner_dopair_gpu_unpack_density(r, s, buf, npacked);

      } else if (t->subtype == task_subtype_gpu_gradient) {

        /* Launch the GPU offload */
        runner_dopair_gpu_launch_gradient(r, buf, stream, d_a, d_H);

        /* Unpack the results into CPU memory */
        runner_dopair_gpu_unpack_gradient(r, s, buf, npacked);

      } else if (t->subtype == task_subtype_gpu_force) {

        /* Launch the GPU offload */
        runner_dopair_gpu_launch_force(r, buf, stream, d_a, d_H);

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
        int n_leaves_new = md->n_leaves - md->leaf_pairs_packed;
        /* How many leaves does this task have in total? */
        int task_n_leaves = md->task_n_leaves;
        /* Store launch_leftovers in case we still need to do that after we
         * finish packing all leaf cell pairs */
        char launch_leftovers = md->launch_leftovers;

        /* Shift the leaf cells down to index 0 in their arrays. */
        /* Reminder: md->leaf_pairs_packed is the current number of leaf pairs
         * in the buffers. md->n_leaves is the total number of leaf pairs we
         * have identified for offloading, including all of this task's leaves.
         */
        for (int i = md->leaf_pairs_packed; i < md->n_leaves; i++) {
          const int shift_ind = i - md->leaf_pairs_packed;
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
        md->ci_super[0] = ci;
        md->cj_super[0] = cj;
        md->n_leaves = n_leaves_new;
        md->task_n_leaves = task_n_leaves;
        md->tasks_in_list = 1;

        /* Whatever the remaining leaf cells are, they will belong to the
         * current task. Any previous task will already have been offloaded. */
        tflplp[0][0] = 0; /* First index is now 0 */
        tflplp[0][1] = 0; /* Nothing's packed yet. */
        /* Same for first particle index of task */
        md->task_first_part[0] = 0;

      } /* Launched, but not finished packing */
    } /* if launch or launch_leftovers */
  } /* while npacked < md->task_n_leaves */

  md->launch_leftovers = 0;
  md->launch = 0;
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

  /* Collect cell interaction data recursively*/
  runner_dopair_gpu_recurse(r, s, buf, ci, cj, /*depth=*/0, /*timer=*/1);

  /* Check to see if this is the last task in the queue. If so, set
   * launch_leftovers to 1 and recursively pack and launch on GPU */
  unsigned int qid = r->qid;
  lock_lock(&s->queues[qid].lock);
  s->queues[qid].n_packs_pair_left_d--;
  if (s->queues[qid].n_packs_pair_left_d < 1) buf->md.launch_leftovers = 1;
  (void)lock_unlock(&s->queues[qid].lock);

  /* pack the data and run, if enough data has been gathered */
  runner_dopair_gpu_pack_and_launch(r, s, ci, cj, buf, t, stream, d_a, d_H);
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

  /* Collect cell interaction data recursively*/
  runner_dopair_gpu_recurse(r, s, buf, ci, cj, /*depth=*/0, /*timer=*/1);

  /* A. Nasar: Check to see if this is the last task in the queue.
   * If so, set launch_leftovers to 1 and recursively pack and launch
   * daughter tasks on GPU */
  unsigned int qid = r->qid;
  lock_lock(&s->queues[qid].lock);
  s->queues[qid].n_packs_pair_left_g--;
  if (s->queues[qid].n_packs_pair_left_g < 1) buf->md.launch_leftovers = 1;
  (void)lock_unlock(&s->queues[qid].lock);

  /* pack the data and run, if enough data has been gathered */
  runner_dopair_gpu_pack_and_launch(r, s, ci, cj, buf, t, stream, d_a, d_H);
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

  /* Collect cell interaction data recursively*/
  runner_dopair_gpu_recurse(r, s, buf, ci, cj, /*depth=*/0, /*timer=*/1);

  /* A. Nasar: Check to see if this is the last task in the queue. If so, set
   * launch_leftovers to 1 and recursively pack and launch daughter tasks on
   * GPU */

  unsigned int qid = r->qid;
  lock_lock(&s->queues[qid].lock);
  s->queues[qid].n_packs_pair_left_f--;
  if (s->queues[qid].n_packs_pair_left_f < 1) buf->md.launch_leftovers = 1;
  (void)lock_unlock(&s->queues[qid].lock);

  /* pack the data and run, if enough data has been gathered */
  runner_dopair_gpu_pack_and_launch(r, s, ci, cj, buf, t, stream, d_a, d_H);
}

#ifdef __cplusplus
}
#endif

#endif /* RUNNER_GPU_PACK_FUNCTIONS_H */
