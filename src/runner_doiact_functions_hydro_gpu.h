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
 * @brief Packing procedure for the self density tasks
 *
 * @param r The runner
 * @param s The scheduler
 * @param buf the gpu data buffers
 * @param ci the associated cell
 * @param t the associated task to pack
 * @param mode: 0 for density, 1 for gradient, 2 for force
 */
__attribute__((always_inline)) INLINE static void runner_doself_gpu_pack(
    struct cell *ci, struct task *t, struct gpu_offload_data *restrict buf,
    const enum task_subtypes task_subtype) {

  /* Grab a hold of the packing buffers */
  struct gpu_pack_metadata *md = &(buf->md);

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

#ifdef SWIFT_DEBUG_CHECKS
    const int local_pack_position = md->count_parts;
    const int count_max_parts_tmp = md->params.count_max_parts;
    if (local_pack_position + count >= count_max_parts_tmp) {
      error(
          "Exceeded count_max_parts_tmp. Make arrays bigger! "
          "count_max %d count %d",
          count_max_parts_tmp, local_pack_position + count);
    }
#endif

    /* This re-arranges the particle data from cell->hydro->parts into a
       long array of part structs */
    if (task_subtype == task_subtype_gpu_pack_d) {
      gpu_pack_self_density(ci, buf);
    } else if (task_subtype == task_subtype_gpu_pack_g) {
      gpu_pack_self_gradient(ci, buf);
    } else if (task_subtype == task_subtype_gpu_pack_f) {
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
 */
__attribute__((always_inline)) INLINE static void
runner_doself_gpu_pack_density(const struct runner *r, struct scheduler *s,
                               struct gpu_offload_data *restrict buf, struct cell *ci,
                               struct task *t) {

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
 */
__attribute__((always_inline)) INLINE static void
runner_doself_gpu_pack_gradient(const struct runner *r, struct scheduler *s,
                                struct gpu_offload_data *restrict buf, struct cell *ci,
                                struct task *t) {

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
 */
__attribute__((always_inline)) INLINE static void runner_doself_gpu_pack_force(
    const struct runner *r, struct scheduler *s, struct gpu_offload_data *restrict buf,
    struct cell *ci, struct task *t) {

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
 */
static void runner_dopair_gpu_recurse(const struct runner *r,
                                      const struct scheduler *s,
                                      struct gpu_offload_data *restrict buf,
                                      struct cell *ci, struct cell *cj,
                                      const struct task *t, const int depth,
                                      const char timer) {

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
    /* Keep track of how many leaves we've packed already before we enter the
     * recursion here */
    /* TODO: remove this? */
    /* md->n_leaves_in_list = md->n_leaves; */

    /* Reset counter before we recurse down */
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
                                  t, depth + 1, /*timer=*/0);
      }
    }
  } else if (cell_is_active_hydro(ci, e) || cell_is_active_hydro(cj, e)) {

    /* if any cell empty: skip */
    if (ci->hydro.count == 0 || cj->hydro.count == 0) return;

    /* Found leaves with work to do. Add them to list. */
    ci_leaves[md->n_leaves] = ci;
    cj_leaves[md->n_leaves] = cj;

    md->task_n_leaves++;

#ifdef SWIFT_DEBUG_CHECKS
    if (md->n_leaves >= md->ci_leaves_size) {
      error("Found more leaf cells (%d) than expected (%d), depth=%i",
          md->n_leaves, md->ci_leaves_size, depth);
    }
    if (md->n_leaves >= md->cj_leaves_size) {
      error("Found more leaf cells (%d) than expected (%d), depth=%i",
          md->n_leaves, md->cj_leaves_size, depth);
    }
#endif

  }

  if (timer) TIMER_TOC(timer_dopair_gpu_recurse);
}

/**
 * @brief Generic function to pack GPU pair tasks' leaf cells
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
  /* if (count_ci == 0 || count_cj == 0) return; */
  if (count_ci == 0 || count_cj == 0)
    error(
        "ToDo: We should be doing none of this if the cells are not interacting, "
        "but that behaviour is untested. Code below assumes both cells are active "
        "and have non-zero particle counts.");
#endif

  /* Grab handles */
  const struct engine *e = r->e;
  int4 *fparti_fpartj_lparti_lpartj = buf->fparti_fpartj_lparti_lpartj;
  struct gpu_pack_metadata *md = &buf->md;

  /* Get the index for the leaf cell pair */
  const int lid = md->leaf_pairs_packed;

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

  ////////////////////////
  // THIS IS A PROBLEM!!!
  ////////////////////////
  /* TODO ABOUZIED: Is the 'problem' comment above still accurate? */
  /* Store first and last particle indices in the buffers for ci and cj. */
  fparti_fpartj_lparti_lpartj[lid].x = md->count_parts;
  fparti_fpartj_lparti_lpartj[lid].y = md->count_parts + count_ci;
  fparti_fpartj_lparti_lpartj[lid].z = md->count_parts + count_ci;
  fparti_fpartj_lparti_lpartj[lid].w = md->count_parts + count_ci + count_cj;

  /* Pack the data into the CPU-side buffers for offloading. */
  if (task_subtype == task_subtype_gpu_pack_d) {
    gpu_pack_pair_density(buf, ci, cj, shift);
  } else if (task_subtype == task_subtype_gpu_pack_g) {
    gpu_pack_pair_gradient(buf, ci, cj, shift);
  } else if (task_subtype == task_subtype_gpu_pack_f) {
    gpu_pack_pair_force(buf, ci, cj, shift);
  }
#ifdef SWIFT_DEBUG_CHECKS
  else {
    error("Unknown task subtype %s", subtaskID_names[task_subtype]);
  }
#endif

  /* Update incremented pack length accordingly */
  md->count_parts += count_ci + count_cj;

  /* Identify first particle for each bundle of tasks */
  const int bundle_size = md->params.bundle_size_pair;
  if (lid % bundle_size == 0) {
    int bid = lid / bundle_size;
    md->bundle_first_part[bid] = fparti_fpartj_lparti_lpartj[lid].x;

    /* A. Nasar: This is possibly a problem! */
    /* TODO: Why? Is this still accurate? */
    md->bundle_first_leaf[bid] = lid;
  }

  /* Record that we have now done a pair pack leaf cell pair & increment number
   * of leaf cell pairs to offload */
  md->leaf_pairs_packed++;
};

/**
 * Wrapper to pack data for density pair tasks on the GPU.
 */
__attribute__((always_inline)) INLINE static void
runner_dopair_gpu_pack_density(const struct runner *r,
                               struct gpu_offload_data *restrict buf,
                               const struct cell *ci, const struct cell *cj,
                               const struct task *t) {

  TIMER_TIC;
  runner_dopair_gpu_pack(r, buf, ci, cj, t->subtype);
  TIMER_TOC(timer_dopair_gpu_pack_d);
}

/**
 * Wrapper to pack data for density pair tasks on the GPU.
 */
__attribute__((always_inline)) INLINE static void
runner_dopair_gpu_pack_gradient(const struct runner *r,
                                struct gpu_offload_data *restrict buf,
                                const struct cell *ci, const struct cell *cj,
                                const struct task *t) {

  TIMER_TIC;
  runner_dopair_gpu_pack(r, buf, ci, cj, t->subtype);
  TIMER_TOC(timer_dopair_gpu_pack_g);
}

/**
 * Wrapper to pack data for density pair tasks on the GPU.
 */
__attribute__((always_inline)) INLINE static void runner_dopair_gpu_pack_force(
    const struct runner *r, struct gpu_offload_data *restrict buf,
    const struct cell *ci, const struct cell *cj, const struct task *t) {

  TIMER_TIC;
  runner_dopair_gpu_pack(r, buf, ci, cj, t->subtype);
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

  /* How many tasks have we packed? */
  const int tasks_packed = md->self_tasks_packed;

  /* How many tasks should be in a bundle? */
  const int bundle_size = md->params.bundle_size;

  if (md->launch_leftovers) {
    /* Special case for incomplete bundles (when not having enough leftover
     * tasks to fill a bundle) */

    if (tasks_packed == 0)
      error("zero tasks packed but somehow got into GPU loop");

    /* Compute how many bundles we actually have, rounding up */
    n_bundles = (tasks_packed + bundle_size - 1) / bundle_size;

    /* Get the index of the first particle of the last (incomplete) bundle */
    md->bundle_first_part[n_bundles] = buf->self_task_first_last_part[tasks_packed - 1].x;
  }

  /* Store how many bundles we'll need to unpack */
  md->n_bundles_unpack = n_bundles;

  /* Identify the last particle for each bundle of tasks */
  for (int bid = 0; bid < n_bundles - 1; bid++) {
    md->bundle_last_part[bid] = md->bundle_first_part[bid + 1];
  }

  /* special treatment for the last bundle */
  if (n_bundles > 1)
    md->bundle_last_part[n_bundles - 1] = md->count_parts;
  else
    md->bundle_last_part[0] = md->count_parts;

  /* Launch the copies for each bundle and run the GPU kernel */
  /* We don't go into this loop if tasks_left_self == 1 as
   n_bundles will be zero DUHDUHDUHDUHHHHHH!!!!!*/
  for (int bid = 0; bid < n_bundles; bid++) {

    int max_parts = 0;
    for (int tid = bid * bundle_size; tid < (bid + 1) * bundle_size; tid++) {
      if (tid < tasks_packed) {
        /* Get an estimate for the max number of parts per cell in the bundle.
         * Used for determining the number of GPU CUDA blocks*/
        int count = buf->self_task_first_last_part[tid].y - buf->self_task_first_last_part[tid].x;
        max_parts = max(max_parts, count);
      }
    }

    const int first_part = md->bundle_first_part[bid];
    const int bundle_n_parts = md->bundle_last_part[bid] - first_part;
    const int tasks_left = (bid == n_bundles - 1) ? tasks_packed - (n_bundles - 1) * bundle_size : bundle_size;

    /* Will launch a 2d grid of GPU thread blocks (number of tasks is
       the y dimension and max_parts is the x dimension */
    const int numBlocks_y = tasks_left;
    const int numBlocks_x = (max_parts + GPU_THREAD_BLOCK_SIZE - 1) / GPU_THREAD_BLOCK_SIZE;

    const int first_task = md->bundle_first_leaf[bid];
    const int last_task = first_task + tasks_left;
#ifdef SWIFT_DEBUG_CHECKS
    if (last_task > buf->d_self_task_first_last_part_size)
      error("Trying to access out-of-boundary in d_task_first_part");
    if (last_task > buf->self_task_first_last_part_size)
      error("Trying to access out-of-boundary in d_task_first_part");
#endif

    /* Copy data over to GPU */
    /* First the meta-data */
    cudaError_t cu_error = cudaMemcpyAsync(&buf->d_self_task_first_last_part[first_task],
                    &buf->self_task_first_last_part[first_task],
                    (last_task + 1 - first_task) * sizeof(int2),
                    cudaMemcpyHostToDevice, stream[bid]);

    if (cu_error != cudaSuccess) {
      error("H2D memcpy metadata self: CUDA error '%s' for task_sybtype %s: "
          "cpuid=%i, first_task=%d, size=%lu",
            cudaGetErrorString(cu_error), subtaskID_names[task_subtype],
            r->cpuid, first_task, (last_task + 1 - first_task) * sizeof(int2)
            );
    }

    /* Now the actual particle data */
    if (task_subtype == task_subtype_gpu_pack_d) {

      cu_error = cudaMemcpyAsync(&buf->d_parts_send_d[first_part],
                      &buf->parts_send_d[first_part],
                      bundle_n_parts * sizeof(struct gpu_part_send_d),
                      cudaMemcpyHostToDevice, stream[bid]);

    } else if (task_subtype == task_subtype_gpu_pack_g) {

      cu_error = cudaMemcpyAsync(&buf->d_parts_send_g[first_part],
                      &buf->parts_send_g[first_part],
                      bundle_n_parts * sizeof(struct gpu_part_send_g),
                      cudaMemcpyHostToDevice, stream[bid]);

    } else if (task_subtype == task_subtype_gpu_pack_f) {

      cu_error = cudaMemcpyAsync(&buf->d_parts_send_f[first_part],
                      &buf->parts_send_f[first_part],
                      bundle_n_parts * sizeof(struct gpu_part_send_f),
                      cudaMemcpyHostToDevice, stream[bid]);

    } else {
      error("Unknown task subtype %s", subtaskID_names[task_subtype]);
    }

    if (cu_error != cudaSuccess) {
      error("H2D memcpy self: CUDA error '%s' for task_subtype %s: cpuid=%i",
            cudaGetErrorString(cu_error), subtaskID_names[task_subtype], r->cpuid);
    }

    /* Launch the kernel */
    if (task_subtype == task_subtype_gpu_pack_d) {

      gpu_launch_self_density(buf->d_parts_send_d, buf->d_parts_recv_d, d_a,
                              d_H, stream[bid], numBlocks_x, numBlocks_y,
                              first_task, buf->d_self_task_first_last_part);

    } else if (task_subtype == task_subtype_gpu_pack_g) {

      gpu_launch_self_gradient(buf->d_parts_send_g, buf->d_parts_recv_g, d_a,
                               d_H, stream[bid], numBlocks_x, numBlocks_y,
                               first_task, buf->d_self_task_first_last_part);

    } else if (task_subtype == task_subtype_gpu_pack_f) {

      gpu_launch_self_force(buf->d_parts_send_f, buf->d_parts_recv_f, d_a, d_H,
                            stream[bid], numBlocks_x, numBlocks_y,
                            first_task, buf->d_self_task_first_last_part);
    }
#ifdef SWIFT_DEBUG_CHECKS
    else {
      error("Unknown task subtype %s", subtaskID_names[task_subtype]);
    }
#endif

    cu_error = cudaGetLastError();
    if (cu_error != cudaSuccess) {
      error( "kernel launch self: CUDA error '%s' for task_subtype %s: cpuid=%i",
            cudaGetErrorString(cu_error), subtaskID_names[task_subtype], r->cpuid);
    }

    /* Copy back */
    if (task_subtype == task_subtype_gpu_pack_d) {

      cu_error = cudaMemcpyAsync(&buf->parts_recv_d[first_part],
                      &buf->d_parts_recv_d[first_part],
                      bundle_n_parts * sizeof(struct gpu_part_recv_d),
                      cudaMemcpyDeviceToHost, stream[bid]);

    } else if (task_subtype == task_subtype_gpu_pack_g) {

      cu_error = cudaMemcpyAsync(&buf->parts_recv_g[first_part],
                      &buf->d_parts_recv_g[first_part],
                      bundle_n_parts * sizeof(struct gpu_part_recv_g),
                      cudaMemcpyDeviceToHost, stream[bid]);

    } else if (task_subtype == task_subtype_gpu_pack_f) {

      cu_error = cudaMemcpyAsync(&buf->parts_recv_f[first_part],
                      &buf->d_parts_recv_f[first_part],
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
            cudaGetErrorString(cu_error), subtaskID_names[task_subtype], r->cpuid);
    }

    /* Record this event */
    cu_error = cudaEventRecord(buf->event_end[bid], stream[bid]);
    swift_assert(cu_error == cudaSuccess);

  } /*End of looping over bundles to launch in streams*/

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

  /* Copy the data back from the CPU thread-local buffers to the cells */
  /* Pack length counter for use in unpacking */
  int pack_length = 0;
  cudaError_t cu_error;
  for (int bid = 0; bid < n_bundles; bid++) {

    cu_error = cudaEventSynchronize(buf->event_end[bid]);
    swift_assert(cu_error == cudaSuccess);

    for (int tid = bid * bundle_size; tid < (bid + 1) * bundle_size && tid < tasks_packed; tid++) {

      struct cell *c = md->ci_list[tid];
      struct task *t = md->task_list[tid];
      const int count = c->hydro.count;

      /* Anything to do here? */
      if (count == 0) continue;
#ifdef SWIFT_DEBUG_CHECKS
      if (!cell_is_active_hydro(c, e))
        error("We packed an inactive cell for self tasks???");
#endif

      while (cell_locktree(c)) {
        ; /* spin until we acquire the lock */
      }

#ifdef SWIFT_DEBUG_CHECKS
      if (pack_length + count >= md->params.count_max_parts) {
        error(
            "Exceeded count_max_parts. Make arrays bigger! pack_length is "
            "%d, count is %d, max_parts is %d",
            pack_length, count, md->params.count_max_parts);
      }
#endif

      /* Do the copy */
      if (task_subtype == task_subtype_gpu_pack_d) {
        gpu_unpack_self_density(c, buf->parts_recv_d, pack_length, count, e);
      } else if (task_subtype == task_subtype_gpu_pack_g) {
        gpu_unpack_self_gradient(c, buf->parts_recv_g, pack_length, count, e);
      } else if (task_subtype == task_subtype_gpu_pack_f) {
        gpu_unpack_self_force(c, buf->parts_recv_f, pack_length, count, e);
      }
#ifdef SWIFT_DEBUG_CHECKS
      else {
        error("Unknown task subtype %s", subtaskID_names[task_subtype]);
      }
#endif

      /* Increase our index in the buffer with the newly unpacked size. */
      pack_length += count;

      /* TODO: What exactly is happening here? Please document. */
      pthread_mutex_lock(&s->sleep_mutex);
      atomic_dec(&s->waiting);
      pthread_cond_broadcast(&s->sleep_cond);
      pthread_mutex_unlock(&s->sleep_mutex);

      /* Release the lock */
      cell_unlocktree(c);

      /* schedule my dependencies (Only unpacks really) */
      enqueue_dependencies(s, t);

    } /* Loop over tasks in bundle */
  } /* Loop over bundles */

  /* Zero counters for the next pack operations */
  gpu_pack_metadata_reset(md);
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
  int4 *fi_fj_li_lj = buf->fparti_fpartj_lparti_lpartj;
  cudaEvent_t *pair_end = buf->event_end;

  /* How many tasks have we packed? */
  const int leaves_packed = md->leaf_pairs_packed;

  /* How many tasks should be in a bundle? */
  const int bundle_size = md->params.bundle_size_pair;

  /* Identify the number of GPU bundles to run in ideal case */
  int n_bundles = md->params.n_bundles_pair;

  /* Special case for incomplete bundles (when having leftover tasks not enough
   * to fill a bundle) */
  if (md->launch_leftovers) {
    n_bundles = (leaves_packed + bundle_size - 1) / bundle_size;
    /* TODO: We shouldn't have to do this, this should already have happened.
     * Test removing this later. */
    md->bundle_first_part[n_bundles] = fi_fj_li_lj[leaves_packed - 1].x;

#ifdef SWIFT_DEBUG_CHECKS
    if (n_bundles > md->params.bundle_size_pair)
      error("Launching leftovers with too many bundles?");
#endif
  }

  /* Identify the last particle for each bundle of tasks */
  for (int bid = 0; bid < n_bundles - 1; bid++) {
    md->bundle_last_part[bid] = md->bundle_first_part[bid + 1];
  }

  /* Special treatment for the case of 1 bundle */
  if (n_bundles > 1){
    /* Note: This also correctly handles the case when launching leftovers. */
    md->bundle_last_part[n_bundles - 1] = md->count_parts;
  } else {
    md->bundle_last_part[0] = md->count_parts;
  }

  /* Launch the copies for each bundle and run the GPU kernel. Each bundle gets
   * its own stream. */
  for (int bid = 0; bid < n_bundles; bid++) {

    const int first_part_i = md->bundle_first_part[bid];
    const int bundle_n_parts = md->bundle_last_part[bid] - first_part_i;
    cudaError_t cu_error;

    /* Transfer memory to device */
    if (task_subtype == task_subtype_gpu_launch_d) {

      cu_error = cudaMemcpyAsync(&buf->d_parts_send_d[first_part_i],
                      &buf->parts_send_d[first_part_i],
                      bundle_n_parts * sizeof(struct gpu_part_send_d),
                      cudaMemcpyHostToDevice, stream[bid]);

    } else if (task_subtype == task_subtype_gpu_launch_g) {

      cu_error = cudaMemcpyAsync(&buf->d_parts_send_g[first_part_i],
                      &buf->parts_send_g[first_part_i],
                      bundle_n_parts * sizeof(struct gpu_part_send_g),
                      cudaMemcpyHostToDevice, stream[bid]);

    } else if (task_subtype == task_subtype_gpu_launch_f) {

      cu_error = cudaMemcpyAsync(&buf->d_parts_send_f[first_part_i],
                      &buf->parts_send_f[first_part_i],
                      bundle_n_parts * sizeof(struct gpu_part_send_f),
                      cudaMemcpyHostToDevice, stream[bid]);
    }
#ifdef SWIFT_DEBUG_CHECKS
    else {
      error("Unknown task subtype %s", subtaskID_names[task_subtype]);
    }
#endif

    if (cu_error != cudaSuccess) {
      /* If we're here, assume something's messed up with our code, not with CUDA. */
      error(
          "H2D memcpy pair: CUDA error '%s' for task_subtype %s: cpuid=%i "
          "first_part=%d bundle_n_parts=%d",
          cudaGetErrorString(cu_error),
          subtaskID_names[task_subtype], r->cpuid,
          first_part_i, bundle_n_parts);
    }

    /* Launch the GPU kernels for ci & cj as a 1D grid */
    /* TODO: num_blocks_y is not used anymore. Purge it. */
    const int num_blocks_x = (bundle_n_parts + GPU_THREAD_BLOCK_SIZE - 1) / GPU_THREAD_BLOCK_SIZE;
    const int num_blocks_y = 0;
    const int bundle_part_0 = md->bundle_first_part[bid];

    /* Launch the kernel for ci using data for ci and cj */
    if (task_subtype == task_subtype_gpu_launch_d) {

      gpu_launch_pair_density(buf->d_parts_send_d, buf->d_parts_recv_d, d_a,
                              d_H, stream[bid], num_blocks_x, num_blocks_y,
                              bundle_part_0, bundle_n_parts);

    } else if (task_subtype == task_subtype_gpu_launch_g) {

      gpu_launch_pair_gradient(buf->d_parts_send_g, buf->d_parts_recv_g, d_a,
                               d_H, stream[bid], num_blocks_x, num_blocks_y,
                               bundle_part_0, bundle_n_parts);

    } else if (task_subtype == task_subtype_gpu_launch_f) {

      gpu_launch_pair_force(buf->d_parts_send_f, buf->d_parts_recv_f, d_a, d_H,
                            stream[bid], num_blocks_x, num_blocks_y,
                            bundle_part_0, bundle_n_parts);
    }
#ifdef SWIFT_DEBUG_CHECKS
    else {
      error("Unknown task subtype %s", subtaskID_names[task_subtype]);
    }
#endif

    cu_error = cudaGetLastError();
    if (cu_error != cudaSuccess) {
      /* If we're here, assume something's messed up with our code, not with CUDA. */
      error(
          "kernel launch pair: CUDA error '%s' for task_subtype %s: cpuid=%i "
          "nbx=%i nby=%i",
          cudaGetErrorString(cu_error), subtaskID_names[task_subtype], r->cpuid,
          num_blocks_x, num_blocks_y);
    }

    /* Copy results back to CPU BUFFERS */
    if (task_subtype == task_subtype_gpu_launch_d) {

      cu_error = cudaMemcpyAsync(&buf->parts_recv_d[first_part_i],
                      &buf->d_parts_recv_d[first_part_i],
                      bundle_n_parts * sizeof(struct gpu_part_recv_d),
                      cudaMemcpyDeviceToHost, stream[bid]);

    } else if (task_subtype == task_subtype_gpu_launch_g) {

      cu_error = cudaMemcpyAsync(&buf->parts_recv_g[first_part_i],
                      &buf->d_parts_recv_g[first_part_i],
                      bundle_n_parts * sizeof(struct gpu_part_recv_g),
                      cudaMemcpyDeviceToHost, stream[bid]);

    } else if (task_subtype == task_subtype_gpu_launch_f) {

      cu_error = cudaMemcpyAsync(&buf->parts_recv_f[first_part_i],
                      &buf->d_parts_recv_f[first_part_i],
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
      error(
          "D2H async memcpy: CUDA error '%s' for task_subtype %s: cpuid=%i",
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
    if (cu_error != cudaSuccess){
      error(
          "cudaEventSynchronize failed: '%s' for task subtype %s,"
          " cpuid=%d, bundle=%d",
          cudaGetErrorString(cu_error), subtaskID_names[task_subtype],
          r->cpuid, bid); }
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

  runner_dopair_gpu_launch(r, buf, stream, d_a, d_H, task_subtype_gpu_launch_d);

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

  runner_dopair_gpu_launch(r, buf, stream, d_a, d_H, task_subtype_gpu_launch_g);

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

  runner_dopair_gpu_launch(r, buf, stream, d_a, d_H, task_subtype_gpu_launch_f);

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
#pragma GCC push_options
#pragma GCC optimize ("O0")
__attribute__((always_inline)) INLINE static void runner_dopair_gpu_unpack(
    const struct runner *r, struct scheduler *s,
    struct gpu_offload_data *restrict buf, const int npacked,
    const enum task_subtypes task_subtype) {

  /* Grab handles */
  struct gpu_pack_metadata *md = &buf->md;
  int n_leaves_packed = md->leaf_pairs_packed;

  struct cell **ci_leaves = md->ci_leaves;
  struct cell **cj_leaves = md->cj_leaves;
  struct cell **ci_super = md->ci_super;
  struct cell **cj_super = md->cj_super;
  int **tfllp = md->task_first_last_leaf_pair;

  int unpack_index = 0;

  /* Loop over tasks */
  for (int tid = 0; tid < md->tasks_in_list; tid++) {

    /* TODO: if this is a bottleneck, what we could do is not spin until we get
     * the lock, but continue with the main for loop and return back to the
     * unfinished ones later. Basically wrap the for loop into a while loop,
     * while keeping track which indices we've finished already. */
    while (cell_locktree(ci_super[tid])) {
      ; /* spin until we acquire the lock */
    }
    while (cell_locktree(cj_super[tid])) {
      ; /* spin until we acquire the lock */
    }

    /* Loop through leaf cell pairs by index */
    for (int lid = tfllp[tid][0]; lid < tfllp[tid][1]; lid++) {

      /*Get pointers to the leaf cells*/
      struct cell *cii_l = ci_leaves[lid];
      struct cell *cjj_l = cj_leaves[lid];

      /* Not a typo: task subtype is task_subtype_pack_*. The unpacking gets
       * called at the end of packing, running, and possibly launching. */
      /* Note that these calls increment pack_length_unpack. */
      if (task_subtype == task_subtype_gpu_pack_d) {
        /* TODO: WHY IS THERE A FACTOR OF 2 HERE? LEAVE A COMMENT. */
        gpu_unpack_pair_density(r, cii_l, cjj_l, buf->parts_recv_d,
                                &unpack_index,
                                2 * md->params.count_max_parts);
      } else if (task_subtype == task_subtype_gpu_pack_g) {
        gpu_unpack_pair_gradient(r, cii_l, cjj_l, buf->parts_recv_g,
                                 &unpack_index,
                                 2 * md->params.count_max_parts);
      } else if (task_subtype == task_subtype_gpu_pack_f) {
        gpu_unpack_pair_force(r, cii_l, cjj_l, buf->parts_recv_f,
                              &unpack_index,
                              2 * md->params.count_max_parts);
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

    /* For some reason the code fails if we get a leaf pair task
     * this if->continue statement stops the code from trying to unlock same
     * cells twice*/
    /* TODO (Mladen): I don't understand, but this seems like a genuine problem
     * we need to get to the bottom of. */
    if (tid == md->tasks_in_list - 1 && npacked != n_leaves_packed) {
#ifdef SWIFT_DEBUG_CHECKS
      warning("Encountered edge case we need to look into.");
#endif
      continue;
    }

    /* TODO: DOCUMENT WHAT'S HAPPENING HERE AND WHY */
    enqueue_dependencies(s, md->task_list[tid]);
    pthread_mutex_lock(&s->sleep_mutex);
    atomic_dec(&s->waiting);
    pthread_cond_broadcast(&s->sleep_cond);
    pthread_mutex_unlock(&s->sleep_mutex);
  }
}
#pragma GCC pop_options

/**
 * Wrapper for the density unpacking of pair tasks:
 * Provide the correct subtype to use and time the runtime separately
 * TODO: Documentation: WHAT IS NPACKED??
 */
__attribute__((always_inline)) INLINE static void
runner_dopair_gpu_unpack_density(const struct runner *r, struct scheduler *s,
                                 struct gpu_offload_data *restrict buf,
                                 const int npacked) {

  TIMER_TIC;

  runner_dopair_gpu_unpack(r, s, buf, npacked, task_subtype_gpu_pack_d);

  TIMER_TOC(timer_dopair_gpu_unpack_d);
}

/**
 * Wrapper for the gradient unpacking of pair tasks:
 * Provide the correct subtype to use and time the runtime separately
 * TODO: Documentation: WHAT IS NPACKED??
 */
__attribute__((always_inline)) INLINE static void
runner_dopair_gpu_unpack_gradient(const struct runner *r, struct scheduler *s,
                                  struct gpu_offload_data *restrict buf,
                                  const int npacked) {

  TIMER_TIC;

  runner_dopair_gpu_unpack(r, s, buf, npacked, task_subtype_gpu_pack_g);

  TIMER_TOC(timer_dopair_gpu_unpack_g);
}

/**
 * Wrapper for the force unpacking of pair tasks:
 * Provide the correct subtype to use and time the runtime separately
 * TODO: Documentation: WHAT IS NPACKED??
 */
__attribute__((always_inline)) INLINE static void
runner_dopair_gpu_unpack_force(const struct runner *r, struct scheduler *s,
                               struct gpu_offload_data *restrict buf,
                               const int npacked) {

  TIMER_TIC;

  runner_dopair_gpu_unpack(r, s, buf, npacked, task_subtype_gpu_pack_f);

  TIMER_TOC(timer_dopair_gpu_unpack_f);
}

/**
 * Generic function to pack pair tasks and launch them
 * on the device depending on the task subtype
 */
__attribute__((always_inline)) INLINE static void
runner_dopair_gpu_pack_and_launch(const struct runner *r, struct scheduler *s,
                                  struct cell *ci, struct cell *cj,
                                  struct gpu_offload_data *restrict buf,
                                  struct task *t, cudaStream_t *stream,
                                  const float d_a, const float d_H) {

  /* Grab handles */
  struct gpu_pack_metadata *md = &buf->md;
  int **tfllp = md->task_first_last_leaf_pair;
  /* Nr of super-level tasks we've accounted for in the meda-data arrays. */
  const int tind = md->tasks_in_list;

#ifdef SWIFT_DEBUG_CHECKS
  if (tind >= md->task_list_size)
    error("Writing out of top_task_packed array bounds: %d/%d",
        tind, md->task_list_size);
  if (tind >= md->ci_super_size)
    error("Writing out of ci_top array bounds: %d/%d",
        tind, md->ci_super_size);
  if (tind >= md->cj_super_size)
    error("Writing out of cj_top array bounds: %d/%d",
        tind, md->cj_super_size);
#endif

  /* Keep track of first and last index of leaf cell pairs in lists per
   * super-level pair task in case we are packing more than one super-level
   * task into this buffer */
  tfllp[tind][0] = md->n_leaves;
  /* TODO: Add last in list here too? */
  /* tfllp[tind][1] = md->n_leaves + md->task_n_leaves; */

  /* TODO: Do we need this? We already keep track of the task, and the
   * task has access to t->ci, t->cj */
  /* Possibly necessary due to swap of cell pointers in
   * space_getsid_and_swap_cells */
  /* Same for super-level cells */
  md->ci_super[tind] = ci;
  md->cj_super[tind] = cj;

  /* Get pointer to task. Needed to enqueue dependencies after we're done. */
  md->task_list[tind] = t;

  /* Increment how many tasks we've accounted for */
  md->tasks_in_list++;

  /* Update the total number of leaf pair interactions we found through the
   * recursion. */
  md->n_leaves += md->task_n_leaves;

  /* Index of the last leaf cell pair we need to pack */
  int last_leaf_ind = md->n_leaves;

  /* How many leaf cell interactions do we want to offload at once? */
  const int target_n_leaves = md->params.pack_size_pair;

  /* Counter for how many leaf cell pairs of this task we've packed */
  int npacked = 0;

  /* Loop through the leaf cell interactions we found */
  int copy_index = md->leaf_pairs_packed;

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

    /* Have we launched the computation on the GPU in this iteration? */
    /* int launched = 0; */

#ifdef SWIFT_DEBUG_CHECKS
    if (copy_index >= md->ci_leaves_size)
      error("Writing out of ci_d array bounds: %d/%d",
          copy_index, md->ci_leaves_size);
    if (copy_index >= md->cj_leaves_size)
      error("Writing out of cj_d array bounds: %d/%d",
          copy_index, md->cj_leaves_size);
#endif

    /* Grab handles. */
    struct cell *cii = md->ci_leaves[copy_index];
    struct cell *cjj = md->cj_leaves[copy_index];

#ifdef SWIFT_DEBUG_CHECKS
    if (cii->hydro.count == 0)
      error("Found cell cii with particle count=0 during packing. "
          "It should have been excluded during the recursion.");
    if (cjj->hydro.count == 0)
      error("Found cell cjj with particle count=0 during packing. "
          "It should have been excluded during the recursion.");
#endif

    /* Pack the particle data */
    if (t->subtype == task_subtype_gpu_pack_d) {
      runner_dopair_gpu_pack_density(r, buf, cii, cjj, t);
    } else if (t->subtype == task_subtype_gpu_pack_g) {
      runner_dopair_gpu_pack_gradient(r, buf, cii, cjj, t);
    } else if (t->subtype == task_subtype_gpu_pack_f) {
      runner_dopair_gpu_pack_force(r, buf, cii, cjj, t);
    }
#ifdef SWIFT_DEBUG_CHECKS
    else {
      error("Unknown task subtype %s", subtaskID_names[t->subtype]);
    }
#endif

    /* record number of leaf cell pairs we've copied since last launch */
    copy_index++;
    /* record how many leaves we've packed in total during this while loop */
    npacked++;

    /* Record the !current! last leaf cell pair of this task */
    tfllp[tind][1] = tfllp[tind][0] + copy_index;

    /* Can we launch? */
    if (md->leaf_pairs_packed == target_n_leaves) md->launch = 1;

    /* Are we launching, or are we launching leftovers AND have packed all
     * remaining leaves? */
    if (md->launch || (md->launch_leftovers && npacked == md->task_n_leaves)) {

      /* launched = 1; */

      if (t->subtype == task_subtype_gpu_pack_d) {

        /* Launch the GPU offload */
        runner_dopair_gpu_launch_density(r, buf, stream, d_a, d_H);

        /* Unpack the results into CPU memory */
        runner_dopair_gpu_unpack_density(r, s, buf, npacked);

      } else if (t->subtype == task_subtype_gpu_pack_g) {

        /* Launch the GPU offload */
        runner_dopair_gpu_launch_gradient(r, buf, stream, d_a, d_H);

        /* Unpack the results into CPU memory */
        runner_dopair_gpu_unpack_gradient(r, s, buf, npacked);

      } else if (t->subtype == task_subtype_gpu_pack_f) {

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

      if (npacked == md->n_leaves) {
        /* We have launched, finished all leaf cell pairs, and now we're done. */
        gpu_pack_metadata_reset(md);
      }
      else {
        /* Launched, but have not packed all leaf cell pairs. Re-set counters
         * and set this task to be the first in the list so that we can
         * continue packing correctly. */
        md->task_list[0] = t;
        /* Move all tasks forward in list so that the first next task will be
         * packed to index 0. Move remaining cell indices so that their indexing
         * starts from zero and ends in n_daughters_left */
        for (int i = md->leaf_pairs_packed; i < last_leaf_ind; i++) {
          int shuffle = i - md->leaf_pairs_packed;
          md->ci_leaves[shuffle] = md->ci_leaves[i];
          md->cj_leaves[shuffle] = md->cj_leaves[i];
#ifdef SWIFT_DEBUG_CHECKS
          md->ci_leaves[i] = NULL;
          md->cj_leaves[i] = NULL;
#endif
        }

        last_leaf_ind -= md->leaf_pairs_packed;
        copy_index = 0;

        tfllp[0][0] = 0;
        tfllp[0][1] = last_leaf_ind;

        md->cj_super[0] = cj;
        md->ci_super[0] = ci;

        md->tasks_in_list = 1;
        md->leaf_pairs_packed = 0;
        md->count_parts = 0;
        md->launch = 0;
        md->launch_leftovers = 0;

#ifdef SWIFT_DEBUG_CHECKS
        for (int i = 1; i < md->ci_super_size; i++)
          md->ci_super[i] = NULL;
        for (int i = 1; i < md->cj_super_size; i++)
          md->cj_super[i] = NULL;
        for (int i = 1; i < md->task_first_last_leaf_pair_size; i++){
          tfllp[i][0] = -234;
          tfllp[i][1] = -234;
        }
#endif
      }

    } /* <- if launch or launch_leftovers */
  } /* while npacked < md->task_n_leaves */

  /* Launch-leftovers counter re-set to zero and cells unlocked */
  /* TODO: No cell unlocking happening here. Is comment still accurate? */
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
  runner_dopair_gpu_recurse(r, s, buf, ci, cj, t, /*depth=*/0, /*timer=*/1);

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
 * Top level runner function to solve hydro gradient pair tasks on GPU.
 */
static void runner_dopair_gpu_gradient(const struct runner *r,
                                       struct scheduler *s, struct cell *ci,
                                       struct cell *cj,
                                       struct gpu_offload_data *restrict buf,
                                       struct task *t, cudaStream_t *stream,
                                       const float d_a, const float d_H) {

  /* Collect cell interaction data recursively*/
  runner_dopair_gpu_recurse(r, s, buf, ci, cj, t, /*depth=*/0, /*timer=*/1);

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
 * Top level runner function to solve hydro force pair tasks on GPU.
 */
static void runner_dopair_gpu_force(const struct runner *r, struct scheduler *s,
                                    struct cell *ci, struct cell *cj,
                                    struct gpu_offload_data *restrict buf,
                                    struct task *t, cudaStream_t *stream,
                                    const float d_a, const float d_H) {

  /* Collect cell interaction data recursively*/
  runner_dopair_gpu_recurse(r, s, buf, ci, cj, t, /*depth=*/0, /*timer=*/1);

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
