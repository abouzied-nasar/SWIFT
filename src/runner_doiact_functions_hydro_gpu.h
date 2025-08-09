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
    struct cell *ci, struct task *t, struct gpu_offload_data *buf,
    const enum task_subtypes task_subtype) {

  /* Grab a hold of the packing buffers */
  struct gpu_pack_vars *pv = &(buf->pv);

  /* Place pointers to the task and cells packed in an array for use later
   * when unpacking after the GPU offload */
  size_t tasks_packed = pv->tasks_packed;
  pv->task_list[tasks_packed] = t;
  pv->ci_list[tasks_packed] = ci;

  /* Identify row in particle arrays where this task starts*/
  buf->task_first_part[tasks_packed].x = pv->count_parts;

  /* Anything to do here? */
  int count = ci->hydro.count;
  if (count > 0) {

#ifdef SWIFT_DEBUG_CHECKS
    const size_t local_pack_position = buf->pv.count_parts;
    const size_t count_max_parts_tmp = buf->pv.count_max_parts;
    if (local_pack_position + count >= count_max_parts_tmp) {
      error(
          "Exceeded count_max_parts_tmp. Make arrays bigger! "
          "count_max %lu count %lu",
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
    } else {
      error("Unknown task subtype %s", subtaskID_names[task_subtype]);
    }

    /* Increment pack length accordingly */
    buf->pv.count_parts += count;
  }

  /* Identify the row in the array where this task ends (row id of its
     last particle)*/
  buf->task_first_part[tasks_packed].y = buf->pv.count_parts;

  /* Identify first particle for each bundle of tasks */
  const size_t bundle_size = buf->pv.bundle_size;
  if (tasks_packed % bundle_size == 0) {
    size_t bid = tasks_packed / bundle_size;
    buf->pv.bundle_first_part[bid] = buf->task_first_part[tasks_packed].x;
    buf->pv.bundle_first_task_list[bid] = tasks_packed;
  }

  /* Tell the cell it has been packed */
#ifdef CUDA_DEBUG
  ci->pack_done++;
#endif

  /* Record that we have now done a packing (self) */
  t->done = 1;
  buf->pv.tasks_packed++;
  buf->pv.launch = 0;
  buf->pv.launch_leftovers = 0;

  /* Have we packed enough tasks to offload to GPU? */
  if (buf->pv.tasks_packed == buf->pv.target_n_tasks) {
    buf->pv.launch = 1;
  }

  /* Release the cell. */
  /* TODO: WHERE IS THIS TREE LOCKED?????????? */
  cell_unlocktree(ci);
}

/**
 * @brief packs the data required for the self density tasks.
 */
__attribute__((always_inline)) INLINE static void
runner_doself_gpu_pack_density(const struct runner *r, struct scheduler *s,
                               struct gpu_offload_data *buf, struct cell *ci,
                               struct task *t) {

  TIMER_TIC;

  runner_doself_gpu_pack(ci, t, buf, t->subtype);

  /* Get a lock to the queue so we can safely decrement counter and check for
   * launch leftover condition*/
  int qid = r->qid;
  lock_lock(&s->queues[qid].lock);
  s->queues[qid].n_packs_self_left_d--;
  if (s->queues[qid].n_packs_self_left_d < 1) {
    buf->pv.launch_leftovers = 1;
  }
  (void)lock_unlock(&s->queues[qid].lock);

  TIMER_TOC(timer_doself_gpu_pack_d);
}

/**
 * @brief packs the data required for the self gradient tasks.
 */
__attribute__((always_inline)) INLINE static void
runner_doself_gpu_pack_gradient(const struct runner *r, struct scheduler *s,
                                struct gpu_offload_data *buf, struct cell *ci,
                                struct task *t) {

  TIMER_TIC;

  runner_doself_gpu_pack(ci, t, buf, t->subtype);

  /* Get a lock to the queue so we can safely decrement counter and check for
   * launch leftover condition*/
  int qid = r->qid;
  lock_lock(&s->queues[qid].lock);
  s->queues[qid].n_packs_self_left_g--;
  if (s->queues[qid].n_packs_self_left_g < 1) {
    buf->pv.launch_leftovers = 1;
  }
  (void)lock_unlock(&s->queues[qid].lock);

  TIMER_TOC(timer_doself_gpu_pack_g);
}

/**
 * @brief packs the data required for the self force tasks.
 */
__attribute__((always_inline)) INLINE static void runner_doself_gpu_pack_force(
    const struct runner *r, struct scheduler *s, struct gpu_offload_data *buf,
    struct cell *ci, struct task *t) {

  TIMER_TIC;

  runner_doself_gpu_pack(ci, t, buf, t->subtype);

  /* Get a lock to the queue so we can safely decrement counter and check for
   * launch leftover condition*/
  int qid = r->qid;
  lock_lock(&s->queues[qid].lock);
  s->queues[qid].n_packs_self_left_f--;
  if (s->queues[qid].n_packs_self_left_f < 1) {
    buf->pv.launch_leftovers = 1;
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

  /* Should we even bother? A. Nasar: For GPU code we need to be clever about
   * this */
  const struct engine *e = r->e;
  if (!cell_is_active_hydro(ci, e) && !cell_is_active_hydro(cj, e)) return;
  if (ci->hydro.count == 0 || cj->hydro.count == 0) return;

  /* Grab some handles. */
  /* packing data and metadata */
  struct gpu_pack_vars *pack_vars = &buf->pv;

  /* Arrays for daughter cells */
  struct cell **ci_d = buf->ci_d;
  struct cell **cj_d = buf->cj_d;

  if (depth == 0) {
    /* while at the top level, reset counters. */
    pack_vars->n_daughters_packed_index = pack_vars->n_daughters_total;
    pack_vars->n_leaves_found = 0;
  }

  const size_t n_daughters = pack_vars->n_daughters_total;

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

    /*Add leaf cells to list for each top_level task*/
    int leaves_found = pack_vars->n_leaves_found;
    ci_d[n_daughters + leaves_found] = ci;
    cj_d[n_daughters + leaves_found] = cj;

    pack_vars->n_leaves_found++;
    if (pack_vars->n_leaves_found >= pack_vars->n_expected_pair_tasks)
      error("Created %i more than expected leaf cells. depth %i",
            pack_vars->n_leaves_found, depth);
  }

  if (timer) TIMER_TOC(timer_dopair_gpu_recurse);
}

/**
 * Generic function to pack GPU pair tasks
 */
__attribute__((always_inline)) INLINE static void runner_dopair_gpu_pack(
    const struct runner *r, struct gpu_offload_data *restrict buf,
    const struct cell *ci, const struct cell *cj,
    const enum task_subtypes task_subtype) {

  /* Grab handles */
  const struct engine *e = r->e;
  int4 *fparti_fpartj_lparti_lpartj = buf->fparti_fpartj_lparti_lpartj;
  struct gpu_pack_vars *pack_vars = &buf->pv;

  const size_t count_ci = ci->hydro.count;
  const size_t count_cj = cj->hydro.count;

  /*Get the id for the daughter task*/
  const size_t tid = pack_vars->tasks_packed;

  /* Get the relative distance between the pairs, wrapping (For M. S.: what does
   * we mean by wrapping??). */
  double shift[3] = {0.0, 0.0, 0.0};
  for (int k = 0; k < 3; k++) {
    if (cj->loc[k] - ci->loc[k] < -e->s->dim[k] / 2.0) {
      shift[k] = e->s->dim[k];
    } else if (cj->loc[k] - ci->loc[k] > e->s->dim[k] / 2.0) {
      shift[k] = -e->s->dim[k];
    }
  }

  /* Pass into packing as double3 (Attempt at making packing more efficient) */
  double3 shift_tmp = {shift[0], shift[1], shift[2]};

  /* Find first parts in task for ci and cj. Packed_tmp is index for cell i.
   * packed_tmp+1 is index for cell j */
  ////////////////////////
  // THIS IS A PROBLEM!!!
  ////////////////////////
  fparti_fpartj_lparti_lpartj[tid].x = pack_vars->count_parts;
  fparti_fpartj_lparti_lpartj[tid].y = pack_vars->count_parts + count_ci;

  /* This re-arranges the particle data from cell->hydro->parts into a
  long array of part structs*/
  if (task_subtype == task_subtype_gpu_pack_d) {
    gpu_pack_pair_density(buf, ci, cj, shift_tmp);
  } else if (task_subtype == task_subtype_gpu_pack_g) {
    gpu_pack_pair_gradient(buf, ci, cj, shift_tmp);
  } else if (task_subtype == task_subtype_gpu_pack_f) {
    gpu_pack_pair_force(buf, ci, cj, shift_tmp);
  } else {
    error("Unknown task subtype %s", subtaskID_names[task_subtype]);
  }

  /* Find last parts in task for ci and cj*/
  ////////////////////////
  // THIS IS A PROBLEM!!!
  ////////////////////////
  fparti_fpartj_lparti_lpartj[tid].z = pack_vars->count_parts - count_cj;
  fparti_fpartj_lparti_lpartj[tid].w = pack_vars->count_parts;

  /* Tell the cells they have been packed */
#ifdef CUDA_DEBUG
  /* TODO: REMOVE THIS? PREVENTS CELL FROM BEING CONST ARGUMENT*/
  /* ci->pack_done++; */
  /* cj->pack_done++; */
#endif

  /* Identify first particle for each bundle of tasks */
  const int bundle_size = pack_vars->bundle_size;
  if (tid % bundle_size == 0) {
    int bid = tid / bundle_size;
    pack_vars->bundle_first_part[bid] = fparti_fpartj_lparti_lpartj[tid].x;

    /* A. Nasar: This is possibly a problem! */
    pack_vars->bundle_first_task_list[bid] = tid;
  }

  /* Record that we have now done a pair pack task & increment number of tasks
   * to offload*/
  pack_vars->tasks_packed++;
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

  /* Grab pack_vars */
  struct gpu_pack_vars *pack_vars = &buf->pv;

  /* Identify the number of GPU bundles to run in ideal case */
  size_t n_bundles = pack_vars->n_bundles;

  /* How many tasks have we packed? */
  const size_t tasks_packed = pack_vars->tasks_packed;

  /* How many tasks should be in a bundle? */
  const size_t bundle_size = pack_vars->bundle_size;

  /* Special case for incomplete bundles (when having leftover tasks not enough
   * to fill a bundle) */
  if (pack_vars->launch_leftovers) {
    n_bundles = (tasks_packed + bundle_size - 1) / bundle_size;
    if (tasks_packed == 0)
      error("zero tasks packed but somehow got into GPU loop");
    pack_vars->bundle_first_part[n_bundles] =
        buf->task_first_part[tasks_packed - 1].x;
  }
  pack_vars->n_bundles_unpack = n_bundles;

  /* Identify the last particle for each bundle of tasks */
  for (size_t bid = 0; bid < n_bundles - 1; bid++) {
    pack_vars->bundle_last_part[bid] = pack_vars->bundle_first_part[bid + 1];
  }

  /* special treatment for the last bundle */
  if (n_bundles > 1)
    pack_vars->bundle_last_part[n_bundles - 1] = pack_vars->count_parts;
  else
    pack_vars->bundle_last_part[0] = pack_vars->count_parts;

  /* Launch the copies for each bundle and run the GPU kernel */
  /* We don't go into this loop if tasks_left_self == 1 as
   n_bundles will be zero DUHDUHDUHDUHHHHHH!!!!!*/
  size_t max_parts;
  for (size_t bid = 0; bid < n_bundles; bid++) {

    max_parts = 0;
    const size_t first_task = bid * bundle_size;
    size_t last_task = (bid + 1) * bundle_size;
    for (size_t tid = bid * bundle_size; tid < (bid + 1) * bundle_size; tid++) {
      if (tid < tasks_packed) {
        /* Get an estimate for the max number of parts per cell in the bundle.
         * Used for determining the number of GPU CUDA blocks*/
        size_t count =
            buf->task_first_part[tid].y - buf->task_first_part[tid].x;
        max_parts = max(max_parts, count);
        last_task = tid;
      }
    }

    const size_t first_part_tmp = pack_vars->bundle_first_part[bid];
    const size_t bundle_n_parts =
        pack_vars->bundle_last_part[bid] - first_part_tmp;
    const size_t tasksperbundle = pack_vars->tasksperbundle;
    const size_t tasks_left =
        (bid == n_bundles - 1) ? tasks_packed - (n_bundles - 1) * tasksperbundle
                               : tasksperbundle;

    /* Will launch a 2d grid of GPU thread blocks (number of tasks is
       the y dimension and max_parts is the x dimension */
    const size_t numBlocks_y = tasks_left;
    const size_t numBlocks_x = (max_parts + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const size_t bundle_first_task = pack_vars->bundle_first_task_list[bid];

    /* Copy data over to GPU */
    /* First the meta-data */
    cudaMemcpyAsync(&buf->d_task_first_part[first_task],
                    &buf->task_first_part[first_task],
                    (last_task + 1 - first_task) * sizeof(int2),
                    cudaMemcpyHostToDevice, stream[bid]);

    /* Now the actual particle data */
    if (task_subtype == task_subtype_gpu_pack_d) {

      cudaMemcpyAsync(&buf->d_parts_send_d[first_part_tmp],
                      &buf->parts_send_d[first_part_tmp],
                      bundle_n_parts * sizeof(struct gpu_part_send_d),
                      cudaMemcpyHostToDevice, stream[bid]);

    } else if (task_subtype == task_subtype_gpu_pack_g) {

      cudaMemcpyAsync(&buf->d_parts_send_g[first_part_tmp],
                      &buf->parts_send_g[first_part_tmp],
                      bundle_n_parts * sizeof(struct gpu_part_send_g),
                      cudaMemcpyHostToDevice, stream[bid]);

    } else if (task_subtype == task_subtype_gpu_pack_f) {

      cudaMemcpyAsync(&buf->d_parts_send_f[first_part_tmp],
                      &buf->parts_send_f[first_part_tmp],
                      bundle_n_parts * sizeof(struct gpu_part_send_f),
                      cudaMemcpyHostToDevice, stream[bid]);

    } else {
      error("Unknown task subtype %s", subtaskID_names[task_subtype]);
    }

#ifdef CUDA_DEBUG
    cudaError_t cu_error = cudaPeekAtLastError();
    if (cu_error != cudaSuccess) {
      error("CUDA error in task subtype %s H2D memcpy: '%s' cpuid id is: %i",
            subtaskID_names[task_subtype], cudaGetErrorString(cu_error),
            r->cpuid);
    }
#endif

    /* Launch the kernel */
    if (task_subtype == task_subtype_gpu_pack_d) {
      gpu_launch_self_density(buf->d_parts_send_d, buf->d_parts_recv_d, d_a,
                              d_H, stream[bid], numBlocks_x, numBlocks_y,
                              bundle_first_task, buf->d_task_first_part);

    } else if (task_subtype == task_subtype_gpu_pack_g) {

      gpu_launch_self_gradient(buf->d_parts_send_g, buf->d_parts_recv_g, d_a,
                               d_H, stream[bid], numBlocks_x, numBlocks_y,
                               bundle_first_task, buf->d_task_first_part);

    } else if (task_subtype == task_subtype_gpu_pack_f) {

      gpu_launch_self_force(buf->d_parts_send_f, buf->d_parts_recv_f, d_a, d_H,
                            stream[bid], numBlocks_x, numBlocks_y,
                            bundle_first_task, buf->d_task_first_part);
    } else {
      error("Unknown task subtype %s", subtaskID_names[task_subtype]);
    }

#ifdef CUDA_DEBUG
    cu_error = cudaPeekAtLastError();
    if (cu_error != cudaSuccess) {
      error("CUDA error in task subtype %s kernel launch: '%s' cpuid id is: %i",
            subtaskID_names[task_subtype], cudaGetErrorString(cu_error),
            r->cpuid);
    }
#endif

    /* Copy back */
    if (task_subtype == task_subtype_gpu_pack_d) {

      cudaMemcpyAsync(&buf->parts_recv_d[first_part_tmp],
                      &buf->d_parts_recv_d[first_part_tmp],
                      bundle_n_parts * sizeof(struct gpu_part_recv_d),
                      cudaMemcpyDeviceToHost, stream[bid]);

    } else if (task_subtype == task_subtype_gpu_pack_g) {

      cudaMemcpyAsync(&buf->parts_recv_g[first_part_tmp],
                      &buf->d_parts_recv_g[first_part_tmp],
                      bundle_n_parts * sizeof(struct gpu_part_recv_g),
                      cudaMemcpyDeviceToHost, stream[bid]);

    } else if (task_subtype == task_subtype_gpu_pack_f) {

      cudaMemcpyAsync(&buf->parts_recv_f[first_part_tmp],
                      &buf->d_parts_recv_f[first_part_tmp],
                      bundle_n_parts * sizeof(struct gpu_part_recv_f),
                      cudaMemcpyDeviceToHost, stream[bid]);
    } else {
      error("Unknown task subtype %s", subtaskID_names[task_subtype]);
    }

    /* Record this event */
    cudaEventRecord(buf->event_end[bid], stream[bid]);

#ifdef CUDA_DEBUG
    cu_error = cudaPeekAtLastError();
    if (cu_error != cudaSuccess) {
      error("CUDA error in task subtype %s D2H memcpy: %s cpuid id is: %i",
            subtaskID_names[task_subtype], cudaGetErrorString(cu_error),
            r->cpuid);
    }
#endif

  } /*End of looping over bundles to launch in streams*/

  /* Make sure all the kernels and copies back are finished */
  /* cudaDeviceSynchronize(); */
}

/**
 * @ brief Unpacks the completed self tasks for the corresponding task_subtype
 */
__attribute__((always_inline)) INLINE static void runner_doself_gpu_unpack(
    const struct runner *r, struct scheduler *s, struct gpu_offload_data *buf,
    const enum task_subtypes task_subtype) {

  const struct engine *e = r->e;

  /* Grab pack_vars */
  struct gpu_pack_vars *pack_vars = &buf->pv;

  /* Identify the number of GPU bundles to run in ideal case */
  const size_t n_bundles = pack_vars->n_bundles_unpack;

  /*How many tasks have we packed?*/
  const size_t tasks_packed = pack_vars->tasks_packed;

  /*How many tasks should be in a bundle?*/
  const size_t bundle_size = pack_vars->bundle_size;

  /* Copy the data back from the CPU thread-local buffers to the cells */
  /* Pack length counter for use in unpacking */
  size_t pack_length = 0;
  for (size_t bid = 0; bid < n_bundles; bid++) {

    /* cudaStreamSynchronize(stream[bid]); */
    cudaEventSynchronize(buf->event_end[bid]);

    for (size_t tid = bid * bundle_size;
         tid < (bid + 1) * bundle_size && tid < tasks_packed; tid++) {

      struct cell *c = pack_vars->ci_list[tid];
      struct task *t = pack_vars->task_list[tid];

      while (cell_locktree(c)) {
        ; /* spin until we acquire the lock */
      }

      /* Anything to do here? */
      if (c->hydro.count == 0) return;
      if (!cell_is_active_hydro(c, e)) return;

      size_t count = c->hydro.count;

#ifdef SWIFT_DEBUG_CHECKS
      if (pack_length + count >= pack_vars->count_max_parts) {
        error(
            "Exceeded count_max_parts. Make arrays bigger! pack_length is "
            "%lu, count is %lu, max_parts is %lu",
            pack_length, count, pack_vars->count_max_parts);
      }
#endif

      /* Do the copy */
      if (task_subtype == task_subtype_gpu_pack_d) {

        gpu_unpack_self_density(c, buf->parts_recv_d, pack_length, count, e);
        /* Record things for debugging */
#ifdef CUDA_DEBUG
        c->gpu_done++;
#endif

      } else if (task_subtype == task_subtype_gpu_pack_g) {

        gpu_unpack_self_gradient(c, buf->parts_recv_g, pack_length, count, e);
        /* Record things for debugging */
#ifdef CUDA_DEBUG
        c->gpu_done_g++;
#endif

      } else if (task_subtype == task_subtype_gpu_pack_f) {

        gpu_unpack_self_force(c, buf->parts_recv_f, pack_length, count, e);
        /* Record things for debugging */
#ifdef CUDA_DEBUG
        c->gpu_done_f++;
#endif

      } else {
        error("Unknown task subtype %s", subtaskID_names[task_subtype]);
      }

      /* Increase our index in the buffer with the newly unpacked size. */
      pack_length += count;

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
  pack_vars->count_parts = 0;
  pack_vars->tasks_packed = 0;
  pack_vars->n_bundles_unpack = 0;
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
  char launch_leftovers = buf->pv.launch_leftovers;

  /* Packed enough tasks. Let's go*/
  char launch = buf->pv.launch;

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
  char launch_leftovers = buf->pv.launch_leftovers;

  /* Packed enough tasks. Let's go*/
  char launch = buf->pv.launch;

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
  char launch_leftovers = buf->pv.launch_leftovers;

  /*Packed enough tasks let's go*/
  char launch = buf->pv.launch;

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
 * Generic function to launch GPU pair tasks
 */
__attribute__((always_inline)) INLINE static void runner_dopair_gpu_launch(
    const struct runner *r, struct gpu_offload_data *restrict buf,
    cudaStream_t *stream, const float d_a, const float d_H,
    const enum task_subtypes task_subtype) {

  /* Grab handles */
  struct gpu_pack_vars *pack_vars = &buf->pv;
  int4 *fparti_fpartj_lparti_lpartj_dens = buf->fparti_fpartj_lparti_lpartj;
  cudaEvent_t *pair_end = buf->event_end;

  /* Identify the number of GPU bundles to run in ideal case*/
  size_t n_bundles = pack_vars->n_bundles;

  /* How many tasks have we packed? */
  const size_t tasks_packed = pack_vars->tasks_packed;

  /* How many tasks should be in a bundle? */
  const size_t bundle_size = pack_vars->bundle_size;

  /* Special case for incomplete bundles (when having leftover tasks not enough
   * to fill a bundle) */
  if (pack_vars->launch_leftovers) {
    n_bundles = (tasks_packed + bundle_size - 1) / bundle_size;
    pack_vars->bundle_first_part[n_bundles] =
        fparti_fpartj_lparti_lpartj_dens[tasks_packed - 1].x;
  }

  /* Identify the last particle for each bundle of tasks */
  for (size_t bid = 0; bid < n_bundles - 1; bid++) {
    pack_vars->bundle_last_part[bid] = pack_vars->bundle_first_part[bid + 1];
  }

  /* special treatment for the case of 1 bundle */
  if (n_bundles > 1)
    pack_vars->bundle_last_part[n_bundles - 1] = pack_vars->count_parts;
  else
    pack_vars->bundle_last_part[0] = pack_vars->count_parts;

  /* Launch the copies for each bundle and run the GPU kernel */
  for (size_t bid = 0; bid < n_bundles; bid++) {

    int max_parts_i = 0;
    int max_parts_j = 0;
    for (size_t tid = bid * bundle_size; tid < (bid + 1) * bundle_size; tid++) {
      if (tid < tasks_packed) {
        /* Get an estimate for the max number of parts per cell in each bundle.
         * Used for determining the number of GPU CUDA blocks */
        int count_i = fparti_fpartj_lparti_lpartj_dens[tid].z -
                      fparti_fpartj_lparti_lpartj_dens[tid].x;
        max_parts_i = max(max_parts_i, count_i);
        int count_j = fparti_fpartj_lparti_lpartj_dens[tid].w -
                      fparti_fpartj_lparti_lpartj_dens[tid].y;
        max_parts_j = max(max_parts_j, count_j);
      }
    }
    const size_t first_part_tmp_i = pack_vars->bundle_first_part[bid];
    const size_t bundle_n_parts =
        pack_vars->bundle_last_part[bid] - first_part_tmp_i;

    /* Transfer memory to device */
    if (task_subtype == task_subtype_gpu_launch_d) {

      cudaMemcpyAsync(&buf->d_parts_send_d[first_part_tmp_i],
                      &buf->parts_send_d[first_part_tmp_i],
                      bundle_n_parts * sizeof(struct gpu_part_send_d),
                      cudaMemcpyHostToDevice, stream[bid]);
    } else if (task_subtype == task_subtype_gpu_launch_g) {
      cudaMemcpyAsync(&buf->d_parts_send_g[first_part_tmp_i],
                      &buf->parts_send_g[first_part_tmp_i],
                      bundle_n_parts * sizeof(struct gpu_part_send_g),
                      cudaMemcpyHostToDevice, stream[bid]);
    } else if (task_subtype == task_subtype_gpu_launch_f) {
      cudaMemcpyAsync(&buf->d_parts_send_f[first_part_tmp_i],
                      &buf->parts_send_f[first_part_tmp_i],
                      bundle_n_parts * sizeof(struct gpu_part_send_f),
                      cudaMemcpyHostToDevice, stream[bid]);
    } else {
      error("Unknown task subtype %s", subtaskID_names[task_subtype]);
    }

#ifdef CUDA_DEBUG
    cudaError_t cu_error = cudaPeekAtLastError();
    if (cu_error != cudaSuccess) {
      error(
          "CUDA error with task subtype %s H2D async memcpy ci: %s cpuid id "
          "is: %i"
          "Something's up with your cuda code first_part %lu bundle size %lu",
          subtaskID_names[task_subtype], cudaGetErrorString(cu_error), r->cpuid,
          first_part_tmp_i, bundle_n_parts);
    }
#endif

    /* LAUNCH THE GPU KERNELS for ci & cj */
    /* Setup 2d grid of GPU thread blocks for ci (number of tasks is
     * the y dimension and max_parts is the x dimension */
    int numBlocks_y = 0; /* tasks_left; //Changed this to 1D grid of blocks so
                            this is no longer necessary */
    int numBlocks_x = (bundle_n_parts + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int bundle_part_0 = pack_vars->bundle_first_part[bid];

    /* Launch the kernel for ci using data for ci and cj */
    if (task_subtype == task_subtype_gpu_launch_d) {
      gpu_launch_pair_density(buf->d_parts_send_d, buf->d_parts_recv_d, d_a,
                              d_H, stream[bid], numBlocks_x, numBlocks_y,
                              bundle_part_0, bundle_n_parts);
    } else if (task_subtype == task_subtype_gpu_launch_g) {
      gpu_launch_pair_gradient(buf->d_parts_send_g, buf->d_parts_recv_g, d_a,
                               d_H, stream[bid], numBlocks_x, numBlocks_y,
                               bundle_part_0, bundle_n_parts);
    } else if (task_subtype == task_subtype_gpu_launch_f) {
      gpu_launch_pair_force(buf->d_parts_send_f, buf->d_parts_recv_f, d_a, d_H,
                            stream[bid], numBlocks_x, numBlocks_y,
                            bundle_part_0, bundle_n_parts);
    } else {
      error("Unknown task subtype %s", subtaskID_names[task_subtype]);
    }

#ifdef CUDA_DEBUG
    cu_error = cudaPeekAtLastError();
    if (cu_error != cudaSuccess) {
      error(
          "CUDA error with task subtype %s kernel launch: %s cpuid id is: %i\n "
          "nbx %i nby %i max_parts_i %i max_parts_j %i\n"
          "Something's up with kernel launch.",
          subtaskID_names[task_subtype], cudaGetErrorString(cu_error), r->cpuid,
          numBlocks_x, numBlocks_y, max_parts_i, max_parts_j);
    }
#endif

    /* Copy results back to CPU BUFFERS */
    if (task_subtype == task_subtype_gpu_launch_d) {
      cudaMemcpyAsync(&buf->parts_recv_d[first_part_tmp_i],
                      &buf->d_parts_recv_d[first_part_tmp_i],
                      bundle_n_parts * sizeof(struct gpu_part_recv_d),
                      cudaMemcpyDeviceToHost, stream[bid]);
    } else if (task_subtype == task_subtype_gpu_launch_g) {
      cudaMemcpyAsync(&buf->parts_recv_g[first_part_tmp_i],
                      &buf->d_parts_recv_g[first_part_tmp_i],
                      bundle_n_parts * sizeof(struct gpu_part_recv_g),
                      cudaMemcpyDeviceToHost, stream[bid]);
    } else if (task_subtype == task_subtype_gpu_launch_f) {
      cudaMemcpyAsync(&buf->parts_recv_f[first_part_tmp_i],
                      &buf->d_parts_recv_f[first_part_tmp_i],
                      bundle_n_parts * sizeof(struct gpu_part_recv_f),
                      cudaMemcpyDeviceToHost, stream[bid]);
    } else {
      error("Unknown task subtype %s", subtaskID_names[task_subtype]);
    }

    /* Issue event to be recorded by GPU after copy back to CPU */
    cudaEventRecord(pair_end[bid], stream[bid]);

#ifdef CUDA_DEBUG
    cu_error = cudaPeekAtLastError();
    if (cu_error != cudaSuccess) {
      error(
          "CUDA error with task subtype %s D2H memcpy: %s cpuid id is: %i\n"
          "Something's up with your cuda code",
          subtaskID_names[task_subtype], cudaGetErrorString(cu_error),
          r->cpuid);
    }
#endif

  } /*End of looping over bundles to launch in streams*/

  /* Issue synchronisation commands for all events recorded by GPU
   * Should swap with one cuda Device Synchronise really if we decide to go this
   * way with unpacking done separately */
  for (size_t bid = 0; bid < n_bundles; bid++) {
    cudaEventSynchronize(pair_end[bid]);
  }
}

/**
 * Wrapper to launch density pair tasks on the GPU.
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
 * Wrapper to launch density pair tasks on the GPU.
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
 * Generic function to unpack a GPU pair task depending on the task subtype.
 * TODO: DOCUMENT WHAT IS NPACKED??
 */
__attribute__((always_inline)) INLINE static void runner_dopair_gpu_unpack(
    const struct runner *r, struct scheduler *s,
    struct gpu_offload_data *restrict buf, const int npacked,
    const enum task_subtypes task_subtype) {

  /* Grab handles */
  struct gpu_pack_vars *pack_vars = &buf->pv;
  int n_leaves_found = buf->pv.n_leaves_found;

  struct cell **ci_d = buf->ci_d;
  struct cell **cj_d = buf->cj_d;
  struct cell **ci_top = buf->ci_top;
  struct cell **cj_top = buf->cj_top;
  int **f_l_daughters = buf->first_and_last_daughters;

  /////////////////////////////////
  // Should this be reset to zero HERE???
  /////////////////////////////////
  size_t pack_length_unpack = 0;
  /*Loop over top level tasks*/
  for (size_t topid = 0; topid < pack_vars->top_tasks_packed; topid++) {

    while (cell_locktree(ci_top[topid])) {
      ; /* spin until we acquire the lock */
    }
    while (cell_locktree(cj_top[topid])) {
      ; /* spin until we acquire the lock */
    }

    /* Loop through each daughter task */
    for (int tid = f_l_daughters[topid][0]; tid < f_l_daughters[topid][1];
         tid++) {

      /*Get pointers to the leaf cells*/
      struct cell *cii_l = ci_d[tid];
      struct cell *cjj_l = cj_d[tid];

      /* Not a typo: task subtype is task_subtype_pack_*. The unpacking gets
       * called at the end of packing, running, and possibly launching */
      if (task_subtype == task_subtype_gpu_pack_d) {
        /* TODO: WHY IS THERE A FACTOR OF 2 HERE? LEAVE A COMMENT. */
        gpu_unpack_pair_density(r, cii_l, cjj_l, buf->parts_recv_d,
                                &pack_length_unpack,
                                2 * pack_vars->count_max_parts);
      } else if (task_subtype == task_subtype_gpu_pack_g) {
        gpu_unpack_pair_gradient(r, cii_l, cjj_l, buf->parts_recv_g,
                                 &pack_length_unpack,
                                 2 * pack_vars->count_max_parts);
      } else if (task_subtype == task_subtype_gpu_pack_f) {
        gpu_unpack_pair_force(r, cii_l, cjj_l, buf->parts_recv_f,
                              &pack_length_unpack,
                              2 * pack_vars->count_max_parts);
      } else {
        error("Unknown task subtype %s", subtaskID_names[task_subtype]);
      }
    }

    /* Release the cells */
    /* TODO: WHERE HAVE THEY BEEN LOCKED? */
    cell_unlocktree(ci_top[topid]);
    cell_unlocktree(cj_top[topid]);

    /* For some reason the code fails if we get a leaf pair task
     * this if->continue statement stops the code from trying to unlock same
     * cells twice*/
    if (topid == pack_vars->top_tasks_packed - 1 && npacked != n_leaves_found) {
      continue;
    }

    /* TODO: DOCUMENT WHAT'S HAPPENING HERE AND WHY */
    enqueue_dependencies(s, pack_vars->top_task_list[topid]);
    pthread_mutex_lock(&s->sleep_mutex);
    atomic_dec(&s->waiting);
    pthread_cond_broadcast(&s->sleep_cond);
    pthread_mutex_unlock(&s->sleep_mutex);
  }
}

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
  struct gpu_pack_vars *pack_vars = &buf->pv;
  int n_leaves_found = buf->pv.n_leaves_found;
  int **f_l_daughters = buf->first_and_last_daughters;
  int top_tasks_packed = pack_vars->top_tasks_packed;

  /* TODO: WHY IS THIS HERE AND NOT IN RUNNER_GPU_RECURSE??? */
  pack_vars->n_daughters_total += n_leaves_found;

  /* Keep separate */
  f_l_daughters[top_tasks_packed][0] = pack_vars->n_daughters_packed_index;

  /* Keep separate */
  buf->ci_top[top_tasks_packed] = ci;
  buf->cj_top[top_tasks_packed] = cj;

  /* TODO ABOUZIED: WE HAVE BOTH LEAVES TOTAL AND DAUGHTERS TOTAL. DO WE NEED
   * BOTH? */
  pack_vars->n_leaves_total += n_leaves_found;

  int first_cell_to_move = pack_vars->n_daughters_packed_index;
  int n_daughters_left = pack_vars->n_daughters_total;

  /* Get pointer to top level task. Needed to enqueue deps*/
  pack_vars->top_task_list[top_tasks_packed] = t;

  /* Increment how many top tasks we've packed */
  pack_vars->top_tasks_packed++;

  /* A. Nasar: Remove this from struct as not needed. Was only used for
   * de-bugging */
  /* TODO: Abouzied, do this please! */
  /* How many daughter tasks do we want to offload at once?*/
  size_t target_n_tasks_tmp = pack_vars->target_n_tasks;

  /* Counter for how many tasks we've packed */
  int npacked = 0;
  int launched = 0;

  /* A. Nasar: Loop through the daughter tasks we found */
  int copy_index = pack_vars->n_daughters_packed_index;

  /* not strictly true!!! Could be that we packed and moved on without launching
   */
  int had_prev_task = 0;
  if (pack_vars->n_daughters_packed_index > 0) had_prev_task = 1;

  cell_unlocktree(ci);
  cell_unlocktree(cj);

  while (npacked < n_leaves_found) {

    top_tasks_packed = pack_vars->top_tasks_packed;
    struct cell *cii = buf->ci_d[copy_index];
    struct cell *cjj = buf->cj_d[copy_index];

    if (t->subtype == task_subtype_gpu_pack_d) {
      runner_dopair_gpu_pack_density(r, buf, cii, cjj, t);
    } else if (t->subtype == task_subtype_gpu_pack_g) {
      runner_dopair_gpu_pack_gradient(r, buf, cii, cjj, t);
    } else if (t->subtype == task_subtype_gpu_pack_f) {
      runner_dopair_gpu_pack_force(r, buf, cii, cjj, t);
    } else {
      error("Unknown task subtype %s", subtaskID_names[t->subtype]);
    }

    /* record number of tasks we've copied from last launch */
    copy_index++;

    /* Record how many daughters we've packed in total for while loop */
    npacked++;
    first_cell_to_move++;

    f_l_daughters[top_tasks_packed - 1][1] =
        f_l_daughters[top_tasks_packed - 1][0] + copy_index;
    if (had_prev_task)
      f_l_daughters[top_tasks_packed - 1][1] =
          f_l_daughters[top_tasks_packed - 1][0] + copy_index -
          pack_vars->n_daughters_packed_index;

    if (pack_vars->tasks_packed == target_n_tasks_tmp) pack_vars->launch = 1;

    if (pack_vars->launch ||
        (pack_vars->launch_leftovers && npacked == n_leaves_found)) {

      launched = 1;

      if (t->subtype == task_subtype_gpu_pack_d) {
        /* Launch the tasks */
        runner_dopair_gpu_launch_density(r, buf, stream, d_a, d_H);
        /* Unpack the results into CPU memory */
        runner_dopair_gpu_unpack_density(r, s, buf, npacked);
      } else if (t->subtype == task_subtype_gpu_pack_g) {
        /* Launch the tasks */
        runner_dopair_gpu_launch_gradient(r, buf, stream, d_a, d_H);
        /* Unpack the results into CPU memory */
        runner_dopair_gpu_unpack_gradient(r, s, buf, npacked);
      } else if (t->subtype == task_subtype_gpu_pack_f) {
        /* Launch the tasks */
        runner_dopair_gpu_launch_force(r, buf, stream, d_a, d_H);
        /* runner_dopair_launch_f( */
        /* Unpack the results into CPU memory */
        runner_dopair_gpu_unpack_force(r, s, buf, npacked);
        /* runner_dopair_unpack_f( */
      } else {
        error("Unknown task subtype %s", subtaskID_names[t->subtype]);
      }

      if (npacked == n_leaves_found) {
        pack_vars->n_daughters_total = 0;
        pack_vars->top_tasks_packed = 0;
      }
      /* Special treatment required here. Launched but have not packed all tasks
       */
      else {
        /* If we launch but still have daughters left re-set this task to be
         * the first in the list so that we can continue packing correctly */
        pack_vars->top_task_list[0] = t;
        /* Move all tasks forward in list so that the first next task will be
         * packed to index 0 Move remaining cell indices so that their indexing
         * starts from zero and ends in n_daughters_left */
        for (int i = first_cell_to_move; i < n_daughters_left; i++) {
          int shuffle = i - first_cell_to_move;
          buf->ci_d[shuffle] = buf->ci_d[i];
          buf->cj_d[shuffle] = buf->cj_d[i];
        }
        copy_index = 0;
        f_l_daughters[0][0] = 0;
        f_l_daughters[0][1] = n_daughters_left - first_cell_to_move;

        buf->cj_top[0] = cj;
        buf->ci_top[0] = ci;

        n_daughters_left -= first_cell_to_move;
        first_cell_to_move = 0;

        pack_vars->top_tasks_packed = 1;
        had_prev_task = 0;
      }
      pack_vars->tasks_packed = 0;
      pack_vars->count_parts = 0;
      pack_vars->launch = 0;
    }
    /* case when we have launched then gone back to pack but did not pack
     * enough to launch again */
    else if (npacked == n_leaves_found) {
      if (launched == 1) {
        pack_vars->n_daughters_total = n_daughters_left;
        buf->cj_top[0] = cj;
        buf->ci_top[0] = ci;
        f_l_daughters[0][0] = 0;
        f_l_daughters[0][1] = n_daughters_left;
        pack_vars->top_tasks_packed = 1;
        pack_vars->top_task_list[0] = t;
        launched = 0;
      } else {
        f_l_daughters[top_tasks_packed - 1][0] =
            pack_vars->n_daughters_packed_index;
        f_l_daughters[top_tasks_packed - 1][1] = pack_vars->n_daughters_total;
      }
      pack_vars->launch = 0;
    }
    pack_vars->launch = 0;
  }

  /* A. Nasar: Launch-leftovers counter re-set to zero and cells unlocked */
  pack_vars->launch_leftovers = 0;
  pack_vars->launch = 0;
}

/**
 * Top level runner function to solve hydro density pair tasks on GPU.
 */
static void runner_dopair_gpu_density(const struct runner *r,
                                      struct scheduler *s, struct cell *ci,
                                      struct cell *cj,
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
  s->queues[qid].n_packs_pair_left_d--;
  if (s->queues[qid].n_packs_pair_left_d < 1) buf->pv.launch_leftovers = 1;
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
  if (s->queues[qid].n_packs_pair_left_g < 1) buf->pv.launch_leftovers = 1;
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
  if (s->queues[qid].n_packs_pair_left_f < 1) buf->pv.launch_leftovers = 1;
  (void)lock_unlock(&s->queues[qid].lock);

  /* pack the data and run, if enough data has been gathered */
  runner_dopair_gpu_pack_and_launch(r, s, ci, cj, buf, t, stream, d_a, d_H);
}

#ifdef __cplusplus
}
#endif

#endif /* RUNNER_GPU_PACK_FUNCTIONS_H */
