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
#ifndef RUNNER_GPU_PACK_FUNCTIONS_H
#define RUNNER_GPU_PACK_FUNCTIONS_H

#include "../config.h"
#include "active.h"
#include "engine.h"
#include "inline.h"
#include "runner.h"
#include "timers.h"

/* Temporary warning during dev works. */
#if !(defined(HAVE_CUDA) || defined(HAVE_HIP))
#pragma warning "Don't have CUDA nor HIP"
#endif

#ifdef WITH_CUDA
#include "cuda/gpu_offload_data.h"
#include "cuda/gpu_part_pack_functions.h"
#include "cuda/gpu_part_structs.h"
#endif

#ifdef WITH_HIP
#pragma error "Header inclusions missing"
#endif

/**
 * @brief Generic function to unpack data received from the GPU depending on
 * the task subtype.
 *
 * @param r the #runner
 * @param s the #scheduler
 * @param buf the particle data buffers
 * @param npacked how many (pairs of) leaf cells have been packed during the
 * current pair task offloading call. May differ from the total number of
 * packed leaf cell pairs if there have been leftover leaf cell pairs from a
 * previous task.
 * @param task_subtype this task's subtype
 */
__attribute__((always_inline)) INLINE static void runner_gpu_unpack_pre_sorted(
    const struct runner *r, struct scheduler *s,
    struct gpu_offload_data *restrict buf, const int npacked,
    const enum task_subtypes task_subtype) {

  /* Grab handles */
  struct gpu_pack_metadata *md = &buf->md;
  const struct engine *e = r->e;

  /*Let's unpack the unique particle data first.
   * We get on to enqueueing dependencies after this*/
  int unpack_index = 0;
  if(task_subtype == task_subtype_gpu_density){
    for(int i = 0; i < md->n_unique; i++){
      struct cell * c = md->unique_cells[i];
      const int count = c->hydro.count;
      while(cell_locktree(c)){
        ;
      }
      if (cell_is_active_hydro(c, e))
      gpu_unpack_part_density(c, buf->parts_recv_d, unpack_index,
          count, e);
      unpack_index += count + 1;
      cell_unlocktree(c);
    }
  }
  else if(task_subtype == task_subtype_gpu_force){
    for(int i = 0; i < md->n_unique; i++){
      struct cell * c = md->unique_cells[i];
      const int count = c->hydro.count;
      while(cell_locktree(c)){
        ;
      }
      if (cell_is_active_hydro(c, e))
      gpu_unpack_part_force(c, buf->parts_recv_f, unpack_index,
          count, e);
      unpack_index += count + 1;
      cell_unlocktree(c);
    }
  }
  else if(task_subtype == task_subtype_gpu_gradient){
    for(int i = 0; i < md->n_unique; i++){
      struct cell * c = md->unique_cells[i];
      const int count = c->hydro.count;
      while(cell_locktree(c)){
        ;
      }
      if (cell_is_active_hydro(c, e))
      gpu_unpack_part_gradient(c, buf->parts_recv_g, unpack_index,
          count, e);
      unpack_index += count + 1;
      cell_unlocktree(c);
    }
  }


  /* Loop over all tasks that we have offloaded */
  for (int tid = 0; tid < md->tasks_in_list; tid++) {

    /* If we haven't finished packing the currently handled task's leaf cells,
     * we mustn't unlock its dependencies yet. ("Currently handled task" is
     * the one for which the offloading cycle is currently underway in
     * runner_gpu_pack_and_launch) */
    if ((tid == md->tasks_in_list - 1) && (npacked != md->task_n_leaves))
    	continue;

    /* If we're here, we're completely done with this task. Mark it as
     * completed. */

    /* schedule my dependencies */
    scheduler_enqueue_dependencies(s, md->task_list[tid]);

    /* Tell the scheduler's bookkeeping that this task is done */
    pthread_mutex_lock(&s->sleep_mutex);
    atomic_dec(&s->waiting);
    pthread_cond_broadcast(&s->sleep_cond);
    pthread_mutex_unlock(&s->sleep_mutex);

    /* Mark the task as done. */
    md->task_list[tid]->skip = 1;

  } /* Loop over tasks in list */
}

/**
 * @brief Wrapper to unpack the density data.
 *
 * @param r the #runner
 * @param s the #scheduler
 * @param buf the particle data buffers
 * @param npacked how many leaf cell pairs have been packed during the current
 * pair task offloading call. May differ from the total number of packed leaf
 * cell pairs if there have been leftover leaf cell pairs from a previous task.
 */
__attribute__((always_inline)) INLINE static void runner_gpu_unpack_density(
    const struct runner *r, struct scheduler *s,
    struct gpu_offload_data *restrict buf, const int npacked) {

  TIMER_TIC;

  runner_gpu_unpack_pre_sorted(r, s, buf, npacked, task_subtype_gpu_density);

  TIMER_TOC(timer_gpu_unpack_d);
}

/**
 * @brief Wrapper to unpack gradient data.
 *
 * @param r the #runner
 * @param s the #scheduler
 * @param buf the particle data buffers
 * @param npacked how many leaf cell pairs have been packed during the current
 * pair task offloading call. May differ from the total number of packed leaf
 * cell pairs if there have been leftover leaf cell pairs from a previous task.
 */
__attribute__((always_inline)) INLINE static void runner_gpu_unpack_gradient(
    const struct runner *r, struct scheduler *s,
    struct gpu_offload_data *restrict buf, const int npacked) {

  TIMER_TIC;

  runner_gpu_unpack_pre_sorted(r, s, buf, npacked, task_subtype_gpu_gradient);

  TIMER_TOC(timer_gpu_unpack_g);
}

/**
 * @brief Wrapper to unpack the force data.
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

  runner_gpu_unpack_pre_sorted(r, s, buf, npacked, task_subtype_gpu_force);

  TIMER_TOC(timer_gpu_unpack_f);
}

#endif /* RUNNER_GPU_PACK_FUNCTIONS_H */
