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

#include "logging_struct.h"

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
 * @brief Generic function to pack data of a single (pair of) leaf cells into
 * CPU-side buffers destined for GPU offloading
 *
 * @param r the #runner
 * @param buf the CPU-side buffer to copy into
 * @param ci first leaf #cell
 * @param cj second leaf #cell.
 * @param task_subtype this task's subtype
 */
__attribute__((always_inline)) INLINE static void runner_gpu_pack(
    const struct runner *r, struct gpu_offload_data *restrict buf,
    const struct cell *ci, const struct cell *cj,
    const enum task_subtypes task_subtype,
    struct logging_data* logdata) {

  /* Grab handles */
  const struct engine *e = r->e;
  struct gpu_pack_metadata *md = &buf->md;

#ifdef SWIFT_DEBUG_CHECKS
  if (ci == NULL) error("Got NULL cell ci?");
  if (cj == NULL) error("Got NULL cell cj?");
#endif

  const int count_ci = ci->hydro.count;
  const int count_cj = cj->hydro.count;

#ifdef SWIFT_DEBUG_CHECKS
  /* Anything to do here? */
  if (count_ci == 0 || count_cj == 0)
    error("Empty cells should've been excluded during the recursion.");
#endif

  /* Get how many particles we've packed until now */
  int pack_ind = md->count_parts;

#ifdef SWIFT_DEBUG_CHECKS
  int last_ind = pack_ind + count_ci;
  if (ci != cj) last_ind += count_cj; /* packing pair interaction */
  if (last_ind >= md->params.part_buffer_size) {
    error(
        "Exceeded particle buffer size. Increase "
        "Scheduler:gpu_part_buffer_size."
        "ind=%d, counts=%d %d, buffer_size=%d, task_subtype=%s, is self "
        "task?=%d",
        pack_ind, count_ci, count_cj, md->params.part_buffer_size,
        subtaskID_names[task_subtype], ci == cj);
  }
#endif

  /* Get first and last particles of cell i */
  const int cis = pack_ind;
  const int cie = pack_ind + count_ci;

  if (ci == cj) { /* This is a self interaction. */

    const double shift[3] = {0.0, 0.0, 0.0};

    /* Pack the data into the CPU-side buffers for offloading. */
    ticks tic_start = getticks();
    if (task_subtype == task_subtype_gpu_density) {
      gpu_pack_part_density(ci, buf->parts_send_d, pack_ind, shift, cis, cie);
    } else if (task_subtype == task_subtype_gpu_gradient) {
      gpu_pack_part_gradient(ci, buf->parts_send_g, pack_ind, shift, cis, cie);
    } else if (task_subtype == task_subtype_gpu_force) {
      gpu_pack_part_force(ci, buf->parts_send_f, pack_ind, shift, cis, cie);
    }
#ifdef SWIFT_DEBUG_CHECKS
    else {
      error("Unknown task subtype %s", subtaskID_names[task_subtype]);
    }
#endif
    ticks tic_end = getticks();

    char subtype = '0';
    if (task_subtype == task_subtype_gpu_density) {
      subtype = 'd';
    } else if (task_subtype == task_subtype_gpu_gradient) {
      subtype = 'g';
    } else if (task_subtype == task_subtype_gpu_force) {
      subtype = 'f';
    }
    struct logging_entry ent = {subtype, 'p', ci->hydro.parts - logdata->all_parts, ci->hydro.count, clocks_diff_ticks(tic_end, tic_start) };
    logdata->entries[logdata->count] = ent;
    logdata->count++;

  } else { /* This is a pair interaction. */

    /* Get the relative distance between the pairs and apply wrapping in case
     * of periodic boundary conditions */
    double shift[3] = {0., 0., 0.};
    for (int k = 0; k < 3; k++) {
      if (cj->loc[k] - ci->loc[k] < -e->s->dim[k] * 0.5) {
        shift[k] = e->s->dim[k];
      } else if (cj->loc[k] - ci->loc[k] > e->s->dim[k] * 0.5) {
        shift[k] = -e->s->dim[k];
      }
    }

    /* Get the shift for cell i */
    const double shift_i[3] = {shift[0] + cj->loc[0], shift[1] + cj->loc[1],
                               shift[2] + cj->loc[2]};

    /* Get first and last particles of cell j */
    const int cjs = pack_ind + count_ci;
    const int cje = pack_ind + count_ci + count_cj;

    /* Pack cell i */
    ticks tic_start = getticks();
    if (task_subtype == task_subtype_gpu_density) {
      gpu_pack_part_density(ci, buf->parts_send_d, pack_ind, shift_i, cjs, cje);
    } else if (task_subtype == task_subtype_gpu_gradient) {
      gpu_pack_part_gradient(ci, buf->parts_send_g, pack_ind, shift_i, cjs,
                             cje);
    } else if (task_subtype == task_subtype_gpu_force) {
      gpu_pack_part_force(ci, buf->parts_send_f, pack_ind, shift_i, cjs, cje);
    }
#ifdef SWIFT_DEBUG_CHECKS
    else {
      error("Unknown task subtype %s", subtaskID_names[task_subtype]);
    }
#endif
    ticks tic_end = getticks();

    char subtype2 = '0';
    if (task_subtype == task_subtype_gpu_density) {
      subtype2 = 'd';
    } else if (task_subtype == task_subtype_gpu_gradient) {
      subtype2 = 'g';
    } else if (task_subtype == task_subtype_gpu_force) {
      subtype2 = 'f';
    }
    struct logging_entry ent2 = {subtype2, 'p', ci->hydro.parts - logdata->all_parts, ci->hydro.count, clocks_diff_ticks(tic_end, tic_start) };
    logdata->entries[logdata->count] = ent2;
    logdata->count++;


    /* Update the packed particles counter */
    /* Note: md->count_parts will be increased later */
    pack_ind += count_ci;

    /* Do the same for cj */
    const double shift_j[3] = {cj->loc[0], cj->loc[1], cj->loc[2]};

    tic_start = getticks();
    if (task_subtype == task_subtype_gpu_density) {
      gpu_pack_part_density(cj, buf->parts_send_d, pack_ind, shift_j, cis, cie);
    } else if (task_subtype == task_subtype_gpu_gradient) {
      gpu_pack_part_gradient(cj, buf->parts_send_g, pack_ind, shift_j, cis,
                             cie);
    } else if (task_subtype == task_subtype_gpu_force) {
      gpu_pack_part_force(cj, buf->parts_send_f, pack_ind, shift_j, cis, cie);
    }
#ifdef SWIFT_DEBUG_CHECKS
    else {
      error("Unknown task subtype %s", subtaskID_names[task_subtype]);
    }
#endif
    tic_end = getticks();

    char subtype = '0';
    if (task_subtype == task_subtype_gpu_density) {
      subtype = 'd';
    } else if (task_subtype == task_subtype_gpu_gradient) {
      subtype = 'g';
    } else if (task_subtype == task_subtype_gpu_force) {
      subtype = 'f';
    }
    struct logging_entry ent = {subtype, 'p', cj->hydro.parts - logdata->all_parts, cj->hydro.count, clocks_diff_ticks(tic_end, tic_start) };
    logdata->entries[logdata->count] = ent;
    logdata->count++;
  }

  /* Now finish up the bookkeeping. */

  /* Get the index for the leaf cell */
  const int lid = md->n_leaves_packed;

  /* Identify first particle for each bundle of tasks */
  const int bundle_size = md->params.bundle_size;
  if (lid % bundle_size == 0) {
    int bid = lid / bundle_size;
    /* Store this before we increment md->count_parts */
    md->bundle_first_part[bid] = md->count_parts;
  }

  /* Update incremented pack length accordingly */
  if (ci == cj) {
    /* We packed a self interaction */
    md->count_parts += count_ci;
  } else {
    /* We packed a pair interaction */
    md->count_parts += count_ci + count_cj;
  }

  /* Record that we have now packed a new leaf cell (pair) & increment number
   * of leaf cells to offload */
  md->n_leaves_packed++;
};

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
__attribute__((always_inline)) INLINE static void runner_gpu_unpack(
    const struct runner *r, struct scheduler *s,
    struct gpu_offload_data *restrict buf, const int npacked,
    const enum task_subtypes task_subtype,
    struct logging_data* logdata) {

  /* Grab handles */
  struct gpu_pack_metadata *md = &buf->md;
  const struct engine *e = r->e;

  struct cell **ci_leaves = md->ci_leaves;
  struct cell **cj_leaves = md->cj_leaves;
  int *task_fp = md->task_first_packed_leaf;
  int *task_lp = md->task_last_packed_leaf;

  /* Keep track which tasks in our list we've unpacked already */
  char *task_unpacked = malloc(md->tasks_in_list * sizeof(char));
  for (int i = 0; i < md->tasks_in_list; i++) task_unpacked[i] = 0;
  int ntasks_unpacked = 0;

  while (ntasks_unpacked < md->tasks_in_list) {

    /* Loop over all tasks that we have offloaded */
    for (int tid = 0; tid < md->tasks_in_list; tid++) {

      /* Anything to do here? */
      if (task_unpacked[tid]) continue;

      struct task *t = md->task_list[tid];

      /* Can we get the locks? */
      if (task_lock(t) == 0) continue;

      /* We got it! Mark that. */
      task_unpacked[tid] = 1;
      ntasks_unpacked++;

      /* Get the index in the particle buffer array where to read from */
      int unpack_index = md->task_first_packed_part[tid];

      /* Loop through leaf cell pairs of this task by index */
      for (int lid = task_fp[tid]; lid < task_lp[tid]; lid++) {

        /* Get pointers to the leaf cells */
        struct cell *cii = ci_leaves[lid];
        struct cell *cjj = cj_leaves[lid];

        if (!cell_is_active_hydro(cii, e) && !cell_is_active_hydro(cjj, e)) {
          /* To be fixed and double-checked later */
          error("In unpack, subtype %s: Inactive cell",
                subtaskID_names[task_subtype]);
          return;
        }

        const int count_ci = cii->hydro.count;
        const int count_cj = cjj->hydro.count;

#ifdef SWIFT_DEBUG_CHECKS
        int last_ind = unpack_index + count_ci;
        if (cii != cjj) last_ind += count_cj;

        if (last_ind >= md->params.part_buffer_size) {
          error(
              "Exceeded part_buffer_size=%d. "
              "Increase Scheduler:gpu_part_buffer_size. "
              "ind=%d, counts=%d %d, is self interaction?=%d",
              md->params.part_buffer_size, unpack_index, count_ci, count_cj,
              cii == cjj);
        }
#endif

        /* Get the particle data into CPU-side buffers. */
        if (cell_is_active_hydro(cii, e)) {
          ticks tic_start = getticks();
          if (task_subtype == task_subtype_gpu_density) {
            gpu_unpack_part_density(cii, buf->parts_recv_d, unpack_index,
                                    count_ci, e);
          } else if (task_subtype == task_subtype_gpu_gradient) {
            gpu_unpack_part_gradient(cii, buf->parts_recv_g, unpack_index,
                                     count_ci, e);
          } else if (task_subtype == task_subtype_gpu_force) {
            gpu_unpack_part_force(cii, buf->parts_recv_f, unpack_index,
                                  count_ci, e);
          }
#ifdef SWIFT_DEBUG_CHECKS
          else {
            error("Unknown task subtype %s", subtaskID_names[task_subtype]);
          }
#endif
          ticks tic_end = getticks();

          char subtype = '0';
          if (task_subtype == task_subtype_gpu_density) {
            subtype = 'd';
          } else if (task_subtype == task_subtype_gpu_gradient) {
            subtype = 'g';
          } else if (task_subtype == task_subtype_gpu_force) {
            subtype = 'f';
          }
          struct logging_entry ent = {subtype, 'u', cii->hydro.parts - logdata->all_parts, cii->hydro.count, clocks_diff_ticks(tic_end, tic_start) };
          logdata->entries[logdata->count] = ent;
          logdata->count++;

          unpack_index += count_ci;
        }

        if (cii != cjj) {
          /* We have a pair interaction. Get the other cell too. */
          if (cell_is_active_hydro(cjj, e)) {
            ticks tic_start = getticks();
            if (task_subtype == task_subtype_gpu_density) {
              gpu_unpack_part_density(cjj, buf->parts_recv_d, unpack_index,
                                      count_cj, e);
            } else if (task_subtype == task_subtype_gpu_gradient) {
              gpu_unpack_part_gradient(cjj, buf->parts_recv_g, unpack_index,
                                       count_cj, e);
            } else if (task_subtype == task_subtype_gpu_force) {
              gpu_unpack_part_force(cjj, buf->parts_recv_f, unpack_index,
                                    count_cj, e);
            }
#ifdef SWIFT_DEBUG_CHECKS
            else {
              error("Unknown task subtype %s", subtaskID_names[task_subtype]);
            }
#endif
            ticks tic_end = getticks();

            char subtype = '0';
            if (task_subtype == task_subtype_gpu_density) {
              subtype = 'd';
            } else if (task_subtype == task_subtype_gpu_gradient) {
              subtype = 'g';
            } else if (task_subtype == task_subtype_gpu_force) {
              subtype = 'f';
            }
            struct logging_entry ent = {subtype, 'u', cjj->hydro.parts - logdata->all_parts, cjj->hydro.count, clocks_diff_ticks(tic_end, tic_start) };
            logdata->entries[logdata->count] = ent;
            logdata->count++;

            unpack_index += count_cj;
          }
        }

      } /* Loop over all leaves of task */

      /* We're done with this task. Release the cells. */
      task_unlock(t);

      /* If we haven't finished packing the currently handled task's leaf cells,
       * we mustn't unlock its dependencies yet. ("Currently handled task" is
       * the one for which the offloading cycle is currently underway in
       * runner_gpu_pack_and_launch) */
      if ((tid == md->tasks_in_list - 1) && (npacked != md->task_n_leaves)) {
        continue;
      }

      /* If we're here, we're completely done with this task. Mark it as
       * completed. */

      /* schedule my dependencies */
      scheduler_enqueue_dependencies(s, t);

      /* Tell the scheduler's bookkeeping that this task is done */
      pthread_mutex_lock(&s->sleep_mutex);
      atomic_dec(&s->waiting);
      pthread_cond_broadcast(&s->sleep_cond);
      pthread_mutex_unlock(&s->sleep_mutex);

      /* Mark the task as done. */
      t->skip = 1;

    } /* Loop over tasks in list */
  } /* While there are unpacked tasks */

  /* clean up after yourself */
  free(task_unpacked);
}

/**
 * @brief Wrapper to pack data for density tasks on the GPU.
 */
__attribute__((always_inline)) INLINE static void runner_gpu_pack_density(
    const struct runner *r, struct gpu_offload_data *restrict buf,
    const struct cell *ci, const struct cell *cj,
    struct logging_data* logdata) {

  TIMER_TIC;

  runner_gpu_pack(r, buf, ci, cj, task_subtype_gpu_density, logdata);

  TIMER_TOC(timer_gpu_pack_d);
}

/**
 * @brief Wrapper to pack data for gradient tasks on the GPU.
 */
__attribute__((always_inline)) INLINE static void runner_gpu_pack_gradient(
    const struct runner *r, struct gpu_offload_data *restrict buf,
    const struct cell *ci, const struct cell *cj,
    struct logging_data* logdata) {

  TIMER_TIC;

  runner_gpu_pack(r, buf, ci, cj, task_subtype_gpu_gradient, logdata);

  TIMER_TOC(timer_gpu_pack_g);
}

/**
 * @brief Wrapper to pack data for force tasks on the GPU.
 */
__attribute__((always_inline)) INLINE static void runner_gpu_pack_force(
    const struct runner *r, struct gpu_offload_data *restrict buf,
    const struct cell *ci, const struct cell *cj,
    struct logging_data* logdata) {

  TIMER_TIC;

  runner_gpu_pack(r, buf, ci, cj, task_subtype_gpu_force, logdata);

  TIMER_TOC(timer_gpu_pack_f);
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
    struct gpu_offload_data *restrict buf, const int npacked,
    struct logging_data* logdata) {

  TIMER_TIC;

  runner_gpu_unpack(r, s, buf, npacked, task_subtype_gpu_density, logdata);

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
    struct gpu_offload_data *restrict buf, const int npacked,
    struct logging_data* logdata) {

  TIMER_TIC;

  runner_gpu_unpack(r, s, buf, npacked, task_subtype_gpu_gradient, logdata);

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
                               const int npacked,
                               struct logging_data* logdata) {

  TIMER_TIC;

  runner_gpu_unpack(r, s, buf, npacked, task_subtype_gpu_force, logdata);

  TIMER_TOC(timer_gpu_unpack_f);
}

#endif /* RUNNER_GPU_PACK_FUNCTIONS_H */
