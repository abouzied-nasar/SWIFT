/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2012 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
 *                    Matthieu Schaller (schaller@strw.leidenuniv.nl)
 *               2015 Peter W. Draper (p.w.draper@durham.ac.uk)
 *               2022 Abouzied M. A. Nasar (abouzied.nasar@manchester.ac.uk)
 *               2025 Mladen Ivkovic (mladen.ivkovic@durham.ac.uk)
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

#ifdef WITH_CUDA

#ifdef __cplusplus
extern "C" {
#endif

/* Config parameters. */
#include "../config.h"

/* MPI headers. */
#ifdef WITH_MPI
#include <mpi.h>
#endif

/* Cuda headers */
#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#endif

/* This object's header. */
#include "runner.h"

/* Local headers. */
#include "cell.h" // TODO(mivkov): Check if necessary after refactor
#include "cuda/GPU_offload_data.h"
#include "cuda/GPU_part_structs.h"
#include "cuda/GPU_runner_functions.h"
#include "cuda/GPU_utils.h"
#include "cuda/cuda_config.h"
#include "engine.h"
#include "feedback.h"
#include "runner_doiact_functions_hydro_gpu.h"
#include "runner_gpu_pack_functions.h"
#include "scheduler.h"
#include "space_getsid.h"
#include "timers.h"


/* Import the gravity loop functions. */
#include "runner_doiact_grav.h"

/* Import the density loop functions. */
#define FUNCTION density
#define FUNCTION_TASK_LOOP TASK_LOOP_DENSITY
#include "runner_doiact_hydro.h"
#include "runner_doiact_undef.h"

/* Import the gradient loop functions (if required). */
#ifdef EXTRA_HYDRO_LOOP
#define FUNCTION gradient
#define FUNCTION_TASK_LOOP TASK_LOOP_GRADIENT
#include "runner_doiact_hydro.h"
#include "runner_doiact_undef.h"
#endif

/* Import the force loop functions. */
#define FUNCTION force
#define FUNCTION_TASK_LOOP TASK_LOOP_FORCE
#include "runner_doiact_hydro.h"
#include "runner_doiact_undef.h"

/* Import the limiter loop functions. */
#define FUNCTION limiter
#define FUNCTION_TASK_LOOP TASK_LOOP_LIMITER
#include "runner_doiact_limiter.h"
#include "runner_doiact_undef.h"

/* Import the stars density loop functions. */
#define FUNCTION density
#define FUNCTION_TASK_LOOP TASK_LOOP_DENSITY
#include "runner_doiact_stars.h"
#include "runner_doiact_undef.h"

#ifdef EXTRA_STAR_LOOPS

/* Import the stars prepare1 loop functions. */
#define FUNCTION prep1
#define FUNCTION_TASK_LOOP TASK_LOOP_STARS_PREP1
#include "runner_doiact_stars.h"
#include "runner_doiact_undef.h"

/* Import the stars prepare2 loop functions. */
#define FUNCTION prep2
#define FUNCTION_TASK_LOOP TASK_LOOP_STARS_PREP2
#include "runner_doiact_stars.h"
#include "runner_doiact_undef.h"

#endif /* EXTRA_STAR_LOOPS */

/* Import the stars feedback loop functions. */
#define FUNCTION feedback
#define FUNCTION_TASK_LOOP TASK_LOOP_FEEDBACK
#include "runner_doiact_stars.h"
#include "runner_doiact_undef.h"

/* Import the black hole density loop functions. */
#define FUNCTION density
#define FUNCTION_TASK_LOOP TASK_LOOP_DENSITY
#include "runner_doiact_black_holes.h"
#include "runner_doiact_undef.h"

/* Import the black hole feedback loop functions. */
#define FUNCTION swallow
#define FUNCTION_TASK_LOOP TASK_LOOP_SWALLOW
#include "runner_doiact_black_holes.h"
#include "runner_doiact_undef.h"

#define FUNCTION feedback
#define FUNCTION_TASK_LOOP TASK_LOOP_FEEDBACK
#include "runner_doiact_black_holes.h"
#include "runner_doiact_undef.h"

/* Import the sink density loop functions. */
#define FUNCTION density
#define FUNCTION_TASK_LOOP TASK_LOOP_DENSITY
#include "runner_doiact_sinks.h"
#include "runner_doiact_undef.h"

/* Import the sink swallow loop functions. */
#define FUNCTION swallow
#define FUNCTION_TASK_LOOP TASK_LOOP_SWALLOW
#include "runner_doiact_sinks.h"
#include "runner_doiact_undef.h"

/* Import the RT gradient loop functions */
#define FUNCTION rt_gradient
#define FUNCTION_TASK_LOOP TASK_LOOP_RT_GRADIENT
#include "runner_doiact_hydro.h"
#include "runner_doiact_undef.h"

/* Import the RT transport (force) loop functions. */
#define FUNCTION rt_transport
#define FUNCTION_TASK_LOOP TASK_LOOP_RT_TRANSPORT
#include "runner_doiact_hydro.h"
#include "runner_doiact_undef.h"

/**
 * @brief The #runner main thread routine.
 *
 * @param data A pointer to this thread's data.
 **/
void *runner_main_cuda(void *data) {
  struct runner *r = (struct runner *)data;
  struct engine *e = r->e;
  struct scheduler *sched = &e->sched;

  /* Initialise cuda context for this thread. */
  gpu_init_thread(e, r->cpuid);

  /* Get estimates for array sizes et al. */
  struct gpu_global_pack_params gpu_pack_params;
  gpu_get_pack_params(&gpu_pack_params, sched, e->s->eta_neighbours);

  /* Declare and allocate GPU launch control data structures which need to be in scope */
  struct gpu_offload_data gpu_buf_self_dens;
  struct gpu_offload_data gpu_buf_self_grad;
  struct gpu_offload_data gpu_buf_self_forc;
  struct gpu_offload_data gpu_buf_pair_dens;
  struct gpu_offload_data gpu_buf_pair_grad;
  struct gpu_offload_data gpu_buf_pair_forc;

  gpu_init_data_buffers(&gpu_buf_self_dens, &gpu_pack_params,
      sizeof(struct part_aos_f4_send_d), sizeof(struct part_aos_f4_recv_d), /*is_pair_task=*/0);
  gpu_init_data_buffers(&gpu_buf_self_grad, &gpu_pack_params,
      sizeof(struct part_aos_f4_send_g), sizeof(struct part_aos_f4_recv_g), /*is_pair_task=*/0);
  gpu_init_data_buffers(&gpu_buf_self_forc, &gpu_pack_params,
      sizeof(struct part_aos_f4_send_f), sizeof(struct part_aos_f4_recv_f), /*is_pair_task=*/0);
  gpu_init_data_buffers(&gpu_buf_pair_dens, &gpu_pack_params,
      sizeof(struct part_aos_f4_send_d), sizeof(struct part_aos_f4_recv_d), /*is_pair_task=*/1);
  gpu_init_data_buffers(&gpu_buf_pair_grad, &gpu_pack_params,
      sizeof(struct part_aos_f4_send_g), sizeof(struct part_aos_f4_recv_g), /*is_pair_task=*/1);
  gpu_init_data_buffers(&gpu_buf_pair_forc, &gpu_pack_params,
      sizeof(struct part_aos_f4_send_f), sizeof(struct part_aos_f4_recv_f), /*is_pair_task=*/1);


  /* TODO: MOVE TO CUDA_INIT_STREAMS ? */
  cudaStream_t stream[gpu_pack_params.n_bundles];
  cudaStream_t stream_pairs[gpu_pack_params.n_bundles_pair];

  for (size_t i = 0; i < gpu_pack_params.n_bundles; ++i)
    cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
  for (size_t i = 0; i < gpu_pack_params.n_bundles_pair; ++i)
    cudaStreamCreateWithFlags(&stream_pairs[i], cudaStreamNonBlocking);

  /* Declare some global variables */
  int step = 0;

  /* Tell me how much memory we're using. */
  gpu_print_free_mem(e, r->cpuid);

  /* Main loop. */
  while (1) {
    /* Wait at the barrier. */
    engine_barrier(e);

    /* Can we go home yet? */
    if (e->step_props & engine_step_prop_done) break;

    gpu_init_data_buffers_step(&gpu_buf_self_dens);
    gpu_init_data_buffers_step(&gpu_buf_self_grad);
    gpu_init_data_buffers_step(&gpu_buf_self_forc);
    gpu_init_data_buffers_step(&gpu_buf_pair_dens);
    gpu_init_data_buffers_step(&gpu_buf_pair_grad);
    gpu_init_data_buffers_step(&gpu_buf_pair_forc);

    /* Get some global variables' values for this step */
    const float d_a = e->cosmology->a;
    const float d_H = e->cosmology->H;

    /* Re-set the pointer to the previous task, as there is none. */
    struct task *t = NULL;
    struct task *prev = NULL;
    /*Some bits for output in case of debug*/

    if (step == 0) cudaProfilerStart();
    step++;

    /* Loop while there are tasks... */
    while (1) {
      // A. Nasar: Get qid for re-use later
      int qid = r->qid;
      /* If there's no old task, try to get a new one. */
      if (t == NULL) {
        /* Get the task. */
        TIMER_TIC
        t = scheduler_gettask(sched, qid, prev);
        TIMER_TOC(timer_gettask);
        /* Did I get anything? */
        if (t == NULL) break;
      }

/* TODO MLADEN: REMOVE */
/* message("RUNNING TASK %s/%s", taskID_names[t->type], subtaskID_names[t->subtype]); */
/* fflush(stdout); */

      /* Get the cells. */
      struct cell *ci = t->ci;
      struct cell *cj = t->cj;

      if (ci == NULL &&
          (t->subtype != task_subtype_gpu_unpack_d &&
           t->subtype != task_subtype_gpu_unpack_g &&
           t->subtype != task_subtype_gpu_unpack_f))
        error("This cannot be");

#ifdef SWIFT_DEBUG_TASKS
      /* Mark the thread we run on */
      t->rid = r->cpuid;

      /* And recover the pair direction */
      if (t->type == task_type_pair) {
        struct cell *ci_temp = ci;
        struct cell *cj_temp = cj;
        double shift[3];
        if (t->subtype != task_subtype_gpu_unpack_d &&
            t->subtype != task_subtype_gpu_unpack_g &&
            t->subtype != task_subtype_gpu_unpack_f)
          t->sid = space_getsid_and_swap_cells(e->s, &ci_temp, &cj_temp, shift);
      } else {
        t->sid = -1;
      }
#endif

#ifdef SWIFT_DEBUG_CHECKS
      /* Check that we haven't scheduled an inactive task */
      t->ti_run = e->ti_current;
      /* Store the task that will be running (for debugging only) */
      r->t = t;
#endif

      const ticks task_beg = getticks();
      /* Different types of tasks... */
      switch (t->type) {

        case task_type_self:
          if (t->subtype == task_subtype_grav)
            runner_doself_recursive_grav(r, ci, 1);
          else if (t->subtype == task_subtype_external_grav)
            runner_do_grav_external(r, ci, 1);
          else if (t->subtype == task_subtype_gpu_unpack_d) {
          }
          else if (t->subtype == task_subtype_gpu_unpack_g) {
          }
          else if (t->subtype == task_subtype_gpu_unpack_f) {
          }
          else if (t->subtype == task_subtype_density) {
#ifndef GPUOFFLOAD_DENSITY
            runner_dosub_self1_density(r, ci, /*below_h_max=*/0, 1);
#endif
          } else if (t->subtype == task_subtype_gpu_pack_d) {
#ifdef GPUOFFLOAD_DENSITY
            /* GPU Work */
            runner_doself_gpu_pack_density(r, sched, &gpu_buf_self_dens, ci, t);
            /* No pack tasks left in queue, flag that we want to run */
            char launch_leftovers = gpu_buf_self_dens.pv.launch_leftovers;
            /* Packed enough tasks. Let's go*/
            char launch = gpu_buf_self_dens.pv.launch;
            /* Do we have enough stuff to run the GPU ? */
            if (launch || launch_leftovers) {
              runner_doself_gpu_density(r, sched, &gpu_buf_self_dens, t, stream, d_a, d_H);
            }
#endif
          } /* self / pack */
          else if (t->subtype == task_subtype_gpu_pack_g) {
#ifdef GPUOFFLOAD_GRADIENT
            runner_doself_gpu_pack_gradient(r, sched, &gpu_buf_self_grad, ci, t);
            /* No pack tasks left in queue, flag that we want to run */
            char launch_leftovers = gpu_buf_self_grad.pv.launch_leftovers;
            /*Packed enough tasks let's go*/
            char launch = gpu_buf_self_grad.pv.launch;
            /* Do we have enough stuff to run the GPU ? */
            if (launch || launch_leftovers) {
              runner_doself_gpu_gradient(r, sched, &gpu_buf_self_grad, t, stream, d_a, d_H);
            }
#endif  // GPUGRADSELF
          } else if (t->subtype == task_subtype_gpu_pack_f) {
#ifdef GPUOFFLOAD_FORCE
            runner_doself_gpu_pack_force(r, sched, &gpu_buf_self_forc, ci, t);
            /* No pack tasks left in queue, flag that we want to run */
            char launch_leftovers = gpu_buf_self_forc.pv.launch_leftovers;
            /*Packed enough tasks let's go*/
            char launch = gpu_buf_self_forc.pv.launch;
            /* Do we have enough stuff to run the GPU ? */
            if (launch || launch_leftovers) {
              /*Launch GPU tasks*/
              runner_doself_gpu_force(r, sched, &gpu_buf_self_forc, t, stream, d_a, d_H);
            }
#endif
          }
#ifdef EXTRA_HYDRO_LOOP
          else if (t->subtype == task_subtype_gradient) {
#ifndef GPUOFFLOAD_GRADIENT
#ifdef EXTRA_HYDRO_LOOP_TYPE2
            runner_dosub_self2_gradient(r, ci, /*below_h_max=*/0, 1);
#else
            runner_dosub_self1_gradient(r, ci, /*below_h_max=*/0, 1);
#endif
#endif
          }
#endif
          else if (t->subtype == task_subtype_force) {
#ifndef GPUOFFLOAD_FORCE
            runner_dosub_self2_force(r, ci, /*below_h_max=*/0, 1);
#endif
          } else if (t->subtype == task_subtype_limiter)
            runner_dosub_self1_limiter(r, ci, /*below_h_max=*/0, 1);
          else if (t->subtype == task_subtype_stars_density)
            runner_dosub_self_stars_density(r, ci, /*below_h_max=*/0, 1);
#ifdef EXTRA_STAR_LOOPS
          else if (t->subtype == task_subtype_stars_prep1)
            runner_dosub_self_stars_prep1(r, ci, /*below_h_max=*/0, 1);
          else if (t->subtype == task_subtype_stars_prep2)
            runner_dosub_self_stars_prep2(r, ci, /*below_h_max=*/0, 1);
#endif
          else if (t->subtype == task_subtype_stars_feedback)
            runner_dosub_self_stars_feedback(r, ci, /*below_h_max=*/0, 1);
          else if (t->subtype == task_subtype_bh_density)
            runner_dosub_self_bh_density(r, ci, 1);
          else if (t->subtype == task_subtype_bh_swallow)
            runner_dosub_self_bh_swallow(r, ci, 1);
          else if (t->subtype == task_subtype_do_gas_swallow)
            runner_do_gas_swallow_self(r, ci, 1);
          else if (t->subtype == task_subtype_do_bh_swallow)
            runner_do_bh_swallow_self(r, ci, 1);
          else if (t->subtype == task_subtype_bh_feedback)
            runner_dosub_self_bh_feedback(r, ci, 1);
          else if (t->subtype == task_subtype_rt_gradient)
            runner_dosub_self1_rt_gradient(r, ci, /*below_h_max=*/0, 1);
          else if (t->subtype == task_subtype_rt_transport)
            runner_dosub_self2_rt_transport(r, ci, /*below_h_max=*/0, 1);
          else if (t->subtype == task_subtype_sink_density)
            runner_dosub_self_sinks_density(r, ci, 1);
          else if (t->subtype == task_subtype_sink_swallow)
            runner_dosub_self_sinks_swallow(r, ci, 1);
          else if (t->subtype == task_subtype_sink_do_gas_swallow)
            runner_do_sinks_gas_swallow_self(r, ci, 1);
          else if (t->subtype == task_subtype_sink_do_sink_swallow)
            runner_do_sinks_sink_swallow_self(r, ci, 1);
          else
            error("Unknown/invalid task subtype (%s/%s).",
                  taskID_names[t->type], subtaskID_names[t->subtype]);
          break;

        case task_type_pair:
          if (t->subtype == task_subtype_grav)
            runner_dopair_recursive_grav(r, ci, cj, 1);
          else if (t->subtype == task_subtype_density) {
#ifndef GPUOFFLOAD_DENSITY
            runner_dosub_pair1_density(r, ci, cj, /*below_h_max=*/0, 1);
#endif
          }
          /* GPU WORK */
          else if (t->subtype == task_subtype_gpu_pack_d) {
#ifdef GPUOFFLOAD_DENSITY
#ifndef RECURSE
            ticks tic_cpu_pack = getticks();
            /*Pack data and increment counters checking if we should run on the GPU after packing this task*/
            packing_time_pair +=
                runner_dopair1_pack_f4_d(r, sched, pack_vars_pair_dens, ci,
                                         cj, t, parts_aos_pair_f4_send, e,
                                         fparti_fpartj_lparti_lpartj_dens);
            /* No pack tasks left in queue, flag that we want to run */
            int launch_leftovers = pack_vars_pair_dens->launch_leftovers;
            /*Packed enough tasks let's go*/
            int launch = pack_vars_pair_dens->launch;
            /* Do we have enough stuff to run the GPU ? */
            if (launch || launch_leftovers) {
              /*Launch GPU tasks*/
              runner_dopair1_launch_f4_one_memcpy(
                  r, sched, pack_vars_pair_dens, t, parts_aos_pair_f4_send,
                  parts_aos_pair_f4_recv, d_parts_aos_pair_f4_send,
                  d_parts_aos_pair_f4_recv, stream_pairs, d_a, d_H, e,
                  &packing_time_pair, &time_for_gpu_pair,
                  &unpacking_time_pair, fparti_fpartj_lparti_lpartj_dens,
                  pair_end);
              pack_vars_pair_dens->launch_leftovers = 0;
            } /* End of GPU work Pairs */

#else //RECURSE
            runner_dopair_gpu_recurse(r, sched, &gpu_buf_pair_dens, ci, cj, t, /*depth=*/0, /*timer=*/1);
            runner_gpu_pack_daughters_and_launch_d(r, sched, ci, cj, &gpu_buf_pair_dens, t, stream_pairs, d_a, d_H);

#endif  //RECURSE
#endif  // GPUOFFLOAD_DENSITY
          } /* pair / pack */
          else if (t->subtype == task_subtype_gpu_pack_g) {
#ifdef GPUOFFLOAD_GRADIENT
#ifndef RECURSE
              ticks tic_cpu_pack = getticks();
              packing_time_pair_g +=
                  runner_dopair1_pack_f4_g(r, sched, pack_vars_pair_grad, ci,
                                           cj, t, parts_aos_pair_f4_g_send, e,
                                           fparti_fpartj_lparti_lpartj_grad);
              t->total_cpu_pack_ticks += getticks() - tic_cpu_pack;
              /* No pack tasks left in queue, flag that we want to run */
              int launch_leftovers = pack_vars_pair_grad->launch_leftovers;
              /*Packed enough tasks, let's go*/
              int launch = pack_vars_pair_grad->launch;
              /* Do we have enough stuff to run the GPU ? */
              if (launch || launch_leftovers) {
                /*Launch GPU tasks*/
                int t_packed = pack_vars_pair_grad->tasks_packed;
                //                signal_sleeping_runners(sched, t, t_packed);
                runner_dopair1_launch_f4_g_one_memcpy(
                    r, sched, pack_vars_pair_grad, t, parts_aos_pair_f4_g_send,
                    parts_aos_pair_f4_g_recv, d_parts_aos_pair_f4_g_send,
                    d_parts_aos_pair_f4_g_recv, stream_pairs, d_a, d_H, e,
                    &packing_time_pair_g, &time_for_gpu_pair_g,
                    &unpacking_time_pair_g, fparti_fpartj_lparti_lpartj_grad,
                    pair_end_g);
              }
              pack_vars_pair_grad->launch_leftovers = 0;
#else
              runner_dopair_gpu_recurse(r, sched, &gpu_buf_pair_grad, ci, cj, t, /*depth=*/0, /*timer=*/1);

              runner_gpu_pack_daughters_and_launch_g(r, sched, ci, cj,
                  &gpu_buf_pair_grad.pv, t,
                  gpu_buf_pair_grad.parts_send_g,
                  gpu_buf_pair_grad.parts_recv_g,
                  gpu_buf_pair_grad.d_parts_send_g,
                  gpu_buf_pair_grad.d_parts_recv_g,
                    stream_pairs,
                    d_a, d_H,
                    e,
                    gpu_buf_pair_grad.fparti_fpartj_lparti_lpartj,
                    gpu_buf_pair_grad.event_end,
                    gpu_buf_pair_grad.pv.n_leaves_found,
                    gpu_buf_pair_grad.ci_d,
                    gpu_buf_pair_grad.cj_d,
                    gpu_buf_pair_grad.first_and_last_daughters,
                    gpu_buf_pair_grad.ci_top,
                    gpu_buf_pair_grad.cj_top);
#endif
#endif  // GPUOFFLOAD_GRADIENT
          } else if (t->subtype == task_subtype_gpu_pack_f) {
#ifdef GPUOFFLOAD_FORCE
#ifndef RECURSE
              ticks tic_cpu_pack = getticks();
              /*Pack data and increment counters checking if we should run on the GPU after packing this task*/
              packing_time_pair_f +=
                  runner_dopair1_pack_f4_f(r, sched, pack_vars_pair_forc, ci,
                                           cj, t, parts_aos_pair_f4_f_send, e,
                                           fparti_fpartj_lparti_lpartj_forc);
              /* No pack tasks left in queue, flag that we want to run */
              int launch_leftovers = pack_vars_pair_forc->launch_leftovers;
              /*Packed enough tasks let's go*/
              int launch = pack_vars_pair_forc->launch;
              /* Do we have enough stuff to run the GPU ? */
              if (launch || launch_leftovers) {
                /*Launch GPU tasks*/
                int t_packed = pack_vars_pair_forc->tasks_packed;
                //                signal_sleeping_runners(sched, t, t_packed);
                runner_dopair1_launch_f4_f_one_memcpy(
                    r, sched, pack_vars_pair_forc, t, parts_aos_pair_f4_f_send,
                    parts_aos_pair_f4_f_recv, d_parts_aos_pair_f4_f_send,
                    d_parts_aos_pair_f4_f_recv, stream_pairs, d_a, d_H, e,
                    &packing_time_pair_f, &time_for_gpu_pair_f,
                    &unpacking_time_pair_f, fparti_fpartj_lparti_lpartj_forc,
                    pair_end_f);

                pack_vars_pair_forc->launch_leftovers = 0;
              } /* End of GPU work Pairs */
#else
              runner_dopair_gpu_recurse(r, sched, &gpu_buf_pair_forc, ci, cj, t, /*depth=*/0, /*timer=*/1);

              runner_gpu_pack_daughters_and_launch_f(r, sched, ci, cj, &gpu_buf_pair_forc.pv, t,
                    gpu_buf_pair_forc.parts_send_f, gpu_buf_pair_forc.parts_recv_f,
                    gpu_buf_pair_forc.d_parts_send_f, gpu_buf_pair_forc.d_parts_recv_f,
                    stream_pairs, d_a, d_H, e, gpu_buf_pair_forc.fparti_fpartj_lparti_lpartj,
                    gpu_buf_pair_forc.event_end, gpu_buf_pair_forc.pv.n_leaves_found, gpu_buf_pair_forc.ci_d, gpu_buf_pair_forc.cj_d,
                    gpu_buf_pair_forc.first_and_last_daughters, gpu_buf_pair_forc.ci_top, gpu_buf_pair_forc.cj_top);
#endif
#endif  // GPUOFFLOAD_FORCE
          } else if (t->subtype == task_subtype_gpu_unpack_d) {
          } else if (t->subtype == task_subtype_gpu_unpack_g) {
          } else if (t->subtype == task_subtype_gpu_unpack_f) {
          }

#ifdef EXTRA_HYDRO_LOOP
          else if (t->subtype == task_subtype_gradient) {
#ifndef GPUOFFLOAD_GRADIENT
#ifdef EXTRA_HYDRO_LOOP_TYPE2
            runner_dosub_pair2_gradient(r, ci, cj, /*below_h_max=*/0, 1);
#else
            runner_dosub_pair1_gradient(r, ci, cj, /*below_h_max=*/0, 1);
#endif
#endif
          }
#endif  // EXTRA_HYDRO_LOOP
          else if (t->subtype == task_subtype_force) {
#ifndef GPUOFFLOAD_FORCE
            runner_dosub_pair2_force(r, ci, cj, /*below_h_max=*/0, 1);
#endif  // GPUOFFLOAD_FORCE
          } else if (t->subtype == task_subtype_limiter)
            runner_dosub_pair1_limiter(r, ci, cj, /*below_h_max=*/0, 1);
          else if (t->subtype == task_subtype_stars_density)
            runner_dosub_pair_stars_density(r, ci, cj, /*below_h_max=*/0, 1);
#ifdef EXTRA_STAR_LOOPS
          else if (t->subtype == task_subtype_stars_prep1)
            runner_dosub_pair_stars_prep1(r, ci, cj, /*below_h_max=*/0, 1);
          else if (t->subtype == task_subtype_stars_prep2)
            runner_dosub_pair_stars_prep2(r, ci, cj, /*below_h_max=*/0, 1);
#endif
          else if (t->subtype == task_subtype_stars_feedback)
            runner_dosub_pair_stars_feedback(r, ci, cj, /*below_h_max=*/0, 1);
          else if (t->subtype == task_subtype_bh_density)
            runner_dosub_pair_bh_density(r, ci, cj, 1);
          else if (t->subtype == task_subtype_bh_swallow)
            runner_dosub_pair_bh_swallow(r, ci, cj, 1);
          else if (t->subtype == task_subtype_do_gas_swallow)
            runner_do_gas_swallow_pair(r, ci, cj, 1);
          else if (t->subtype == task_subtype_do_bh_swallow)
            runner_do_bh_swallow_pair(r, ci, cj, 1);
          else if (t->subtype == task_subtype_bh_feedback)
            runner_dosub_pair_bh_feedback(r, ci, cj, 1);
          else if (t->subtype == task_subtype_rt_gradient)
            runner_dosub_pair1_rt_gradient(r, ci, cj, /*below_h_max=*/0, 1);
          else if (t->subtype == task_subtype_rt_transport)
            runner_dosub_pair2_rt_transport(r, ci, cj, /*below_h_max=*/0, 1);
          else if (t->subtype == task_subtype_sink_density)
            runner_dosub_pair_sinks_density(r, ci, cj, 1);
          else if (t->subtype == task_subtype_sink_swallow)
            runner_dosub_pair_sinks_swallow(r, ci, cj, 1);
          else if (t->subtype == task_subtype_sink_do_gas_swallow)
            runner_do_sinks_gas_swallow_pair(r, ci, cj, 1);
          else if (t->subtype == task_subtype_sink_do_sink_swallow)
            runner_do_sinks_sink_swallow_pair(r, ci, cj, 1);
          else
            error("Unknown/invalid task subtype (%s/%s).",
                  taskID_names[t->type], subtaskID_names[t->subtype]);
          break;

        case task_type_sort:
          /* Cleanup only if any of the indices went stale. */
          runner_do_hydro_sort(
              r, ci, t->flags,
              ci->hydro.dx_max_sort_old > space_maxreldx * ci->dmin,
              /*lock=*/0, cell_get_flag(ci, cell_flag_rt_requests_sort),
              /*clock=*/1);
          /* Reset the sort flags as our work here is done. */
          t->flags = 0;
          break;
        case task_type_rt_sort:
          /* Cleanup only if any of the indices went stale.
           * NOTE: we check whether we reset the sort flags when the
           * recv tasks are running. Cells without an RT recv task
           * don't have rt_sort tasks. */
          runner_do_hydro_sort(
              r, ci, t->flags,
              ci->hydro.dx_max_sort_old > space_maxreldx * ci->dmin,
              /*lock=*/0, /*rt_requests_sorts=*/1, /*clock=*/1);
          /* Reset the sort flags as our work here is done. */
          t->flags = 0;
          break;
        case task_type_stars_sort:
          /* Cleanup only if any of the indices went stale. */
          runner_do_stars_sort(
              r, ci, t->flags,
              ci->stars.dx_max_sort_old > space_maxreldx * ci->dmin, 1);
          /* Reset the sort flags as our work here is done. */
          t->flags = 0;
          break;
        case task_type_init_grav:
          runner_do_init_grav(r, ci, 1);
          break;
        case task_type_ghost:
          runner_do_ghost(r, ci, 1);
          break;
#ifdef EXTRA_HYDRO_LOOP
        case task_type_extra_ghost:
          runner_do_extra_ghost(r, ci, 1);
          break;
#endif
        case task_type_stars_ghost:
          runner_do_stars_ghost(r, ci, 1);
          break;
        case task_type_bh_density_ghost:
          runner_do_black_holes_density_ghost(r, ci, 1);
          break;
        case task_type_bh_swallow_ghost3:
          runner_do_black_holes_swallow_ghost(r, ci, 1);
          break;
        case task_type_sink_density_ghost:
          runner_do_sinks_density_ghost(r, ci, 1);
          break;
        case task_type_drift_part:
          runner_do_drift_part(r, ci, 1);
          break;
        case task_type_drift_spart:
          runner_do_drift_spart(r, ci, 1);
          break;
        case task_type_drift_sink:
          runner_do_drift_sink(r, ci, 1);
          break;
        case task_type_drift_bpart:
          runner_do_drift_bpart(r, ci, 1);
          break;
        case task_type_drift_gpart:
          runner_do_drift_gpart(r, ci, 1);
          break;
        case task_type_kick1:
          runner_do_kick1(r, ci, 1);
          break;
        case task_type_kick2:
          runner_do_kick2(r, ci, 1);
          break;
        case task_type_end_hydro_force:
          runner_do_end_hydro_force(r, ci, 1);
          break;
        case task_type_end_grav_force:
          runner_do_end_grav_force(r, ci, 1);
          break;
        case task_type_csds:
          runner_do_csds(r, ci, 1);
          break;
        case task_type_timestep:
          runner_do_timestep(r, ci, 1);
          break;
        case task_type_timestep_limiter:
          runner_do_limiter(r, ci, 0, 1);
          break;
        case task_type_timestep_sync:
          runner_do_sync(r, ci, 0, 1);
          break;
        case task_type_collect:
          runner_do_timestep_collect(r, ci, 1);
          break;
        case task_type_rt_collect_times:
          runner_do_collect_rt_times(r, ci, 1);
          break;
#ifdef WITH_MPI
        case task_type_send:
          if (t->subtype == task_subtype_tend) {
            free(t->buff);
          } else if (t->subtype == task_subtype_sf_counts) {
            free(t->buff);
          } else if (t->subtype == task_subtype_grav_counts) {
            free(t->buff);
          } else if (t->subtype == task_subtype_part_swallow) {
            free(t->buff);
          } else if (t->subtype == task_subtype_bpart_merger) {
            free(t->buff);
          } else if (t->subtype == task_subtype_limiter) {
            free(t->buff);
          }
          break;
        case task_type_recv:
          if (t->subtype == task_subtype_tend) {
            cell_unpack_end_step(ci, (struct pcell_step *)t->buff);
            free(t->buff);
          } else if (t->subtype == task_subtype_sf_counts) {
            cell_unpack_sf_counts(ci, (struct pcell_sf_stars *)t->buff);
            cell_clear_stars_sort_flags(ci, /*clear_unused_flags=*/0);
            free(t->buff);
          } else if (t->subtype == task_subtype_grav_counts) {
            cell_unpack_grav_counts(ci, (struct pcell_sf_grav *)t->buff);
            free(t->buff);
          } else if (t->subtype == task_subtype_xv) {
            runner_do_recv_part(r, ci, 1, 1);
          } else if (t->subtype == task_subtype_rho) {
            runner_do_recv_part(r, ci, 0, 1);
          } else if (t->subtype == task_subtype_gradient) {
            runner_do_recv_part(r, ci, 0, 1);
          } else if (t->subtype == task_subtype_rt_gradient) {
            runner_do_recv_part(r, ci, 2, 1);
          } else if (t->subtype == task_subtype_rt_transport) {
            runner_do_recv_part(r, ci, -1, 1);
          } else if (t->subtype == task_subtype_part_swallow) {
            cell_unpack_part_swallow(ci,
                                     (struct black_holes_part_data *)t->buff);
            free(t->buff);
          } else if (t->subtype == task_subtype_bpart_merger) {
            cell_unpack_bpart_swallow(ci,
                                      (struct black_holes_bpart_data *)t->buff);
            free(t->buff);
          } else if (t->subtype == task_subtype_limiter) {
            /* Nothing to do here. Unpacking done in a separate task */
          } else if (t->subtype == task_subtype_gpart) {
            runner_do_recv_gpart(r, ci, 1);
          } else if (t->subtype == task_subtype_spart_density) {
            runner_do_recv_spart(r, ci, 1, 1);
          } else if (t->subtype == task_subtype_part_prep1) {
            runner_do_recv_part(r, ci, 0, 1);
          } else if (t->subtype == task_subtype_spart_prep2) {
            runner_do_recv_spart(r, ci, 0, 1);
          } else if (t->subtype == task_subtype_bpart_rho) {
            runner_do_recv_bpart(r, ci, 1, 1);
          } else if (t->subtype == task_subtype_bpart_feedback) {
            runner_do_recv_bpart(r, ci, 0, 1);
          } else {
            error("Unknown/invalid task subtype (%d).", t->subtype);
          }
          break;

        case task_type_pack:
          runner_do_pack_limiter(r, ci, &t->buff, 1);
          task_get_unique_dependent(t)->buff = t->buff;
          break;
        case task_type_unpack:
          runner_do_unpack_limiter(r, ci, t->buff, 1);
          break;
#endif
        case task_type_grav_down:
          runner_do_grav_down(r, t->ci, 1);
          break;
        case task_type_grav_long_range:
          runner_do_grav_long_range(r, t->ci, 1);
          break;
        case task_type_grav_mm:
          runner_dopair_grav_mm_progenies(r, t->flags, t->ci, t->cj);
          break;
        case task_type_cooling:
          runner_do_cooling(r, t->ci, 1);
          break;
        case task_type_star_formation:
          runner_do_star_formation(r, t->ci, 1);
          break;
        case task_type_star_formation_sink:
          runner_do_star_formation_sink(r, t->ci, 1);
          break;
        case task_type_stars_resort:
          runner_do_stars_resort(r, t->ci, 1);
          break;
        case task_type_sink_formation:
          runner_do_sink_formation(r, t->ci);
          break;
        case task_type_fof_self:
          runner_do_fof_search_self(r, t->ci, 1);
          break;
        case task_type_fof_pair:
          runner_do_fof_search_pair(r, t->ci, t->cj, 1);
          break;
        case task_type_fof_attach_self:
          runner_do_fof_attach_self(r, t->ci, 1);
          break;
        case task_type_fof_attach_pair:
          runner_do_fof_attach_pair(r, t->ci, t->cj, 1);
          break;
        case task_type_neutrino_weight:
          runner_do_neutrino_weighting(r, ci, 1);
          break;
        case task_type_rt_ghost1:
          runner_do_rt_ghost1(r, t->ci, 1);
          break;
        case task_type_rt_ghost2:
          runner_do_rt_ghost2(r, t->ci, 1);
          break;
        case task_type_rt_tchem:
          runner_do_rt_tchem(r, t->ci, 1);
          break;
        case task_type_rt_advance_cell_time:
          runner_do_rt_advance_cell_time(r, t->ci, 1);
          break;
        default:
          error("Unknown/invalid task type (%d).", t->type);
      }
      r->active_time += (getticks() - task_beg);

/* Mark that we have run this task on these cells */
#ifdef SWIFT_DEBUG_CHECKS
      if (ci != NULL) {
        ci->tasks_executed[t->type]++;
        ci->subtasks_executed[t->subtype]++;
      }
      if (cj != NULL) {
        cj->tasks_executed[t->type]++;
        cj->subtasks_executed[t->subtype]++;
      }
      /* This runner is not doing a task anymore */
      r->t = NULL;
#endif
      /* We're done with this task, see if we get a next one. */
      prev = t;
      if (t->subtype == task_subtype_gpu_pack_d) {
#ifdef GPUOFFLOAD_DENSITY
        /* Don't enqueue unpacks yet. Just signal the runners */
        t->skip = 1;
        t->toc = getticks();
        t->total_ticks += t->toc - t->tic;
        t = NULL;
#else
        t = scheduler_done(sched, t);
#endif
      }

      else if (t->subtype == task_subtype_gpu_pack_g) {
#ifdef GPUOFFLOAD_GRADIENT
        /* Don't enqueue unpacks yet. Just signal the runners */
        t->skip = 1;
        t->toc = getticks();
        t->total_ticks += t->toc - t->tic;
        t = NULL;
#else
        t = scheduler_done(sched, t);
#endif
      }

      else if (t->subtype == task_subtype_gpu_pack_f) {
#ifdef GPUOFFLOAD_FORCE
        /* Don't enqueue unpacks yet. Just signal the runners */
        t->skip = 1;
        t->toc = getticks();
        t->total_ticks += t->toc - t->tic;
        t = NULL;
#else
        t = scheduler_done(sched, t);
#endif
      }

      else if (t->subtype != task_subtype_gpu_pack_d &&
               t->subtype != task_subtype_gpu_pack_g &&
               t->subtype != task_subtype_gpu_pack_f) {
        t = scheduler_done(sched, t);
      }
    } /* main loop. */
  }

  /* TODO: clear/free alloc'd stuff here. */

  /* Be kind, rewind. */
  return NULL;
}

#ifdef __cplusplus
}
#endif

#endif  // WITH_CUDA

