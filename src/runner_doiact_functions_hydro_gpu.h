#ifdef __cplusplus
extern "C" {
#endif

/* #include "atomic.h" */
#include "active.h"
#include "error.h"
/* #include "GPU_pack_vars.h" */
#include "runner.h"
#include "runner_doiact_hydro.h"
#include "runner_gpu_pack_functions.h"
#include "scheduler.h"
#include "space_getsid.h"
#include "task.h"
#include "timers.h"


#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda/GPU_offload_data.h"
#include "cuda/GPU_runner_functions.h"
#include "cuda/cuda_config.h"
#else
#endif

#ifdef WITH_HIP
#include "hip/hip_config.h"
#include "hip/GPU_runner_functions.h"
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
void runner_doself_gpu_pack(struct cell *ci, struct task *t, struct gpu_offload_data *buf, enum task_subtypes task_subtype) {

  /* Grab a hold of the packing buffers */
  struct gpu_pack_vars* pv = &(buf->pv);

  /* Place pointers to the task and cells packed in an array for use later
   * when unpacking after the GPU offload */
  size_t tasks_packed = pv->tasks_packed;
  pv->task_list[tasks_packed] = t;
  pv->ci_list[tasks_packed] = ci;

  /* Identify row in particle arrays where this task starts*/
  buf->task_first_part_f4[tasks_packed].x = pv->count_parts;

  /* Anything to do here? */
  int count = ci->hydro.count;
  if (count > 0) {

#ifdef SWIFT_DEBUG_CHECKS
    const size_t local_pack_position = buf->pv.count_parts;
    const size_t count_max_parts_tmp = buf->pv.count_max_parts;
    if (local_pack_position + count >= count_max_parts_tmp) {
      error("Exceeded count_max_parts_tmp. Make arrays bigger! "
          "count_max %lu count %lu",
          count_max_parts_tmp, local_pack_position + count);
    }
#endif

    /* This re-arranges the particle data from cell->hydro->parts into a
       long array of part structs */
    if (task_subtype == task_subtype_gpu_pack_d) {
      gpu_pack_density_self(ci, buf);
    } else if (task_subtype == task_subtype_gpu_pack_g) {
      gpu_pack_gradient_self(ci, buf);
    } else if (task_subtype == task_subtype_gpu_pack_f){
      gpu_pack_force_self(ci, buf);
    } else {
      error("Unknown task subtype %s", subtaskID_names[task_subtype]);
    }

    /* Increment pack length accordingly */
    buf->pv.count_parts += count;
  }

  /* Identify the row in the array where this task ends (row id of its
     last particle)*/
  buf->task_first_part_f4[tasks_packed].y = buf->pv.count_parts;

  /* Identify first particle for each bundle of tasks */
  const size_t bundle_size = buf->pv.bundle_size;
  if (tasks_packed % bundle_size == 0) {
    size_t bid = tasks_packed / bundle_size;
    buf->pv.bundle_first_part[bid] = buf->task_first_part_f4[tasks_packed].x;
    buf->pv.bundle_first_task_list[bid] = tasks_packed;
  }

  /* Tell the cell it has been packed */
  ci->pack_done++;

  /* Record that we have now done a packing (self) */
  t->done = 1;
  buf->pv.tasks_packed++;
  buf->pv.launch = 0;
  buf->pv.launch_leftovers = 0;

  /* Have we packed enough tasks to offload to GPU? */
  if (buf->pv.tasks_packed == buf->pv.target_n_tasks){
    buf->pv.launch = 1;
  }

  /* Release the cell. */
  /* TODO: WHERE IS THIS TREE LOCKED?????????? */
  cell_unlocktree(ci);
}


void runner_doself_gpu_pack_d(struct runner *r, struct scheduler *s, struct
    gpu_offload_data *buf, struct cell *ci, struct task *t) {

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

void runner_doself_gpu_pack_g(struct runner *r, struct scheduler *s, struct
    gpu_offload_data *buf, struct cell *ci, struct task *t) {

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

void runner_doself_gpu_pack_f(struct runner *r, struct scheduler *s, struct
    gpu_offload_data *buf, struct cell *ci, struct task *t) {

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



void runner_recurse_gpu(struct runner *r, struct scheduler *s,
                        struct gpu_pack_vars *restrict pack_vars,
                        struct cell *ci, struct cell *cj, struct task *t,
                        struct engine *e, int4 *fparti_fpartj_lparti_lpartj, int *n_leafs_found,
                        int depth, int n_expected_tasks, struct cell ** ci_d, struct cell ** cj_d, int n_daughters) {

  /* Should we even bother? A. Nasar: For GPU code we need to be clever about
   * this */
  if (!CELL_IS_ACTIVE(ci, e) && !CELL_IS_ACTIVE(cj, e)) return;
  if (ci->hydro.count == 0 || cj->hydro.count == 0) return;

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
      /*Do we want to do anything before we recurse?*/

      /*We probably want to record */
      if (ci->progeny[pid] != NULL && cj->progeny[pjd] != NULL) {
        runner_recurse_gpu(r, s, pack_vars, ci->progeny[pid], cj->progeny[pjd],
                           t, e, fparti_fpartj_lparti_lpartj,
                           n_leafs_found, depth + 1, n_expected_tasks, ci_d, cj_d, n_daughters);
        //	        message("recursing to depth %i", depth + 1);
      }
    }
  } else if (CELL_IS_ACTIVE(ci, e) || CELL_IS_ACTIVE(cj, e)) {
    /* if any cell empty: skip */
    if (ci->hydro.count == 0 || cj->hydro.count == 0) return;
    int leafs_found = *n_leafs_found;
    /*for all leafs to be sent add to cell list */
    //	cells_left[leafs_found] = ci;
    //	cells_right[leafs_found] = cj;
    /*Add leaf cells to list for each top_level task*/
    ci_d[n_daughters + leafs_found] = ci;
    cj_d[n_daughters + leafs_found] = cj;

//    error("stop");
    (*n_leafs_found)++;  //= leafs_found + 1;
    if (*n_leafs_found >= n_expected_tasks)
      error("Created %i more than expected leaf cells. depth %i", *n_leafs_found, depth);
    //	double dist = sqrt(pow(ci->loc[0] - cj->loc[0], 2) + pow(ci->loc[1] -
    // cj->loc[1], 2)
    //	        + pow(ci->loc[2] - cj->loc[2], 2));
    //	if(dist > 0.5)
    //	  error("Incorect dists %f shifts %f %f %f", dist, shift[0], shift[1],
    // shift[2]);
  }
}

double runner_dopair1_pack_f4(struct runner *r, struct scheduler *s,
                              struct gpu_pack_vars *restrict pack_vars,
                              struct cell *ci, struct cell *cj, struct task *t,
                              struct part_aos_f4_send_d *parts_send,
                              struct engine *e,
                              int4 *fparti_fpartj_lparti_lpartj) {
  /* Timers for how long this all takes.
   * t0 and t1 are from start to finish including GPU calcs
   * tp0 and tp1 only time packing and unpacking*/
  struct timespec t0;
  struct timespec t1;
  clock_gettime(CLOCK_REALTIME, &t0);


  const int count_ci = ci->hydro.count;
  const int count_cj = cj->hydro.count;

  /*Get the id for the daughter task*/
  const int tid = pack_vars->tasks_packed;

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

  /*Pass into packing as double3 (Attempt at making packing more efficient)*/
  double3 shift_tmp = {shift[0], shift[1], shift[2]};

  /* Find first parts in task for ci and cj. Packed_tmp is index for cell i.
   * packed_tmp+1 is index for cell j */
  ////////////////////////
  // THIS IS A PROBLEM!!!
  ////////////////////////
  fparti_fpartj_lparti_lpartj[tid].x = pack_vars->count_parts;
  fparti_fpartj_lparti_lpartj[tid].y = pack_vars->count_parts + count_ci;

  size_t *count_parts = &pack_vars->count_parts;
  /* This re-arranges the particle data from cell->hydro->parts into a
  long array of part structs*/
  runner_do_ci_cj_gpu_pack_neat_aos_f4(
      r, ci, cj, parts_send, 0 /*timer. 0 no timing, 1 for timing*/,
      count_parts, tid, pack_vars->count_max_parts, count_ci, count_cj,
      shift_tmp);
  /* Find last parts in task for ci and cj*/
  ////////////////////////
  // THIS IS A PROBLEM!!!
  ////////////////////////
  fparti_fpartj_lparti_lpartj[tid].z = pack_vars->count_parts - count_cj;
  fparti_fpartj_lparti_lpartj[tid].w = pack_vars->count_parts;

  /* Tell the cells they have been packed */
  ci->pack_done++;
  cj->pack_done++;

  /* Identify first particle for each bundle of tasks */
  const int bundle_size = pack_vars->bundle_size;
  if (tid % bundle_size == 0) {
    int bid = tid / bundle_size;
    pack_vars->bundle_first_part[bid] = fparti_fpartj_lparti_lpartj[tid].x;

    // A. Nasar: This is possibly a problem!
    pack_vars->bundle_first_task_list[bid] = tid;
  }
  /* Record that we have now done a pair pack task & increment number of tasks
   * to offload*/
  pack_vars->tasks_packed++;

  /*Add time to packing_time. Timer for end of GPU work after the if(launch ||
   * launch_leftovers statement)*/
  clock_gettime(CLOCK_REALTIME, &t1);
  return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1000000000.0;
};

double runner_dopair1_pack_f4_gg(struct runner *r, struct scheduler *s,
                              struct gpu_pack_vars *restrict pack_vars,
                              struct cell *ci, struct cell *cj, struct task *t,
                              struct part_aos_f4_send_g *parts_send,
                              struct engine *e,
                              int4 *fparti_fpartj_lparti_lpartj) {
  /* Timers for how long this all takes.
   * t0 and t1 are from start to finish including GPU calcs
   * tp0 and tp1 only time packing and unpacking*/
  struct timespec t0;
  struct timespec t1;  //
  clock_gettime(CLOCK_REALTIME, &t0);

  const int count_ci = ci->hydro.count;
  const int count_cj = cj->hydro.count;

  /*Get the id for the daughter task*/
  const int tid = pack_vars->tasks_packed;

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

  /*Pass into packing as double3 (Attempt at making packing more efficient)*/
  double3 shift_tmp = {shift[0], shift[1], shift[2]};

  /* Find first parts in task for ci and cj. Packed_tmp is index for cell i.
   * packed_tmp+1 is index for cell j */
  ////////////////////////
  // THIS IS A PROBLEM!!!
  ////////////////////////
  fparti_fpartj_lparti_lpartj[tid].x = pack_vars->count_parts;
  fparti_fpartj_lparti_lpartj[tid].y = pack_vars->count_parts + count_ci;

  size_t *count_parts = &pack_vars->count_parts;
  /* This re-arranges the particle data from cell->hydro->parts into a
  long array of part structs*/
  runner_do_ci_cj_gpu_pack_neat_aos_f4_g(
      r, ci, cj, parts_send, 0 /*timer. 0 no timing, 1 for timing*/,
      count_parts, tid, pack_vars->count_max_parts, count_ci, count_cj,
      shift_tmp);
  /* Find last parts in task for ci and cj*/
  ////////////////////////
  // THIS IS A PROBLEM!!!
  ////////////////////////
  fparti_fpartj_lparti_lpartj[tid].z = pack_vars->count_parts - count_cj;
  fparti_fpartj_lparti_lpartj[tid].w = pack_vars->count_parts;

  /* Tell the cells they have been packed */
  ci->pack_done_g++;
  cj->pack_done_g++;

  /* Identify first particle for each bundle of tasks */
  const int bundle_size = pack_vars->bundle_size;
  if (tid % bundle_size == 0) {
    int bid = tid / bundle_size;
    pack_vars->bundle_first_part[bid] = fparti_fpartj_lparti_lpartj[tid].x;

    // A. Nasar: This is possibly a problem!
    pack_vars->bundle_first_task_list[bid] = tid;
  }
  /* Record that we have now done a pair pack task & increment number of tasks
   * to offload*/
  pack_vars->tasks_packed++;

  /*Add time to packing_time. Timer for end of GPU work after the if(launch ||
   * launch_leftovers statement)*/
  clock_gettime(CLOCK_REALTIME, &t1);
  return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1000000000.0;
};

double runner_dopair1_pack_f4_ff(struct runner *r, struct scheduler *s,
                              struct gpu_pack_vars *restrict pack_vars,
                              struct cell *ci, struct cell *cj, struct task *t,
                              struct part_aos_f4_send_f *parts_send,
                              struct engine *e,
                              int4 *fparti_fpartj_lparti_lpartj) {
  /* Timers for how long this all takes.
   * t0 and t1 are from start to finish including GPU calcs
   * tp0 and tp1 only time packing and unpacking*/
  struct timespec t0;
  struct timespec t1;  //
  clock_gettime(CLOCK_REALTIME, &t0);

  const int count_ci = ci->hydro.count;
  const int count_cj = cj->hydro.count;

  /*Get the id for the daughter task*/
  const int tid = pack_vars->tasks_packed;

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

  /*Pass into packing as double3 (Attempt at making packing more efficient)*/
  double3 shift_tmp = {shift[0], shift[1], shift[2]};

  /* Find first parts in task for ci and cj. Packed_tmp is index for cell i.
   * packed_tmp+1 is index for cell j */
  ////////////////////////
  // THIS IS A PROBLEM!!!
  ////////////////////////
  fparti_fpartj_lparti_lpartj[tid].x = pack_vars->count_parts;
  fparti_fpartj_lparti_lpartj[tid].y = pack_vars->count_parts + count_ci;

  size_t *count_parts = &pack_vars->count_parts;
  /* This re-arranges the particle data from cell->hydro->parts into a
  long array of part structs*/
  runner_do_ci_cj_gpu_pack_neat_aos_f4_f(
      r, ci, cj, parts_send, 0 /*timer. 0 no timing, 1 for timing*/,
      count_parts, tid, pack_vars->count_max_parts, count_ci, count_cj,
      shift_tmp);
  /* Find last parts in task for ci and cj*/
  ////////////////////////
  // THIS IS A PROBLEM!!!
  ////////////////////////
  fparti_fpartj_lparti_lpartj[tid].z = pack_vars->count_parts - count_cj;
  fparti_fpartj_lparti_lpartj[tid].w = pack_vars->count_parts;

  /* Tell the cells they have been packed */
  ci->pack_done_f++;
  cj->pack_done_f++;

  /* Identify first particle for each bundle of tasks */
  const int bundle_size = pack_vars->bundle_size;
  if (tid % bundle_size == 0) {
    int bid = tid / bundle_size;
    pack_vars->bundle_first_part[bid] = fparti_fpartj_lparti_lpartj[tid].x;

    // A. Nasar: This is possibly a problem!
    pack_vars->bundle_first_task_list[bid] = tid;
  }
  /* Record that we have now done a pair pack task & increment number of tasks
   * to offload*/
  pack_vars->tasks_packed++;

  /*Add time to packing_time. Timer for end of GPU work after the if(launch ||
   * launch_leftovers statement)*/
  clock_gettime(CLOCK_REALTIME, &t1);
  return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1000000000.0;
};


void runner_doself_gpu_launch(
    const struct runner *r,
    struct scheduler *s,
    struct gpu_offload_data *buf,
    const enum task_subtypes task_subtype,
    cudaStream_t *stream,
    const float d_a,
    const float d_H
    ) {

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
    pack_vars->bundle_first_part[n_bundles] = buf->task_first_part_f4[tasks_packed - 1].x;
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
        size_t count = buf->task_first_part_f4[tid].y - buf->task_first_part_f4[tid].x;
        max_parts = max(max_parts, count);
        last_task = tid;
      }
    }

    const size_t first_part_tmp = pack_vars->bundle_first_part[bid];
    const size_t bundle_n_parts = pack_vars->bundle_last_part[bid] - first_part_tmp;
    const size_t tasksperbundle = pack_vars->tasksperbundle;
    const size_t tasks_left = (bid == n_bundles - 1) ? tasks_packed - (n_bundles - 1) * tasksperbundle : tasksperbundle;

    /* Will launch a 2d grid of GPU thread blocks (number of tasks is
       the y dimension and max_parts is the x dimension */
    size_t numBlocks_y = tasks_left;
    size_t numBlocks_x = (max_parts + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t bundle_first_task = pack_vars->bundle_first_task_list[bid];

    if (task_subtype == task_subtype_density){

      cudaMemcpyAsync(&buf->d_task_first_part_f4[first_task],
                      &buf->task_first_part_f4[first_task],
                      (last_task + 1 - first_task) * sizeof(int2),
                      cudaMemcpyHostToDevice, stream[bid]);
      cudaMemcpyAsync(&buf->d_parts_send_d[first_part_tmp], &buf->parts_send_d[first_part_tmp],
                      bundle_n_parts * sizeof(struct part_aos_f4_send_d),
                      cudaMemcpyHostToDevice, stream[bid]);

      /* Launch the kernel */
      launch_density_aos_f4(buf->d_parts_send_d, buf->d_parts_recv_d, d_a, d_H, stream[bid],
                            numBlocks_x, numBlocks_y, bundle_first_task,
                            buf->d_task_first_part_f4);
      /* Copy back */
      cudaMemcpyAsync(&buf->parts_recv_d[first_part_tmp], &buf->d_parts_recv_d[first_part_tmp],
                      bundle_n_parts * sizeof(struct part_aos_f4_recv_d),
                      cudaMemcpyDeviceToHost, stream[bid]);
      cudaEventRecord(buf->event_end[bid], stream[bid]);
    } else {
      error("Unknown task subtype %s", subtaskID_names[task_subtype]);
    }

  } /*End of looping over bundles to launch in streams*/

  /* Make sure all the kernels and copies back are finished */
  /* cudaDeviceSynchronize(); */
}


void runner_doself_gpu_unpack(
    const struct runner *r,
    struct scheduler *s,
    struct gpu_offload_data *buf,
    const enum task_subtypes task_subtype,
    cudaStream_t *stream
    ) {

  const struct engine* e = r->e;

  /* Grab pack_vars */
  struct gpu_pack_vars *pack_vars = &buf->pv;

  /* Identify the number of GPU bundles to run in ideal case */
  size_t n_bundles = pack_vars->n_bundles_unpack;

  /*How many tasks have we packed?*/
  const size_t tasks_packed = pack_vars->tasks_packed;

  /*How many tasks should be in a bundle?*/
  const size_t bundle_size = pack_vars->bundle_size;

  /* Copy the data back from the CPU thread-local buffers to the cells */
  /* Pack length counter for use in unpacking */
  size_t pack_length_unpack = 0;
  for (size_t bid = 0; bid < n_bundles; bid++) {

    cudaEventSynchronize(buf->event_end[bid]);

    for (size_t tid = bid * bundle_size; tid < (bid + 1) * bundle_size; tid++) {

      if (tid < tasks_packed) {
        struct cell *cii = pack_vars->ci_list[tid];
        struct task *tii = pack_vars->task_list[tid];

        while (cell_locktree(cii)) {
          ; /* spin until we acquire the lock */
        }

        /* Do the copy */
        if (task_subtype == task_subtype_density){
          runner_doself1_gpu_unpack_neat_aos_f4(r, cii, buf->parts_recv_d, 0,
                                              &pack_length_unpack, tid,
                                              pack_vars->count_max_parts, e);
        } else {
          error("Unknown task subtype %s", subtaskID_names[task_subtype]);
        }

        /* Record things for debugging */
        cii->gpu_done++;

        pthread_mutex_lock(&s->sleep_mutex);
        atomic_dec(&s->waiting);
        pthread_cond_broadcast(&s->sleep_cond);
        pthread_mutex_unlock(&s->sleep_mutex);

        /* Release the lock */
        cell_unlocktree(cii);

        /* schedule my dependencies (Only unpacks really) */
        enqueue_dependencies(s, tii);
        /* Signal sleeping runners */
        // MATTHIEU signal_sleeping_runners(s, tii);
      }
    }
  }

  /* Zero counters for the next pack operations */
  pack_vars->count_parts = 0;
  pack_vars->tasks_packed = 0;
}


void runner_doself_gpu_density(
    struct runner *r,
    struct scheduler *s,
    struct gpu_offload_data *buf,
    struct task *t,
    cudaStream_t *stream,
    const float d_a,
    const float d_H
    ) {

  TIMER_TIC;
  runner_doself_gpu_launch(r, s, buf, t->subtype, stream, d_a, d_H);
  TIMER_TOC(timer_doself_gpu_launch_d);

  TIMER_TIC2;
  runner_doself_gpu_unpack(r, s, buf, t->subtype, stream);
  TIMER_TOC2(timer_doself_gpu_unpack_d);
}



void runner_doself_gpu_launch_gradient(
    struct runner *r, struct scheduler *s, struct gpu_pack_vars *pack_vars,
    struct cell *ci, struct task *t, struct part_aos_f4_send_g *parts_send,
    struct part_aos_f4_recv_g *parts_recv,
    struct part_aos_f4_send_g *d_parts_send,
    struct part_aos_f4_recv_g *d_parts_recv, cudaStream_t *stream, float d_a,
    float d_H, struct engine *e,
    int2 *task_first_part_f4, int2 *d_task_first_part_f4, cudaEvent_t *self_end) {

  /* Identify the number of GPU bundles to run in ideal case*/
  int n_bundles_temp = pack_vars->n_bundles;

  /*How many tasks have we packed?*/
  const int tasks_packed = pack_vars->tasks_packed;

  /*How many tasks should be in a bundle?*/
  const int bundle_size = pack_vars->bundle_size;

  /* Special case for incomplete bundles (when having leftover tasks not enough
   * to fill a bundle) */
  if (pack_vars->launch_leftovers) {
    n_bundles_temp = (tasks_packed + bundle_size - 1) / bundle_size;
    //	  if(tasks_packed == 0) error("zero tasks packed but somehow got into
    // GPU loop");
    pack_vars->bundle_first_part[n_bundles_temp] =
        task_first_part_f4[tasks_packed - 1].x;
  }

  /* Identify the last particle for each bundle of tasks */
  for (int bid = 0; bid < n_bundles_temp - 1; bid++) {
    pack_vars->bundle_last_part[bid] = pack_vars->bundle_first_part[bid + 1];
  }

  /* special treatment for the last bundle */
  if (n_bundles_temp > 1)
    pack_vars->bundle_last_part[n_bundles_temp - 1] = pack_vars->count_parts;
  else
    pack_vars->bundle_last_part[0] = pack_vars->count_parts;

  /* Launch the copies for each bundle and run the GPU kernel */
  /* We don't go into this loop if tasks_left_self == 1 as
   n_bundles_temp will be zero DUHDUHDUHDUHHHHHH!!!!!*/
  int max_parts;
  for (int bid = 0; bid < n_bundles_temp; bid++) {

    max_parts = 0;
    const int first_task = bid * bundle_size;
    int last_task = (bid + 1) * bundle_size;
    for (int tid = bid * bundle_size; tid < (bid + 1) * bundle_size; tid++) {
      if (tid < tasks_packed) {
        /*Get an estimate for the max number of parts per cell in the bundle.
         *  Used for determining the number of GPU CUDA blocks*/
        int count = task_first_part_f4[tid].y - task_first_part_f4[tid].x;
        max_parts = max(max_parts, count);
        last_task = tid;
      }
    }

    const int first_part_tmp = pack_vars->bundle_first_part[bid];
    const int bundle_n_parts =
        pack_vars->bundle_last_part[bid] - first_part_tmp;

    cudaMemcpyAsync(&d_task_first_part_f4[first_task],
                    &task_first_part_f4[first_task],
                    (last_task + 1 - first_task) * sizeof(int2),
                    cudaMemcpyHostToDevice, stream[bid]);

    cudaMemcpyAsync(&d_parts_send[first_part_tmp], &parts_send[first_part_tmp],
                    bundle_n_parts * sizeof(struct part_aos_f4_send_g),
                    cudaMemcpyHostToDevice, stream[bid]);

#ifdef CUDA_DEBUG
    cudaError_t cu_error = cudaPeekAtLastError();
    if (cu_error != cudaSuccess) {
      error("CUDA error in gradient self host 2 device memcpy: %s cpuid id is: %i",
          cudaGetErrorString(cu_error), r->cpuid);
    }
#endif
    const int tasksperbundle = pack_vars->tasksperbundle;
    int tasks_left = tasksperbundle;
    if (bid == n_bundles_temp - 1) {
      tasks_left = tasks_packed - (n_bundles_temp - 1) * tasksperbundle;
    }
    // Will launch a 2d grid of GPU thread blocks (number of tasks is
    // the y dimension and max_parts is the x dimension
    int numBlocks_y = tasks_left;
    int numBlocks_x = (max_parts + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int bundle_first_task = pack_vars->bundle_first_task_list[bid];
    //	  const char *loop_type = "density";
    // Launch the kernel
    launch_gradient_aos_f4(d_parts_send, d_parts_recv, d_a, d_H, stream[bid],
                           numBlocks_x, numBlocks_y, bundle_first_task,
                           d_task_first_part_f4);
#ifdef CUDA_DEBUG
    cu_error = cudaPeekAtLastError();  // Get error code
    if (cu_error != cudaSuccess) {
      error("CUDA error with self gradient kernel launch: %s cpuid id is: %i",
          cudaGetErrorString(cu_error), r->cpuid);
    }
#endif
    cudaMemcpyAsync(&parts_recv[first_part_tmp], &d_parts_recv[first_part_tmp],
                    bundle_n_parts * sizeof(struct part_aos_f4_recv_g),
                    cudaMemcpyDeviceToHost, stream[bid]);
    cudaEventRecord(self_end[bid], stream[bid]);

#ifdef CUDA_DEBUG
    cu_error = cudaPeekAtLastError();  // cudaGetLastError();        //
                                       // Get error code
    if (cu_error != cudaSuccess) {
      error("CUDA error with self gradient D2H memcpy: %s cpuid id is: %i",
              cudaGetErrorString(cu_error), r->cpuid);
    }
#endif
  } /*End of looping over bundles to launch in streams*/
  /* Make sure all the kernels and copies back are finished */
  /* cudaDeviceSynchronize(); */

  /* Now copy the data back from the CPU thread-local buffers to the cells */
  /* Pack length counter for use in unpacking */
  size_t pack_length_unpack = 0;
  ticks total_cpu_unpack_ticks = 0.;
  for (int bid = 0; bid < n_bundles_temp; bid++) {

    /* cudaStreamSynchronize(stream[bid]); */
    cudaEventSynchronize(self_end[bid]);

    for (int tid = bid * bundle_size; tid < (bid + 1) * bundle_size; tid++) {

      if (tid < tasks_packed) {

        struct cell *cii = pack_vars->ci_list[tid];
        struct task *tii = pack_vars->task_list[tid];

        while (cell_locktree(cii)) {
          ; /* spin until we acquire the lock */
        }
        const ticks tic = getticks();

        /* Do the copy */
        runner_doself1_gpu_unpack_neat_aos_f4_g(r, cii, parts_recv, 0,
                                                &pack_length_unpack, tid,
                                                pack_vars->count_max_parts, e);
        const ticks toc = getticks();

        total_cpu_unpack_ticks += toc - tic;
        /*Time end of unpacking*/

        /* Record things for debugging */
        cii->gpu_done_g++;
        pthread_mutex_lock(&s->sleep_mutex);
        atomic_dec(&s->waiting);
        pthread_cond_broadcast(&s->sleep_cond);
        pthread_mutex_unlock(&s->sleep_mutex);
        /* Release the lock */
        cell_unlocktree(cii);

        /* schedule my dependencies (Only unpacks really) */
        enqueue_dependencies(s, tii);
        /* Signal sleeping runners */
        // MATTHIEU signal_sleeping_runners(s, tii);
      }
    }
  }
  /* Zero counters for the next pack operations */
  pack_vars->count_parts = 0;
  pack_vars->tasks_packed = 0;
}

void runner_doself_gpu_launch_force(
    struct runner *r, struct scheduler *s, struct gpu_pack_vars *pack_vars,
    struct cell *ci, struct task *t, struct part_aos_f4_send_f *parts_send,
    struct part_aos_f4_recv_f *parts_recv,
    struct part_aos_f4_send_f *d_parts_send,
    struct part_aos_f4_recv_f *d_parts_recv, cudaStream_t *stream, float d_a,
    float d_H, struct engine *e,
    int2 *task_first_part_f4_f, int2 *d_task_first_part_f4_f,
    cudaEvent_t *self_end) {

  /* Identify the number of GPU bundles to run in ideal case*/
  int n_bundles_temp = pack_vars->n_bundles;

  /*How many tasks have we packed?*/
  const int tasks_packed = pack_vars->tasks_packed;

  /*How many tasks should be in a bundle?*/
  const int bundle_size = pack_vars->bundle_size;

  /* Special case for incomplete bundles (when having leftover tasks not enough
   * to fill a bundle) */
  if (pack_vars->launch_leftovers) {
    n_bundles_temp = (tasks_packed + bundle_size - 1) / bundle_size;
    if (tasks_packed == 0)
      error("zero tasks packed but somehow got into GPU loop");
    pack_vars->bundle_first_part[n_bundles_temp] =
        task_first_part_f4_f[tasks_packed - 1].x;
  }

  /* Identify the last particle for each bundle of tasks */
  for (int bid = 0; bid < n_bundles_temp - 1; bid++) {
    pack_vars->bundle_last_part[bid] = pack_vars->bundle_first_part[bid + 1];
  }

  /* special treatment for the last bundle */
  if (n_bundles_temp > 1)
    pack_vars->bundle_last_part[n_bundles_temp - 1] = pack_vars->count_parts;
  else
    pack_vars->bundle_last_part[0] = pack_vars->count_parts;

  /* Launch the copies for each bundle and run the GPU kernel */
  /* We don't go into this loop if tasks_left_self == 1 as
   n_bundles_temp will be zero DUHDUHDUHDUHHHHHH!!!!!*/
  int max_parts = 0;
  for (int bid = 0; bid < n_bundles_temp; bid++) {

    max_parts = 0;
    const int first_task = bid * bundle_size;
    int last_task = (bid + 1) * bundle_size;
    for (int tid = bid * bundle_size; tid < (bid + 1) * bundle_size; tid++) {
      if (tid < tasks_packed) {
        /*Get an estimate for the max number of parts per cell in the bundle.
         *  Used for determining the number of GPU CUDA blocks*/
        int count = task_first_part_f4_f[tid].y - task_first_part_f4_f[tid].x;
        max_parts = max(max_parts, count);
        last_task = tid;
      }
    }

    const int first_part_tmp = pack_vars->bundle_first_part[bid];
    const int bundle_n_parts =
        pack_vars->bundle_last_part[bid] - first_part_tmp;
    cudaMemcpyAsync(&d_task_first_part_f4_f[first_task],
                    &task_first_part_f4_f[first_task],
                    (last_task + 1 - first_task) * sizeof(int2),
                    cudaMemcpyHostToDevice, stream[bid]);

    cudaMemcpyAsync(&d_parts_send[first_part_tmp], &parts_send[first_part_tmp],
                    bundle_n_parts * sizeof(struct part_aos_f4_send_f),
                    cudaMemcpyHostToDevice, stream[bid]);

#ifdef CUDA_DEBUG
    cudaError_t cu_error = cudaPeekAtLastError();
    if (cu_error != cudaSuccess) {
      error("CUDA error in density self host 2 device memcpy: %s cpuid id "
              "is: %i", cudaGetErrorString(cu_error), r->cpuid);
    }
#endif
    const int tasksperbundle = pack_vars->tasksperbundle;
    int tasks_left = tasksperbundle;
    if (bid == n_bundles_temp - 1) {
      tasks_left = tasks_packed - (n_bundles_temp - 1) * tasksperbundle;
    }
    // Will launch a 2d grid of GPU thread blocks (number of tasks is
    // the y dimension and max_parts is the x dimension
    int numBlocks_y = tasks_left;
    int numBlocks_x = (max_parts + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int bundle_first_task = pack_vars->bundle_first_task_list[bid];

    // Launch the kernel
    launch_force_aos_f4(d_parts_send, d_parts_recv, d_a, d_H, stream[bid],
                        numBlocks_x, numBlocks_y, bundle_first_task,
                        d_task_first_part_f4_f);
#ifdef CUDA_DEBUG
    cu_error = cudaPeekAtLastError();  // Get error code
    if (cu_error != cudaSuccess) {
      error("CUDA error with self force kernel launch: %s cpuid id is: %i",
              cudaGetErrorString(cu_error), r->cpuid);
    }
#endif
    cudaMemcpyAsync(&parts_recv[first_part_tmp], &d_parts_recv[first_part_tmp],
                    bundle_n_parts * sizeof(struct part_aos_f4_recv_f),
                    cudaMemcpyDeviceToHost, stream[bid]);
    cudaEventRecord(self_end[bid], stream[bid]);

#ifdef CUDA_DEBUG
    cu_error = cudaPeekAtLastError();
    if (cu_error != cudaSuccess) {
      error("CUDA error with self firce D2H memcpy: %s cpuid id is: %i"
            "Something's up with your cuda code",
            cudaGetErrorString(cu_error), r->cpuid);
    }
#endif
  } /*End of looping over bundles to launch in streams*/

  /* Make sure all the kernels and copies back are finished */
  //	cudaDeviceSynchronize();

  /* Now copy the data back from the CPU thread-local buffers to the cells */
  /* Pack length counter for use in unpacking */
  size_t pack_length_unpack = 0;
  for (int bid = 0; bid < n_bundles_temp; bid++) {

    //		cudaStreamSynchronize(stream[bid]);
    cudaEventSynchronize(self_end[bid]);

    for (int tid = bid * bundle_size; tid < (bid + 1) * bundle_size; tid++) {

      if (tid < tasks_packed) {
        struct cell *cii = pack_vars->ci_list[tid];
        struct task *tii = pack_vars->task_list[tid];

        while (cell_locktree(cii)) {
          ; /* spin until we acquire the lock */
        }

        /* Do the copy */
        runner_doself1_gpu_unpack_neat_aos_f4_f(r, cii, parts_recv, 0,
                                                &pack_length_unpack, tid,
                                                pack_vars->count_max_parts, e);

        /* Record things for debugging */
        cii->gpu_done_f++;
        pthread_mutex_lock(&s->sleep_mutex);
        atomic_dec(&s->waiting);
        pthread_cond_broadcast(&s->sleep_cond);
        pthread_mutex_unlock(&s->sleep_mutex);
        /* Release the lock */
        cell_unlocktree(cii);

        /*schedule my dependencies (Only unpacks really)*/
        enqueue_dependencies(s, tii);
      }
    }
  }

  /* Zero counters for the next pack operations */
  pack_vars->count_parts = 0;
  pack_vars->tasks_packed = 0;
}

void runner_dopair_launch_d(
    struct runner *r, struct scheduler *s, struct gpu_pack_vars *pack_vars,
    struct task *t, struct part_aos_f4_send_d *parts_send,
    struct part_aos_f4_recv_d *parts_recv, struct part_aos_f4_send_d *d_parts_send,
    struct part_aos_f4_recv_d *d_parts_recv, cudaStream_t *stream, float d_a,
    float d_H, struct engine *e, int4 *fparti_fpartj_lparti_lpartj_dens,
    cudaEvent_t *pair_end) {

  /* Identify the number of GPU bundles to run in ideal case*/
  int n_bundles_temp = pack_vars->n_bundles;
  /*How many tasks have we packed?*/
  const int tasks_packed = pack_vars->tasks_packed;

  /*How many tasks should be in a bundle?*/
  const int bundle_size = pack_vars->bundle_size;

  /* Special case for incomplete bundles (when having leftover tasks not enough
   * to fill a bundle) */
  if (pack_vars->launch_leftovers) {
    n_bundles_temp = (tasks_packed + bundle_size - 1) / bundle_size;
    pack_vars->bundle_first_part[n_bundles_temp] =
        fparti_fpartj_lparti_lpartj_dens[tasks_packed - 1].x;
    //    message("Incomplete buundle");
  }
  /* Identify the last particle for each bundle of tasks */
  for (int bid = 0; bid < n_bundles_temp - 1; bid++) {
    pack_vars->bundle_last_part[bid] = pack_vars->bundle_first_part[bid + 1];
  }
  /* special treatment for the case of 1 bundle */
  if (n_bundles_temp > 1)
    pack_vars->bundle_last_part[n_bundles_temp - 1] = pack_vars->count_parts;
  else
    pack_vars->bundle_last_part[0] = pack_vars->count_parts;

  /* Launch the copies for each bundle and run the GPU kernel */
  for (int bid = 0; bid < n_bundles_temp; bid++) {

    int max_parts_i = 0;
    int max_parts_j = 0;
    for (int tid = bid * bundle_size; tid < (bid + 1) * bundle_size; tid++) {
      if (tid < tasks_packed) {
        /*Get an estimate for the max number of parts per cell in each bundle.
         *  Used for determining the number of GPU CUDA blocks*/
        int count_i = fparti_fpartj_lparti_lpartj_dens[tid].z -
                      fparti_fpartj_lparti_lpartj_dens[tid].x;
        max_parts_i = max(max_parts_i, count_i);
        int count_j = fparti_fpartj_lparti_lpartj_dens[tid].w -
                      fparti_fpartj_lparti_lpartj_dens[tid].y;
        max_parts_j = max(max_parts_j, count_j);
      }
    }
    const int first_part_tmp_i = pack_vars->bundle_first_part[bid];
    const int bundle_n_parts =
        pack_vars->bundle_last_part[bid] - first_part_tmp_i;

    cudaMemcpyAsync(&d_parts_send[first_part_tmp_i],
                    &parts_send[first_part_tmp_i],
                    bundle_n_parts * sizeof(struct part_aos_f4_send_d),
                    cudaMemcpyHostToDevice, stream[bid]);

#ifdef CUDA_DEBUG
    cudaError_t cu_error =
        cudaPeekAtLastError();  // cudaGetLastError();        //
                                // Get error code
    if (cu_error != cudaSuccess) {
      error("CUDA error with pair density H2D async  memcpy ci: %s cpuid id is: %i"
            "Something's up with your cuda code first_part %i bundle size %i",
            cudaGetErrorString(cu_error), r->cpuid, first_part_tmp_i, bundle_n_parts);
    }
#endif
    /* LAUNCH THE GPU KERNELS for ci & cj */
    // Setup 2d grid of GPU thread blocks for ci (number of tasks is
    // the y dimension and max_parts is the x dimension
    int numBlocks_y = 0;  // tasks_left; //Changed this to 1D grid of blocks so
                          // this is no longer necessary
    int numBlocks_x = (bundle_n_parts + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int bundle_part_0 = pack_vars->bundle_first_part[bid];
    /* Launch the kernel for ci using data for ci and cj */
    runner_dopair_branch_density_gpu_aos_f4(
        d_parts_send, d_parts_recv, d_a, d_H, stream[bid], numBlocks_x,
        numBlocks_y, bundle_part_0, bundle_n_parts);

#ifdef CUDA_DEBUG
    cu_error = cudaPeekAtLastError();  // Get error code
    if (cu_error != cudaSuccess) {
      error(
          "CUDA error with pair density kernel launch: %s cpuid id is: %i\n "
          "nbx %i nby %i max_parts_i %i max_parts_j %i\n"
          "Something's up with kernel launch.",
          cudaGetErrorString(cu_error), r->cpuid, numBlocks_x, numBlocks_y,
          max_parts_i, max_parts_j);
    }
#endif

    // Copy results back to CPU BUFFERS
    cudaMemcpyAsync(&parts_recv[first_part_tmp_i],
                    &d_parts_recv[first_part_tmp_i],
                    bundle_n_parts * sizeof(struct part_aos_f4_recv_d),
                    cudaMemcpyDeviceToHost, stream[bid]);
    // Issue event to be recorded by GPU after copy back to CPU
    cudaEventRecord(pair_end[bid], stream[bid]);

#ifdef CUDA_DEBUG
    cu_error = cudaPeekAtLastError();  // cudaGetLastError();        //
                                       // Get error code
    if (cu_error != cudaSuccess) {
      error("CUDA error with self density D2H memcpy: %s cpuid id is: %i\n"
            "Something's up with your cuda code",
            cudaGetErrorString(cu_error), r->cpuid);
    }
#endif
  } /*End of looping over bundles to launch in streams*/

  /* Issue synchronisation commands for all events recorded by GPU
   * Should swap with one cuda Device Synchronise really if we decide to go this
   * way with unpacking done separately */
  for (int bid = 0; bid < n_bundles_temp; bid++) {
    /*Time unpacking*/
    cudaEventSynchronize(pair_end[bid]);
  }

} /*End of GPU work*/

void runner_dopair_unpack_d(
    struct runner *r, struct scheduler *s, struct gpu_pack_vars *pack_vars,
    struct task *t, struct part_aos_f4_send_d *parts_send,
    struct part_aos_f4_recv_d *parts_recv, struct part_aos_f4_send_d *d_parts_send,
    struct part_aos_f4_recv_d *d_parts_recv, cudaStream_t *stream, float d_a,
    float d_H, struct engine *e, int4 *fparti_fpartj_lparti_lpartj_dens,
    cudaEvent_t *pair_end, int npacked, int n_leaves_found, struct cell ** ci_d, struct cell ** cj_d, int ** f_l_daughters
    , struct cell ** ci_top, struct cell ** cj_top) {

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
    for (int tid = f_l_daughters[topid][0]; tid < f_l_daughters[topid][1]; tid++){
      /*Get pointers to the leaf cells*/
      struct cell *cii_l = ci_d[tid];
      struct cell *cjj_l = cj_d[tid];
      runner_do_ci_cj_gpu_unpack_neat_aos_f4(r, cii_l, cjj_l, parts_recv, 0,
                                             &pack_length_unpack, tid,
                                             2 * pack_vars->count_max_parts, e);
    }
    cell_unlocktree(ci_top[topid]);
    cell_unlocktree(cj_top[topid]);

    /*For some reason the code fails if we get a leaf pair task
     *this if->continue statement stops the code from trying to unlock same
     *cells twice*/
    if (topid == pack_vars->top_tasks_packed - 1 && npacked != n_leaves_found) {
      continue;
    }
    enqueue_dependencies(s, pack_vars->top_task_list[topid]);
    pthread_mutex_lock(&s->sleep_mutex);
    atomic_dec(&s->waiting);
    pthread_cond_broadcast(&s->sleep_cond);
    pthread_mutex_unlock(&s->sleep_mutex);
  }
}


void runner_gpu_pack_daughters_and_launch_d(struct runner *r, struct scheduler *s,
    struct cell * ci, struct cell * cj,
    struct gpu_pack_vars *pack_vars, struct task *t,
    struct part_aos_f4_send_d *parts_send,
    struct part_aos_f4_recv_d *parts_recv,
    struct part_aos_f4_send_d *d_parts_send,
    struct part_aos_f4_recv_d *d_parts_recv,
    cudaStream_t *stream, float d_a, float d_H,
    struct engine *e, int4 *fparti_fpartj_lparti_lpartj_dens,
    cudaEvent_t *pair_end, int n_leaves_found, struct cell ** ci_d, struct cell ** cj_d, int ** f_l_daughters,
    struct cell ** ci_top, struct cell ** cj_top){

  /* Everything from here on needs moving to runner_doiact_functions_hydro_gpu.h */
  pack_vars->n_daughters_total += n_leaves_found;
  int top_tasks_packed = pack_vars->top_tasks_packed;

  //Keep separate
  f_l_daughters[top_tasks_packed][0] = pack_vars->n_daughters_packed_index;

  //Keep separate
  ci_top[top_tasks_packed] = ci;
  cj_top[top_tasks_packed] = cj;
  pack_vars->n_leaves_total += n_leaves_found;
  int first_cell_to_move = pack_vars->n_daughters_packed_index;
  int n_daughters_left = pack_vars->n_daughters_total;

  /*Get pointer to top level task. Needed to enqueue deps*/
  pack_vars->top_task_list[top_tasks_packed] = t;

  /*Increment how many top tasks we've packed*/
  pack_vars->top_tasks_packed++;
  top_tasks_packed++;

  //A. Nasar: Remove this from struct as not needed. Was only used for de-bugging
  /* How many daughter tasks do we want to offload at once?*/
  size_t target_n_tasks_tmp = pack_vars->target_n_tasks;

  // A. Nasar: Check to see if this is the last task in the queue.
  // If so, set launch_leftovers to 1 and recursively pack and launch daughter tasks on GPU
  unsigned int qid = r->qid;
  lock_lock(&s->queues[qid].lock);
  s->queues[qid].n_packs_pair_left_d--;
  if (s->queues[qid].n_packs_pair_left_d < 1) pack_vars->launch_leftovers = 1;
  (void)lock_unlock(&s->queues[qid].lock);
  /* Counter for how many tasks we've packed */
  int npacked = 0;
  int launched = 0;

  //A. Nasar: Loop through the daughter tasks we found
  int copy_index = pack_vars->n_daughters_packed_index;
  //not strictly true!!! Could be that we packed and moved on without launching
  int had_prev_task = 0;
  if(pack_vars->n_daughters_packed_index > 0)
    had_prev_task = 1;

  cell_unlocktree(ci);
  cell_unlocktree(cj);

  while(npacked < n_leaves_found){
    top_tasks_packed = pack_vars->top_tasks_packed;
    struct cell * cii = ci_d[copy_index];
    struct cell * cjj = cj_d[copy_index];
    runner_dopair1_pack_f4(
        r, s, pack_vars, cii, cjj, t,
        parts_send, e, fparti_fpartj_lparti_lpartj_dens);

    //record number of tasks we've copied from last launch
    copy_index++;
    //Record how many daughters we've packed in total for while loop
    npacked++;
    first_cell_to_move++;

    /*Re-think this. Probably not necessary to have this if statement and we can just use the form in the else statement*/
    if(had_prev_task)
      f_l_daughters[top_tasks_packed - 1][1] = f_l_daughters[top_tasks_packed - 1][0] + copy_index - pack_vars->n_daughters_packed_index;
    else
      f_l_daughters[top_tasks_packed - 1][1] = f_l_daughters[top_tasks_packed - 1][0] + copy_index;

    if(pack_vars->tasks_packed == target_n_tasks_tmp)
        pack_vars->launch = 1;
    if(pack_vars->launch || (pack_vars->launch_leftovers && npacked == n_leaves_found)){

      launched = 1;
      /* Here we only launch the tasks. No unpacking! This is done in next function ;) */
      runner_dopair_launch_d(
            r, s, pack_vars, t, parts_send,
            parts_recv, d_parts_send,
            d_parts_recv, stream, d_a, d_H, e,
            fparti_fpartj_lparti_lpartj_dens,
            pair_end);
      runner_dopair_unpack_d(
            r, s, pack_vars, t, parts_send,
            parts_recv, d_parts_send,
            d_parts_recv, stream, d_a, d_H, e,
            fparti_fpartj_lparti_lpartj_dens,
            pair_end, npacked, n_leaves_found, ci_d, cj_d, f_l_daughters, ci_top, cj_top);
      if(npacked == n_leaves_found){
        pack_vars->n_daughters_total = 0;
        pack_vars->top_tasks_packed = 0;
      }
      //Special treatment required here. Launched but have not packed all tasks
      else{
        //If we launch but still have daughters left re-set this task to be the first in the list
        //so that we can continue packing correctly
        pack_vars->top_task_list[0] = t;
        //Move all tasks forward in list so that the first next task will be packed to index 0
        //Move remaining cell indices so that their indexing starts from zero and ends in n_daughters_left
        for(int i = first_cell_to_move; i < n_daughters_left; i++){
          int shuffle = i - first_cell_to_move;
          ci_d[shuffle] = ci_d[i];
          cj_d[shuffle] = cj_d[i];
        }
        copy_index = 0;
        f_l_daughters[0][0] = 0;
        f_l_daughters[0][1] = n_daughters_left - first_cell_to_move;

        cj_top[0] = cj;
        ci_top[0] = ci;

        n_daughters_left -= first_cell_to_move;
        first_cell_to_move = 0;

        pack_vars->top_tasks_packed = 1;
        had_prev_task = 0;
      }
      pack_vars->tasks_packed = 0;
      pack_vars->count_parts = 0;
      pack_vars->launch = 0;
    }
    //case when we have launched then gone back to pack but did not pack enough to launch again
    else if(npacked == n_leaves_found){
      if(launched == 1){
        pack_vars->n_daughters_total = n_daughters_left;
        cj_top[0] = cj;
        ci_top[0] = ci;
        f_l_daughters[0][0] = 0;
        f_l_daughters[0][1] = n_daughters_left;
        pack_vars->top_tasks_packed = 1;
        pack_vars->top_task_list[0] = t;
        launched = 0;
      }
      else {
        f_l_daughters[top_tasks_packed - 1][0] = pack_vars->n_daughters_packed_index;
        f_l_daughters[top_tasks_packed - 1][1] = pack_vars->n_daughters_total;
      }
      pack_vars->launch = 0;
    }
    pack_vars->launch = 0;
  }
  //A. Nasar: Launch-leftovers counter re-set to zero and cells unlocked
  pack_vars->launch_leftovers = 0;
  pack_vars->launch = 0;

}

void runner_dopair_launch_g(
    struct runner *r, struct scheduler *s, struct gpu_pack_vars *pack_vars,
    struct task *t, struct part_aos_f4_send_g *parts_send,
    struct part_aos_f4_recv_g *parts_recv, struct part_aos_f4_send_g *d_parts_send,
    struct part_aos_f4_recv_g *d_parts_recv, cudaStream_t *stream, float d_a,
    float d_H, struct engine *e, int4 *fparti_fpartj_lparti_lpartj_grad,
    cudaEvent_t *pair_end) {

  /* Identify the number of GPU bundles to run in ideal case*/
  int n_bundles_temp = pack_vars->n_bundles;
  /*How many tasks have we packed?*/
  const int tasks_packed = pack_vars->tasks_packed;

  /*How many tasks should be in a bundle?*/
  const int bundle_size = pack_vars->bundle_size;

  /* Special case for incomplete bundles (when having leftover tasks not enough
   * to fill a bundle) */
  if (pack_vars->launch_leftovers) {
    n_bundles_temp = (tasks_packed + bundle_size - 1) / bundle_size;
    pack_vars->bundle_first_part[n_bundles_temp] =
        fparti_fpartj_lparti_lpartj_grad[tasks_packed - 1].x;
  }
  /* Identify the last particle for each bundle of tasks */
  for (int bid = 0; bid < n_bundles_temp - 1; bid++) {
    pack_vars->bundle_last_part[bid] = pack_vars->bundle_first_part[bid + 1];
  }
  /* special treatment for the case of 1 bundle */
  if (n_bundles_temp > 1)
    pack_vars->bundle_last_part[n_bundles_temp - 1] = pack_vars->count_parts;
  else
    pack_vars->bundle_last_part[0] = pack_vars->count_parts;

  /* Launch the copies for each bundle and run the GPU kernel */
  for (int bid = 0; bid < n_bundles_temp; bid++) {

    int max_parts_i = 0;
    int max_parts_j = 0;
    for (int tid = bid * bundle_size; tid < (bid + 1) * bundle_size; tid++) {
      if (tid < tasks_packed) {
        /*Get an estimate for the max number of parts per cell in each bundle.
         *  Used for determining the number of GPU CUDA blocks*/
        int count_i = fparti_fpartj_lparti_lpartj_grad[tid].z -
                      fparti_fpartj_lparti_lpartj_grad[tid].x;
        max_parts_i = max(max_parts_i, count_i);
        int count_j = fparti_fpartj_lparti_lpartj_grad[tid].w -
                      fparti_fpartj_lparti_lpartj_grad[tid].y;
        max_parts_j = max(max_parts_j, count_j);
      }
    }
    const int first_part_tmp_i = pack_vars->bundle_first_part[bid];
    const int bundle_n_parts =
        pack_vars->bundle_last_part[bid] - first_part_tmp_i;

    cudaMemcpyAsync(&d_parts_send[first_part_tmp_i],
                    &parts_send[first_part_tmp_i],
                    bundle_n_parts * sizeof(struct part_aos_f4_send_g),
                    cudaMemcpyHostToDevice, stream[bid]);

#ifdef CUDA_DEBUG
    cudaError_t cu_error =
        cudaPeekAtLastError();
    if (cu_error != cudaSuccess) {
      error(
              "CUDA error with pair density H2D async  memcpy ci: %s cpuid id is: %i\n"
              "Something's up with your cuda code first_part %i bundle size %i",
              cudaGetErrorString(cu_error), r->cpuid, first_part_tmp_i, bundle_n_parts);
    }
#endif
    /* LAUNCH THE GPU KERNELS for ci & cj */
    // Setup 2d grid of GPU thread blocks for ci (number of tasks is
    // the y dimension and max_parts is the x dimension
    int numBlocks_y = 0;  // tasks_left; //Changed this to 1D grid of blocks so
                          // this is no longer necessary
    int numBlocks_x = (bundle_n_parts + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int bundle_part_0 = pack_vars->bundle_first_part[bid];
    /* Launch the kernel for ci using data for ci and cj */
    runner_dopair_branch_gradient_gpu_aos_f4(
        d_parts_send, d_parts_recv, d_a, d_H, stream[bid], numBlocks_x,
        numBlocks_y, bundle_part_0, bundle_n_parts);

#ifdef CUDA_DEBUG
    cu_error = cudaPeekAtLastError();  // Get error code
    if (cu_error != cudaSuccess) {
      error(
          "CUDA error with pair density kernel launch: %s cpuid id is: %i\n "
          "nbx %i nby %i max_parts_i %i max_parts_j %i\n"
          "Something's up with kernel launch.",
          cudaGetErrorString(cu_error), r->cpuid, numBlocks_x, numBlocks_y,
          max_parts_i, max_parts_j);
    }
#endif

    // Copy results back to CPU BUFFERS
    cudaMemcpyAsync(&parts_recv[first_part_tmp_i],
                    &d_parts_recv[first_part_tmp_i],
                    bundle_n_parts * sizeof(struct part_aos_f4_recv_g),
                    cudaMemcpyDeviceToHost, stream[bid]);
    // Issue event to be recorded by GPU after copy back to CPU
    cudaEventRecord(pair_end[bid], stream[bid]);

#ifdef CUDA_DEBUG
    cu_error = cudaPeekAtLastError();  // cudaGetLastError();        //
                                       // Get error code
    if (cu_error != cudaSuccess) {
      error("CUDA error with self density D2H memcpy: %s cpuid id is: %i\n"
            "Something's up with your cuda code", cudaGetErrorString(cu_error), r->cpuid);
    }
#endif
  } /*End of looping over bundles to launch in streams*/

  /* Issue synchronisation commands for all events recorded by GPU
   * Should swap with one cuda Device Synchronise really if we decide to go this
   * way with unpacking done separately */
  for (int bid = 0; bid < n_bundles_temp; bid++) {
    /*Time unpacking*/
    /* clock_gettime(CLOCK_REALTIME, &t0); */
    cudaEventSynchronize(pair_end[bid]);

    /* clock_gettime(CLOCK_REALTIME, &t1); */
    /* *gpu_time += */
    /*     (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1000000000.0; */
  }
}

void runner_dopair1_unpack_f4_g(
    struct runner *r, struct scheduler *s, struct gpu_pack_vars *pack_vars,
    struct task *t, struct part_aos_f4_send_g *parts_send,
    struct part_aos_f4_recv_g *parts_recv, struct part_aos_f4_send_g *d_parts_send,
    struct part_aos_f4_recv_g *d_parts_recv, cudaStream_t *stream, float d_a,
    float d_H, struct engine *e, int4 *fparti_fpartj_lparti_lpartj_grad,
    cudaEvent_t *pair_end, int npacked, int n_leaves_found, struct cell ** ci_d, struct cell ** cj_d, int ** f_l_daughters
    , struct cell ** ci_top, struct cell ** cj_top) {

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
    for (int tid = f_l_daughters[topid][0]; tid < f_l_daughters[topid][1]; tid++){
      /*Get pointers to the leaf cells*/
      struct cell *cii_l = ci_d[tid];
      struct cell *cjj_l = cj_d[tid];
      runner_do_ci_cj_gpu_unpack_neat_aos_f4_g(r, cii_l, cjj_l, parts_recv, 0,
                                             &pack_length_unpack, tid,
                                             2 * pack_vars->count_max_parts, e);
    }
    cell_unlocktree(ci_top[topid]);
    cell_unlocktree(cj_top[topid]);

    /*For some reason the code fails if we get a leaf pair task
     *this if->continue statement stops the code from trying to unlock same
     *cells twice*/
    if (topid == pack_vars->top_tasks_packed - 1 && npacked != n_leaves_found) {
      continue;
    }
    enqueue_dependencies(s, pack_vars->top_task_list[topid]);
    pthread_mutex_lock(&s->sleep_mutex);
    atomic_dec(&s->waiting);
    pthread_cond_broadcast(&s->sleep_cond);
    pthread_mutex_unlock(&s->sleep_mutex);
  }
}

void runner_gpu_pack_daughters_and_launch_g(struct runner *r, struct scheduler *s, struct cell * ci, struct cell * cj, struct gpu_pack_vars *pack_vars,
    struct task *t, struct part_aos_f4_send_g *parts_send,
    struct part_aos_f4_recv_g *parts_recv, struct part_aos_f4_send_g *d_parts_send,
    struct part_aos_f4_recv_g *d_parts_recv, cudaStream_t *stream, float d_a,
    float d_H, struct engine *e, int4 *fparti_fpartj_lparti_lpartj_grad,
    cudaEvent_t *pair_end, int n_leaves_found, struct cell ** ci_d, struct cell ** cj_d, int ** f_l_daughters
    , struct cell ** ci_top, struct cell ** cj_top){
  //Everything from here on needs moving to runner_doiact_functions_hydro_gpu.h
  pack_vars->n_daughters_total += n_leaves_found;
  int top_tasks_packed = pack_vars->top_tasks_packed;
  f_l_daughters[top_tasks_packed][0] = pack_vars->n_daughters_packed_index;
  ci_top[top_tasks_packed] = ci;
  cj_top[top_tasks_packed] = cj;
  pack_vars->n_leaves_total += n_leaves_found;
  int first_cell_to_move = pack_vars->n_daughters_packed_index;
  int n_daughters_left = pack_vars->n_daughters_total;
  /*Get pointer to top level task. Needed to enqueue deps*/
  pack_vars->top_task_list[top_tasks_packed] = t;
  /*Increment how many top tasks we've packed*/
  pack_vars->top_tasks_packed++;
  top_tasks_packed++;
  //A. Nasar: Remove this from struct as not needed. Was only used for de-bugging
  /*How many daughter tasks do we want to offload at once?*/
  size_t target_n_tasks_tmp = pack_vars->target_n_tasks;
  // A. Nasar: Check to see if this is the last task in the queue.
  // If so, set launch_leftovers to 1 and recursively pack and launch daughter tasks on GPU
  unsigned int qid = r->qid;
  lock_lock(&s->queues[qid].lock);
  s->queues[qid].n_packs_pair_left_g--;
  if (s->queues[qid].n_packs_pair_left_g < 1) pack_vars->launch_leftovers = 1;
  (void)lock_unlock(&s->queues[qid].lock);
  /*Counter for how many tasks we've packed*/
  int npacked = 0;
  int launched = 0;
  //A. Nasar: Loop through the daughter tasks we found
  int copy_index = pack_vars->n_daughters_packed_index;
  //not strictly true!!! Could be that we packed and moved on without launching
  int had_prev_task = 0;
  if(pack_vars->n_daughters_packed_index > 0)
    had_prev_task = 1;

  cell_unlocktree(ci);
  cell_unlocktree(cj);

  while(npacked < n_leaves_found){
    top_tasks_packed = pack_vars->top_tasks_packed;
    struct cell * cii = ci_d[copy_index];
    struct cell * cjj = cj_d[copy_index];
    runner_dopair1_pack_f4_gg(
        r, s, pack_vars, cii, cjj, t,
        parts_send, e, fparti_fpartj_lparti_lpartj_grad);
    //record number of tasks we've copied from last launch
    copy_index++;
    //Record how many daughters we've packed in total for while loop
    npacked++;
    first_cell_to_move++;
    /*Re-think this. Probably not necessary to have this if statement and we can just use the form in the else statement*/
    if(had_prev_task)
      f_l_daughters[top_tasks_packed - 1][1] = f_l_daughters[top_tasks_packed - 1][0] + copy_index - pack_vars->n_daughters_packed_index;
    else
      f_l_daughters[top_tasks_packed - 1][1] = f_l_daughters[top_tasks_packed - 1][0] + copy_index;

    if(pack_vars->tasks_packed == target_n_tasks_tmp)
        pack_vars->launch = 1;
    if(pack_vars->launch || (pack_vars->launch_leftovers && npacked == n_leaves_found)){

//      message("tasks packed gradient %i", pack_vars->tasks_packed);
      launched = 1;
      //Here we only launch the tasks. No unpacking! This is done in next function ;)
      runner_dopair_launch_g(
            r, s, pack_vars, t, parts_send,
            parts_recv, d_parts_send,
            d_parts_recv, stream, d_a, d_H, e,
            fparti_fpartj_lparti_lpartj_grad,
            pair_end);
      runner_dopair1_unpack_f4_g(
            r, s, pack_vars, t, parts_send,
            parts_recv, d_parts_send,
            d_parts_recv, stream, d_a, d_H, e,
            fparti_fpartj_lparti_lpartj_grad,
            pair_end, npacked, n_leaves_found, ci_d, cj_d, f_l_daughters, ci_top, cj_top);
      if(npacked == n_leaves_found){
        pack_vars->n_daughters_total = 0;
        pack_vars->top_tasks_packed = 0;
      }
      //Special treatment required here. Launched but have not packed all tasks
      else{
        //If we launch but still have daughters left re-set this task to be the first in the list
        //so that we can continue packing correctly
        pack_vars->top_task_list[0] = t;
        //Move all tasks forward in list so that the first next task will be packed to index 0
        //Move remaining cell indices so that their indexing starts from zero and ends in n_daughters_left
        for(int i = first_cell_to_move; i < n_daughters_left; i++){
          int shuffle = i - first_cell_to_move;
          ci_d[shuffle] = ci_d[i];
          cj_d[shuffle] = cj_d[i];
        }
        copy_index = 0;
        f_l_daughters[0][0] = 0;
        f_l_daughters[0][1] = n_daughters_left - first_cell_to_move;

        cj_top[0] = cj;
        ci_top[0] = ci;

        n_daughters_left -= first_cell_to_move;
        first_cell_to_move = 0;

        pack_vars->top_tasks_packed = 1;
        had_prev_task = 0;
      }
      pack_vars->tasks_packed = 0;
      pack_vars->count_parts = 0;
      pack_vars->launch = 0;
    }
    //case when we have launched then gone back to pack but did not pack enough to launch again
    else if(npacked == n_leaves_found){
      if(launched == 1){
        pack_vars->n_daughters_total = n_daughters_left;
        cj_top[0] = cj;
        ci_top[0] = ci;
        f_l_daughters[0][0] = 0;
        f_l_daughters[0][1] = n_daughters_left;
        pack_vars->top_tasks_packed = 1;
        pack_vars->top_task_list[0] = t;
        launched = 0;
      }
      else {
        f_l_daughters[top_tasks_packed - 1][0] = pack_vars->n_daughters_packed_index;
        f_l_daughters[top_tasks_packed - 1][1] = pack_vars->n_daughters_total;
      }
      pack_vars->launch = 0;
    }
    pack_vars->launch = 0;
  }
  //A. Nasar: Launch-leftovers counter re-set to zero and cells unlocked
  pack_vars->launch_leftovers = 0;
  pack_vars->launch = 0;

}

void runner_dopair_launch_f(
    struct runner *r, struct scheduler *s, struct gpu_pack_vars *pack_vars,
    struct task *t, struct part_aos_f4_send_f *parts_send,
    struct part_aos_f4_recv_f *parts_recv, struct part_aos_f4_send_f *d_parts_send,
    struct part_aos_f4_recv_f *d_parts_recv, cudaStream_t *stream, float d_a,
    float d_H, struct engine *e, int4 *fparti_fpartj_lparti_lpartj_forc,
    cudaEvent_t *pair_end) {

  /* Identify the number of GPU bundles to run in ideal case*/
  int n_bundles_temp = pack_vars->n_bundles;
  /*How many tasks have we packed?*/
  const int tasks_packed = pack_vars->tasks_packed;

  /*How many tasks should be in a bundle?*/
  const int bundle_size = pack_vars->bundle_size;

  /* Special case for incomplete bundles (when having leftover tasks not enough
   * to fill a bundle) */
  if (pack_vars->launch_leftovers) {
    n_bundles_temp = (tasks_packed + bundle_size - 1) / bundle_size;
    pack_vars->bundle_first_part[n_bundles_temp] =
        fparti_fpartj_lparti_lpartj_forc[tasks_packed - 1].x;
  }
  /* Identify the last particle for each bundle of tasks */
  for (int bid = 0; bid < n_bundles_temp - 1; bid++) {
    pack_vars->bundle_last_part[bid] = pack_vars->bundle_first_part[bid + 1];
  }
  /* special treatment for the case of 1 bundle */
  if (n_bundles_temp > 1)
    pack_vars->bundle_last_part[n_bundles_temp - 1] = pack_vars->count_parts;
  else
    pack_vars->bundle_last_part[0] = pack_vars->count_parts;

  /* Launch the copies for each bundle and run the GPU kernel */
  for (int bid = 0; bid < n_bundles_temp; bid++) {

    int max_parts_i = 0;
    int max_parts_j = 0;
    for (int tid = bid * bundle_size; tid < (bid + 1) * bundle_size; tid++) {
      if (tid < tasks_packed) {
        /*Get an estimate for the max number of parts per cell in each bundle.
         *  Used for determining the number of GPU CUDA blocks*/
        int count_i = fparti_fpartj_lparti_lpartj_forc[tid].z -
                      fparti_fpartj_lparti_lpartj_forc[tid].x;
        max_parts_i = max(max_parts_i, count_i);
        int count_j = fparti_fpartj_lparti_lpartj_forc[tid].w -
                      fparti_fpartj_lparti_lpartj_forc[tid].y;
        max_parts_j = max(max_parts_j, count_j);
      }
    }
    const int first_part_tmp_i = pack_vars->bundle_first_part[bid];
    const int bundle_n_parts =
        pack_vars->bundle_last_part[bid] - first_part_tmp_i;

    cudaMemcpyAsync(&d_parts_send[first_part_tmp_i],
                    &parts_send[first_part_tmp_i],
                    bundle_n_parts * sizeof(struct part_aos_f4_send_f),
                    cudaMemcpyHostToDevice, stream[bid]);

#ifdef CUDA_DEBUG
    cudaError_t cu_error =
        cudaPeekAtLastError();  // cudaGetLastError();        //
                                // Get error code
    if (cu_error != cudaSuccess) {
      error("CUDA error with pair density H2D async  memcpy ci: %s cpuid id is: %i\n"
            "Something's up with your cuda code first_part %i bundle size %i",
            cudaGetErrorString(cu_error), r->cpuid, first_part_tmp_i, bundle_n_parts);
    }
#endif
    /* LAUNCH THE GPU KERNELS for ci & cj */
    // Setup 2d grid of GPU thread blocks for ci (number of tasks is
    // the y dimension and max_parts is the x dimension
    int numBlocks_y = 0;  // tasks_left; //Changed this to 1D grid of blocks so
                          // this is no longer necessary
    int numBlocks_x = (bundle_n_parts + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int bundle_part_0 = pack_vars->bundle_first_part[bid];
    /* Launch the kernel for ci using data for ci and cj */
    runner_dopair_branch_force_gpu_aos_f4(
        d_parts_send, d_parts_recv, d_a, d_H, stream[bid], numBlocks_x,
        numBlocks_y, bundle_part_0, bundle_n_parts);

#ifdef CUDA_DEBUG
    cu_error = cudaPeekAtLastError();  // Get error code
    if (cu_error != cudaSuccess) {
      error("CUDA error with pair density kernel launch: %s cpuid id is: %i\n "
            "nbx %i nby %i max_parts_i %i max_parts_j %i\n"
            "Something's up with kernel launch.",
            cudaGetErrorString(cu_error), r->cpuid, numBlocks_x, numBlocks_y,
            max_parts_i, max_parts_j);
    }
#endif

    // Copy results back to CPU BUFFERS
    cudaMemcpyAsync(&parts_recv[first_part_tmp_i],
                    &d_parts_recv[first_part_tmp_i],
                    bundle_n_parts * sizeof(struct part_aos_f4_recv_f),
                    cudaMemcpyDeviceToHost, stream[bid]);
    // Issue event to be recorded by GPU after copy back to CPU
    cudaEventRecord(pair_end[bid], stream[bid]);

#ifdef CUDA_DEBUG
    cu_error = cudaPeekAtLastError();  // cudaGetLastError();        //
                                       // Get error code
    if (cu_error != cudaSuccess) {
      error("CUDA error with self density D2H memcpy: %s cpuid id is: %i\n"
            "Something's up with your cuda code",
            cudaGetErrorString(cu_error), r->cpuid);
    }
#endif
  } /*End of looping over bundles to launch in streams*/

  /* Issue synchronisation commands for all events recorded by GPU
   * Should swap with one cuda Device Synchronise really if we decide to go this
   * way with unpacking done separately */
  for (int bid = 0; bid < n_bundles_temp; bid++) {
    /*Time unpacking*/
    /* clock_gettime(CLOCK_REALTIME, &t0); */
    cudaEventSynchronize(pair_end[bid]);

    /* clock_gettime(CLOCK_REALTIME, &t1); */
    /* *gpu_time += */
    /*     (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1000000000.0; */
  }
} /*End of GPU work*/

void runner_dopair_unpack_f(
    struct runner *r, struct scheduler *s, struct gpu_pack_vars *pack_vars,
    struct task *t, struct part_aos_f4_send_f *parts_send,
    struct part_aos_f4_recv_f *parts_recv, struct part_aos_f4_send_f *d_parts_send,
    struct part_aos_f4_recv_f *d_parts_recv, cudaStream_t *stream, float d_a,
    float d_H, struct engine *e, int4 *fparti_fpartj_lparti_lpartj_forc,
    cudaEvent_t *pair_end, int npacked, int n_leaves_found, struct cell ** ci_d, struct cell ** cj_d, int ** f_l_daughters
    , struct cell ** ci_top, struct cell ** cj_top) {

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
    for (int tid = f_l_daughters[topid][0]; tid < f_l_daughters[topid][1]; tid++){
      /*Get pointers to the leaf cells*/
      struct cell *cii_l = ci_d[tid];
      struct cell *cjj_l = cj_d[tid];
      runner_do_ci_cj_gpu_unpack_neat_aos_f4_f(r, cii_l, cjj_l, parts_recv, 0,
                                             &pack_length_unpack, tid,
                                             2 * pack_vars->count_max_parts, e);
    }
    cell_unlocktree(ci_top[topid]);
    cell_unlocktree(cj_top[topid]);

    /*For some reason the code fails if we get a leaf pair task
     *this if->continue statement stops the code from trying to unlock same
     *cells twice*/
    if (topid == pack_vars->top_tasks_packed - 1 && npacked != n_leaves_found) {
      continue;
    }
    enqueue_dependencies(s, pack_vars->top_task_list[topid]);
    pthread_mutex_lock(&s->sleep_mutex);
    atomic_dec(&s->waiting);
    pthread_cond_broadcast(&s->sleep_cond);
    pthread_mutex_unlock(&s->sleep_mutex);
  }
}

void runner_gpu_pack_daughters_and_launch_f(struct runner *r, struct scheduler *s,
    struct cell * ci, struct cell * cj, struct gpu_pack_vars *pack_vars, struct
    task *t, struct part_aos_f4_send_f *parts_send,
    struct part_aos_f4_recv_f *parts_recv, struct part_aos_f4_send_f *d_parts_send,
    struct part_aos_f4_recv_f *d_parts_recv, cudaStream_t *stream, float d_a,
    float d_H, struct engine *e, int4 *fparti_fpartj_lparti_lpartj_forc,
    cudaEvent_t *pair_end, int n_leaves_found, struct cell ** ci_d, struct cell
    ** cj_d, int ** f_l_daughters , struct cell ** ci_top, struct cell ** cj_top){
  //Everything from here on needs moving to runner_doiact_functions_hydro_gpu.h
  pack_vars->n_daughters_total += n_leaves_found;
  int top_tasks_packed = pack_vars->top_tasks_packed;
  f_l_daughters[top_tasks_packed][0] = pack_vars->n_daughters_packed_index;
  ci_top[top_tasks_packed] = ci;
  cj_top[top_tasks_packed] = cj;
  pack_vars->n_leaves_total += n_leaves_found;
  int first_cell_to_move = pack_vars->n_daughters_packed_index;
  int n_daughters_left = pack_vars->n_daughters_total;
  /*Get pointer to top level task. Needed to enqueue deps*/
  pack_vars->top_task_list[top_tasks_packed] = t;
  /*Increment how many top tasks we've packed*/
  pack_vars->top_tasks_packed++;
  top_tasks_packed++;
  //A. Nasar: Remove this from struct as not needed. Was only used for de-bugging
  /*How many daughter tasks do we want to offload at once?*/
  size_t target_n_tasks_tmp = pack_vars->target_n_tasks;
  // A. Nasar: Check to see if this is the last task in the queue.
  // If so, set launch_leftovers to 1 and recursively pack and launch daughter tasks on GPU
  unsigned int qid = r->qid;
  lock_lock(&s->queues[qid].lock);
  s->queues[qid].n_packs_pair_left_f--;
  if (s->queues[qid].n_packs_pair_left_f < 1) pack_vars->launch_leftovers = 1;
  (void)lock_unlock(&s->queues[qid].lock);
  /*Counter for how many tasks we've packed*/
  int npacked = 0;
  int launched = 0;
  //A. Nasar: Loop through the daughter tasks we found
  int copy_index = pack_vars->n_daughters_packed_index;
  //not strictly true!!! Could be that we packed and moved on without launching
  int had_prev_task = 0;
  if(pack_vars->n_daughters_packed_index > 0)
    had_prev_task = 1;

  cell_unlocktree(ci);
  cell_unlocktree(cj);

  while(npacked < n_leaves_found){
    top_tasks_packed = pack_vars->top_tasks_packed;
    struct cell * cii = ci_d[copy_index];
    struct cell * cjj = cj_d[copy_index];
    runner_dopair1_pack_f4_ff(
        r, s, pack_vars, cii, cjj, t,
        parts_send, e, fparti_fpartj_lparti_lpartj_forc);
    //record number of tasks we've copied from last launch
    copy_index++;
    //Record how many daughters we've packed in total for while loop
    npacked++;
    first_cell_to_move++;
    /*Re-think this. Probably not necessary to have this if statement and we can just use the form in the else statement*/
    if(had_prev_task)
      f_l_daughters[top_tasks_packed - 1][1] = f_l_daughters[top_tasks_packed - 1][0] + copy_index - pack_vars->n_daughters_packed_index;
    else
      f_l_daughters[top_tasks_packed - 1][1] = f_l_daughters[top_tasks_packed - 1][0] + copy_index;

    if(pack_vars->tasks_packed == target_n_tasks_tmp)
        pack_vars->launch = 1;
    if(pack_vars->launch || (pack_vars->launch_leftovers && npacked == n_leaves_found)){

      launched = 1;
      //Here we only launch the tasks. No unpacking! This is done in next function ;)
      runner_dopair_launch_f(
            r, s, pack_vars, t, parts_send,
            parts_recv, d_parts_send,
            d_parts_recv, stream, d_a, d_H, e,
            fparti_fpartj_lparti_lpartj_forc,
            pair_end);
      runner_dopair_unpack_f(
            r, s, pack_vars, t, parts_send,
            parts_recv, d_parts_send,
            d_parts_recv, stream, d_a, d_H, e,
            fparti_fpartj_lparti_lpartj_forc,
            pair_end, npacked, n_leaves_found, ci_d, cj_d, f_l_daughters, ci_top, cj_top);
      if(npacked == n_leaves_found){
        pack_vars->n_daughters_total = 0;
        pack_vars->top_tasks_packed = 0;
      }
      //Special treatment required here. Launched but have not packed all tasks
      else{
        //If we launch but still have daughters left re-set this task to be the first in the list
        //so that we can continue packing correctly
        pack_vars->top_task_list[0] = t;
        //Move all tasks forward in list so that the first next task will be packed to index 0
        //Move remaining cell indices so that their indexing starts from zero and ends in n_daughters_left
        for(int i = first_cell_to_move; i < n_daughters_left; i++){
          int shuffle = i - first_cell_to_move;
          ci_d[shuffle] = ci_d[i];
          cj_d[shuffle] = cj_d[i];
        }
        copy_index = 0;
        f_l_daughters[0][0] = 0;
        f_l_daughters[0][1] = n_daughters_left - first_cell_to_move;

        cj_top[0] = cj;
        ci_top[0] = ci;

        n_daughters_left -= first_cell_to_move;
        first_cell_to_move = 0;

        pack_vars->top_tasks_packed = 1;
        had_prev_task = 0;
      }
      pack_vars->tasks_packed = 0;
      pack_vars->count_parts = 0;
      pack_vars->launch = 0;
    }
    //case when we have launched then gone back to pack but did not pack enough to launch again
    else if(npacked == n_leaves_found){
      if(launched == 1){
        pack_vars->n_daughters_total = n_daughters_left;
        cj_top[0] = cj;
        ci_top[0] = ci;
        f_l_daughters[0][0] = 0;
        f_l_daughters[0][1] = n_daughters_left;
        pack_vars->top_tasks_packed = 1;
        pack_vars->top_task_list[0] = t;
        launched = 0;
      }
      else {
        f_l_daughters[top_tasks_packed - 1][0] = pack_vars->n_daughters_packed_index;
        f_l_daughters[top_tasks_packed - 1][1] = pack_vars->n_daughters_total;
      }
      pack_vars->launch = 0;
    }
    pack_vars->launch = 0;
  }
  //A. Nasar: Launch-leftovers counter re-set to zero and cells unlocked
  pack_vars->launch_leftovers = 0;
  pack_vars->launch = 0;

}

#ifdef __cplusplus
}
#endif
