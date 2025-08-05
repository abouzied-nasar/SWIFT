#ifndef GPU_PACK_VARS_H
#define GPU_PACK_VARS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <vector_types.h>

/**
 * TODO Abouzeid: documentation
 */
struct gpu_pack_vars {

  /* List of tasks and respective cells to be packed */
  struct task **task_list;
  struct task **top_task_list;
  struct cell **ci_list;
  struct cell **cj_list;
  /* List of cell shifts and positions. Shifts are used for pair tasks,
   * while cell positions are used for self tasks.*/
  union {
    double *cellx;
    double *shiftx;
  };
  union {
    double *celly;
    double *shifty;
  };
  union {
    double *cellz;
    double *shiftz;
  };
  /* List of cell shifts and positions ON DEVICE. Shifts are used for
   * pair tasks, while cell positions are used for self tasks.*/
  union {
    double *d_cellx;
    double *d_shiftx;
  };
  union {
    double *d_celly;
    double *d_shifty;
  };
  union {
    double *d_cellz;
    double *d_shiftz;
  };
  size_t bundle_size;
  /*How many particles in a bundle*/
  /* TODO: make size_t?*/
  size_t count_parts;
  /**/
  size_t tasks_packed;
  size_t top_tasks_packed;
  int *task_first_part;
  int *task_last_part;
  int *d_task_first_part;
  int *d_task_last_part;
  int *bundle_first_part;
  int *bundle_last_part;
  int *bundle_first_task_list;
  size_t count_max_parts;
  char launch;
  char launch_leftovers;
  size_t target_n_tasks;
  size_t n_bundles;
  size_t tasksperbundle;
  size_t n_daughters_total;
  size_t n_daughters_packed_index;
  size_t n_leaves_found;
  size_t n_leaves_total;

};

/**
 * Initialise empty gpu_pack_vars struct
 *
 * TODO: This doesn't need to be inlined or in the header.
 */
__attribute__((always_inline)) inline void gpu_init_pack_vars(struct gpu_pack_vars* pv){

  pv->task_list = NULL;
  pv->top_task_list = NULL;
  pv->ci_list = NULL;
  pv->cj_list = NULL;
  pv->cellx = NULL;
  pv->celly = NULL;
  pv->cellz = NULL;
  pv->d_cellx = NULL;
  pv->d_celly = NULL;
  pv->d_cellz = NULL;

  pv->bundle_size = 0;
  pv->count_parts = 0;
  pv->tasks_packed = 0;

  pv->task_first_part = NULL;
  pv->task_last_part = NULL;
  pv->d_task_first_part = NULL;
  pv->d_task_last_part = NULL;
  pv->bundle_first_part = NULL;
  pv->bundle_last_part = NULL;
  pv->bundle_first_task_list = NULL;

  pv->count_max_parts = 0;
  pv->launch = 0;
  pv->launch_leftovers = 0;

  pv->target_n_tasks = 0;
  pv->n_bundles = 0;
  pv->tasksperbundle = 0;
  pv->n_daughters_total = 0;
  pv->n_daughters_packed_index = 0;
  pv->n_leaves_found = 0;
  pv->n_leaves_total = 0;
}

/**
 * TODO Abouzeid: documentation
 */
struct pack_vars_pair_f4 {
  /*List of tasks and respective cells to be packed*/
  struct task **task_list;
  struct cell **ci_list;
  struct cell **cj_list;
  /*List of cell shifts*/
  float3 *shift;
  /*List of cell shifts*/
  float3 *d_shift;
  int bundle_size;
  /*How many particles in a bundle*/
  int count_parts;
  /**/
  int tasks_packed;
  int4 *fparti_fpartj_lparti_lpartj;
  int4 *d_fparti_fpartj_lparti_lpartj;
  int *bundle_first_part;
  int *bundle_last_part;
  int *bundle_first_task_list;
  int count_max_parts;
  int launch;
  int launch_leftovers;
  int target_n_tasks;
  int nBundles;
  int tasksperbundle;

};

#ifdef __cplusplus
}
#endif

#endif
