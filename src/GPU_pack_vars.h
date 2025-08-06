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

  /*! List of tasks and respective cells to be packed */
  struct task **task_list;

  /*! TODO: documentation */
  struct task **top_task_list;

  /*! TODO: documentation */
  struct cell **ci_list;

  /*! TODO: documentation */
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

  /*! TODO: documentation */
  size_t bundle_size;

  /*! How many particles in a bundle*/
  size_t count_parts;
  /**/
  size_t tasks_packed;

  /*! TODO: documentation */
  size_t top_tasks_packed;

  /*! TODO: documentation */
  int *task_first_part;

  /*! TODO: documentation */
  int *task_last_part;

  /*! TODO: documentation */
  int *d_task_first_part;

  /*! TODO: documentation */
  int *d_task_last_part;

  /*! TODO: documentation */
  int *bundle_first_part;

  /*! TODO: documentation */
  int *bundle_last_part;

  /*! TODO: documentation */
  int *bundle_first_task_list;

  /*! TODO: documentation */
  size_t count_max_parts;

  /*! TODO: documentation */
  char launch;

  /*! TODO: documentation */
  char launch_leftovers;

  /*! TODO: documentation */
  size_t target_n_tasks;

  /*! TODO: documentation */
  size_t n_bundles;

  /*! TODO: documentation */
  size_t tasksperbundle;

  /*! TODO: documentation */
  size_t n_daughters_total;

  /*! TODO: documentation */
  size_t n_daughters_packed_index;

  /*! TODO: documentation */
  size_t n_leaves_found;

  /*! TODO: documentation */
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

#ifdef __cplusplus
}
#endif

#endif
