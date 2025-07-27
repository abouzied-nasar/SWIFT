#ifndef GPU_PACK_VARS_H
#define GPU_PACK_VARS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <vector_types.h>

/**
 * TODO: documentation
 */
static struct pack_vars_self {
  /*List of tasks and respective cells to be packed*/
  struct task **task_list;
  struct task **top_task_list;
  struct cell **cell_list;
  /*List of cell positions*/
  double *cellx;
  double *celly;
  double *cellz;
  /*List of cell positions*/
  double *d_cellx;
  double *d_celly;
  double *d_cellz;
  int bundle_size;
  /*How many particles in a bundle*/
  int count_parts;
  /**/
  int tasks_packed;
  int top_tasks_packed;
  int *task_first_part;
  int *task_last_part;
  int *d_task_first_part;
  int *d_task_last_part;
  int *bundle_first_part;
  int *bundle_last_part;
  int *bundle_first_task_list;
  int count_max_parts;
  int launch;
  int launch_leftovers;
  int target_n_tasks;
  int n_bundles;
  int tasksperbundle;

} pack_vars_self;

/**
 * TODO: documentation
 */
static struct pack_vars_pair {
  /*List of tasks and respective cells to be packed*/
  struct task **task_list;
  struct task **top_task_list;
  struct cell **ci_list;
  struct cell **cj_list;
  /*List of cell shifts*/
  double *shiftx;
  double *shifty;
  double *shiftz;
  /*List of cell shifts*/
  double *d_shiftx;
  double *d_shifty;
  double *d_shiftz;
  int bundle_size;
  /*How many particles in a bundle*/
  int count_parts;
  /**/
  int tasks_packed;
  int top_tasks_packed;
  int *task_first_part;
  int *task_last_part;
  int *d_task_first_part;
  int *d_task_last_part;
  int *bundle_first_part;
  int *bundle_last_part;
  int *bundle_first_task_list;
  int count_max_parts;
  int launch;
  int launch_leftovers;
  int target_n_tasks;
  int n_bundles;
  int tasksperbundle;
  int n_daughters_total;
  int n_daughters_packed_index;
  int n_leaves_found;
  int n_leaves_total;

} pack_vars_pair;

/**
 * TODO: documentation
 */
static struct pack_vars_pair_f4 {
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

} pack_vars_pair_f4;

#ifdef __cplusplus
}
#endif

#endif
