#ifdef __cplusplus
extern "C" {
#endif

#include "GPU_data_buffers.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "error.h"



/**
 * @brief initialise GPU data buffers.
 *
 * @brief TODO: parameter docu
 *
 * @param bundle_size defines the number of leaf-level tasks we will
 *    compute in each stream
 * @param is_pair_task: Whether we allocate enough space for pair tasks
 * @param send_struct_size: size of struct used for send arrays (both host and device)
 * @param recv_struct_size: size of struct used for recv arrays (both host and device)
 */
void gpu_init_data_buffers(
    struct gpu_data_buffers *buf,
    const size_t target_n_tasks,
    const size_t bundle_size,
    const size_t n_bundles, /* differs for pair or self */
    const size_t count_max_parts_tmp,
    const size_t send_struct_size,
    const size_t recv_struct_size,
    const char is_pair_task
    ) {

  /* A. Nasar: n_bundles is the number of task bundles each
  thread has ==> Used to loop through bundles */
  /* const size_t n_bundles = (target_n_tasks + bundle_size - 1) / bundle_size; */

  /* A. Nasar: target_n_tasks defines the total number of leaf-level tasks we will
   * compute for each GPU off-load cycle */
  const size_t tasksperbundle = (target_n_tasks + n_bundles - 1) / n_bundles;
  /* Multiplication factor depending on whether this is for a self or a pair task */
  const size_t self_pair_fact = is_pair_task ? 2 : 1;

  /* Initialise and set up pack_vars */
  struct gpu_pack_vars* pv = &buf->pv;
  gpu_init_pack_vars(pv);

  cudaError_t cu_error = cudaErrorMemoryAllocation;
  cu_error = cudaMallocHost((void **)&pv, sizeof(struct gpu_pack_vars));
  swift_assert(cu_error == cudaSuccess);

  pv->target_n_tasks = target_n_tasks;
  pv->bundle_size = bundle_size;
  pv->n_bundles = n_bundles;

  cu_error=cudaMallocHost((void **)&pv->bundle_first_part, self_pair_fact * n_bundles * sizeof(int));
  swift_assert(cu_error == cudaSuccess);

  cu_error=cudaMallocHost((void **)&pv->bundle_last_part, self_pair_fact * n_bundles * sizeof(int));
  swift_assert(cu_error == cudaSuccess);

  cu_error=cudaMallocHost((void **)&pv->bundle_first_task_list, self_pair_fact * n_bundles * sizeof(int));
  swift_assert(cu_error == cudaSuccess);

  pv->tasksperbundle = tasksperbundle;
  pv->count_parts = 0;
  pv->count_max_parts = count_max_parts_tmp;

  pv->task_list = (struct task **)calloc(target_n_tasks, sizeof(struct task *));
  pv->ci_list = (struct cell **)calloc(target_n_tasks, sizeof(struct cell *));

  /* A. Nasar: Keep track of first and last particles for each self task (particle data is
   * arranged in long arrays containing particles from all the tasks we will
   * work with)
   * Needed for offloading self tasks as we use these to sort through which
   * parts need to interact with which */
  if (is_pair_task){
    buf->task_first_part_f4 = NULL;
    buf->d_task_first_part_f4 = NULL;
  } else {
    cu_error = cudaMallocHost((void **)&buf->task_first_part_f4, target_n_tasks * sizeof(int2));
    swift_assert(cu_error == cudaSuccess);
    cu_error = cudaMalloc((void **)&buf->d_task_first_part_f4, target_n_tasks * sizeof(int2));
    swift_assert(cu_error == cudaSuccess);
  }

  /* Get array of first and last particles for pair interactions. */
  /*A. N.: Needed but only for small part in launch functions. Might
           be useful for recursion on the GPU so keep for now     */
  buf->fparti_fpartj_lparti_lpartj = NULL;
  if (is_pair_task){
    cu_error = cudaMallocHost((void **)&buf->fparti_fpartj_lparti_lpartj,
		  target_n_tasks * sizeof(int4));
    swift_assert(cu_error == cudaSuccess);
  }

  /* Now allocate memory for Buffer and GPU particle arrays */
  cu_error = cudaMalloc((void **)&buf->d_send_d, self_pair_fact * count_max_parts_tmp * send_struct_size);
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaMalloc((void **)&buf->d_recv_d, self_pair_fact * count_max_parts_tmp * recv_struct_size);
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaMallocHost((void **)&buf->send_d,
                 self_pair_fact * count_max_parts_tmp * send_struct_size);
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaMallocHost((void **)&buf->recv_d,
                 self_pair_fact * count_max_parts_tmp * recv_struct_size);
  swift_assert(cu_error == cudaSuccess);


  /* Create streams so that we can off-load different batches of work in
   * different streams and get some con-CURRENCY! Events used to maximise
   * asynchrony further*/
  /* TODO: remove this? */
  /* buf->stream = (cudaStream_t*)malloc(n_bundles * sizeof(cudaStream_t)); */
  /* TODO: Don't do this here? */
  buf->event_end = (cudaEvent_t*)malloc(n_bundles * sizeof(cudaEvent_t));

  for (size_t i = 0; i < n_bundles; i++){
    cudaEventCreate(&buf->event_end[i]);
  }

}


/**
 * @brief perform the initialisations required at the start of each step
 */
void gpu_init_data_buffers_step(struct gpu_data_buffers *buf){

  struct gpu_pack_vars* pv = &buf->pv;

  // Initialise packing counters
  pv->tasks_packed = 0;
  pv->count_parts = 0;
  pv->top_tasks_packed = 0; /* Needed only for pair tasks? */
  pv->n_daughters_total = 0; /* Needed only for pair tasks? */
}

#ifdef __cplusplus
}
#endif
