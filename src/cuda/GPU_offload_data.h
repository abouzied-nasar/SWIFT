#ifndef CUDA_gpu_offload_data_H
#define CUDA_gpu_offload_data_H

#ifdef __cplusplus
extern "C" {
#endif

#include "GPU_pack_vars.h"
#include "GPU_part_structs.h"
#include "cell.h"

#include <stddef.h>

/* TODO: remove again? */
#include <cuda.h>
#include <cuda_runtime.h>

#include "error.h"

/*! Struct to hold all data for the transfer of a single task (sub)type */
struct gpu_offload_data{

  /*! data required for self and pair packing tasks destined for the GPU*/
  struct gpu_pack_vars pv;

  /*! First and last particles for self interactions */
  int2 *task_first_part_f4;
  int2 *d_task_first_part_f4;

  /*! First and last particles of cells i and j for pair interactions */
  int4 *fparti_fpartj_lparti_lpartj;

  /*! TODO: Documentation?? */
  union {
    struct part_aos_f4_send_d *d_send_d;
    struct part_aos_f4_send_g *d_send_g;
    struct part_aos_f4_send_f *d_send_f;
  };

  /*! TODO: Documentation?? */
  union {
    struct part_aos_f4_recv_d *d_recv_d;
    struct part_aos_f4_recv_g *d_recv_g;
    struct part_aos_f4_recv_f *d_recv_f;
  };

  /*! TODO: Documentation?? */
  union {
    struct part_aos_f4_send_d *send_d;
    struct part_aos_f4_send_g *send_g;
    struct part_aos_f4_send_f *send_f;
  };

  /*! TODO: Documentation?? */
  union {
    struct part_aos_f4_recv_d *recv_d;
    struct part_aos_f4_recv_g *recv_g;
    struct part_aos_f4_recv_f *recv_f;
  };

  /*! TODO: Documentation */
  struct cell **ci_d;
  struct cell **cj_d;

  /*! TODO: Documentation */
  int **first_and_last_daughters;

  /*! TODO: Documentation */
  struct cell **ci_top;
  struct cell **cj_top;

  /* cudaStream_t *stream; */
  cudaEvent_t* event_end;
};


void gpu_init_data_buffers(
    struct gpu_offload_data *buf,
    const size_t target_n_tasks,
    const size_t bundle_size,
    const size_t n_bundles,
    const size_t count_max_parts_tmp,
    const size_t send_struct_size,
    const size_t recv_struct_size,
    const char is_pair_task);


void gpu_init_data_buffers_step(struct gpu_offload_data *buf);


#ifdef __cplusplus
}
#endif

#endif
