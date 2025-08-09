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
#ifndef CUDA_GPU_OFFLOAD_DATA_H
#define CUDA_GPU_OFFLOAD_DATA_H


/**
 * @file cuda/gpu_offload_data.h
 * @brief contains the gpu_offload_data struct, which holds data required for
 * offloading, and associated functions
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "cell.h"
#include "gpu_pack_vars.h"
#include "gpu_part_structs.h"

#include <cuda_runtime.h>



/*! Struct to hold all data for the transfer of a single task (sub)type */
struct gpu_offload_data{
#ifdef WITH_CUDA

  /*! data required for self and pair packing tasks destined for the GPU*/
  struct gpu_pack_vars pv;

  /*! First and last particles for self interactions */
  int2 *task_first_part_f4;
  int2 *d_task_first_part_f4;

  /*! First and last particles of cells i and j for pair interactions */
  int4 *fparti_fpartj_lparti_lpartj;

  /*! Arrays used to send particle data on device. A single struct
   * gpu_offload_data will only hold data for either density, gradient, or
   * force task, so we hide them behind a union. We add another one, a pointer
   * to void, which can be used to pass the correct address to cudaMemcpy* functions
   * without discriminating which type of struct it is. */
  union {
    void* d_parts_send;
    struct gpu_part_send_d *d_parts_send_d;
    struct gpu_part_send_g *d_parts_send_g;
    struct gpu_part_send_f *d_parts_send_f;
  };

  /*! TODO: Documentation?? */
  union {
    void* d_parts_recv;
    struct gpu_part_recv_d *d_parts_recv_d;
    struct gpu_part_recv_g *d_parts_recv_g;
    struct gpu_part_recv_f *d_parts_recv_f;
  };

  /*! TODO: Documentation?? */
  union {
    void *parts_send;
    struct gpu_part_send_d *parts_send_d;
    struct gpu_part_send_g *parts_send_g;
    struct gpu_part_send_f *parts_send_f;
  };

  /*! TODO: Documentation?? */
  union {
    void *parts_recv;
    struct gpu_part_recv_d *parts_recv_d;
    struct gpu_part_recv_g *parts_recv_g;
    struct gpu_part_recv_f *parts_recv_f;
  };

  /*! TODO: Documentation */
  struct cell **ci_d;
  struct cell **cj_d;

  /*! TODO: Documentation */
  int **first_and_last_daughters;

  /*! TODO: Documentation */
  struct cell **ci_top;
  struct cell **cj_top;

  /*! TODO: Documentation */
  cudaEvent_t* event_end;

  /*! Size of the struct used to send data to/from device. */
  size_t size_of_send_struct;

  /*! Size of the struct used to send data to/from device. */
  size_t size_of_recv_struct;

#endif /* WITH_CUDA */
};



void gpu_init_data_buffers(
    struct gpu_offload_data *buf,
    const struct gpu_global_pack_params* params,
    const size_t send_struct_size,
    const size_t recv_struct_size,
    const char is_pair_task);


void gpu_init_data_buffers_step(struct gpu_offload_data *buf);


#ifdef __cplusplus
}
#endif

#endif /* CUDA_GPU_OFFLOAD_DATA_H */
