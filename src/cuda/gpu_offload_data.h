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
#include "gpu_pack_metadata.h"
#include "gpu_pack_params.h"
#include "gpu_part_structs.h"

#include <cuda_runtime.h>

/* Rule-of-thumb: Everything related to actual particle data and everything
 * CUDA-specific goes in here. Everything else goes into gpu_pack_metadata
 * struct.*/

/*! Struct to hold all data for the transfer of a single task (sub)type. */
struct gpu_offload_data {
#ifdef WITH_CUDA

  /*! bookkeeping meta-data for offloading */
  struct gpu_pack_metadata md;

  /*! First and last particles for self interactions */
  /* TODO: This should be cuda-independent and moved into gpu_pack_metadata */
  int2 *self_task_first_last_part;
  int2 *d_self_task_first_last_part;

#ifdef SWIFT_DEBUG_CHECKS
  /*! Keep track of allocated array size for boundary checks. */
  int self_task_first_last_part_size;
  int d_self_task_first_last_part_size;
#endif

  /*! First and last particles of cells i and j for pair interactions */
  int4 *fparti_fpartj_lparti_lpartj;
#ifdef SWIFT_DEBUG_CHECKS
  /*! Keep track of allocated array size for boundary checks. */
  int fparti_fpartj_lparti_lpartj_size;
#endif

  /*! Arrays used to send particle data from device to host. A single struct
   * gpu_offload_data will only hold data for either density, gradient, or
   * force task, so we hide them behind a union. */
  union {
    struct gpu_part_send_d *d_parts_send_d;
    struct gpu_part_send_g *d_parts_send_g;
    struct gpu_part_send_f *d_parts_send_f;
  };

#ifdef SWIFT_DEBUG_CHECKS
  /*! Keep track of allocated array size for boundary checks. */
  int d_parts_send_size;
#endif

  /*! Array used to receive particle data on device from host */
  union {
    struct gpu_part_recv_d *d_parts_recv_d;
    struct gpu_part_recv_g *d_parts_recv_g;
    struct gpu_part_recv_f *d_parts_recv_f;
  };

#ifdef SWIFT_DEBUG_CHECKS
  /*! Keep track of allocated array size for boundary checks. */
  int d_parts_recv_size;
#endif

  /*! Array used to send particle data from host to device */
  union {
    struct gpu_part_send_d *parts_send_d;
    struct gpu_part_send_g *parts_send_g;
    struct gpu_part_send_f *parts_send_f;
  };

#ifdef SWIFT_DEBUG_CHECKS
  /*! Keep track of allocated array size for boundary checks. */
  int parts_send_size;
#endif

  /*! Array used to receive particle data from device on host */
  union {
    struct gpu_part_recv_d *parts_recv_d;
    struct gpu_part_recv_g *parts_recv_g;
    struct gpu_part_recv_f *parts_recv_f;
  };

#ifdef SWIFT_DEBUG_CHECKS
  /*! Keep track of allocated array size for boundary checks. */
  int parts_recv_size;
#endif

  /*! Handle on events per cuda stream to register completion of async ops */
  cudaEvent_t *event_end;

#endif /* WITH_CUDA */
};

void gpu_init_data_buffers(struct gpu_offload_data *buf,
                           const struct gpu_global_pack_params *params,
                           const size_t send_struct_size,
                           const size_t recv_struct_size,
                           const char is_pair_task);

void gpu_init_data_buffers_step(struct gpu_offload_data *buf);

void gpu_free_data_buffers(struct gpu_offload_data *buf,
                           const char is_pair_task);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_GPU_OFFLOAD_DATA_H */
