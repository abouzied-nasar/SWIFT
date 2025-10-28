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

  /*! Arrays used to send particle data from device to host. A single struct
   * gpu_offload_data will only hold data for either density, gradient, or
   * force task, so we hide them behind a union. */
  union {
    struct gpu_part_send_d *d_parts_send_d;
    struct gpu_part_send_g *d_parts_send_g;
    struct gpu_part_send_f *d_parts_send_f;
  };

  /*! Array used to receive particle data on device from host */
  union {
    struct gpu_part_recv_d *d_parts_recv_d;
    struct gpu_part_recv_g *d_parts_recv_g;
    struct gpu_part_recv_f *d_parts_recv_f;
  };

  /*! Array used to send particle data from host to device */
  union {
    struct gpu_part_send_d *parts_send_d;
    struct gpu_part_send_g *parts_send_g;
    struct gpu_part_send_f *parts_send_f;
  };

  /*! Array used to receive particle data from device on host */
  union {
    struct gpu_part_recv_d *parts_recv_d;
    struct gpu_part_recv_g *parts_recv_g;
    struct gpu_part_recv_f *parts_recv_f;
  };

  /*! Arrays used to receive positions of first/last particle in cells in the particle arrays above
   * from device to host. cell_i_j_start_end.x is ci_start, y is ci_end, z is cj_start, w is cj_end*/
  int4 *cell_i_j_start_end;

  /*! Arrays used to send positions of first particle in cells in the particle arrays above
   * from host to device*/
  int4 *d_cell_i_j_start_end;

  /*! Handle on events per cuda stream to register completion of async ops */
  cudaEvent_t *event_end;

#endif /* WITH_CUDA */
};

void gpu_data_buffers_init(struct gpu_offload_data *buf,
                           const struct gpu_global_pack_params *params,
                           const size_t send_struct_size,
                           const size_t recv_struct_size,
                           const char is_pair_task);
void gpu_data_buffers_init_step(struct gpu_offload_data *buf);
void gpu_data_buffers_reset(struct gpu_offload_data *buf);
void gpu_data_buffers_free(struct gpu_offload_data *buf);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_GPU_OFFLOAD_DATA_H */
