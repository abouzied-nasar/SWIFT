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

/**
 * @file src/cuda/GPU_utils.c
 * @brief misc GPU utilities
 */

#include "gpu_utils.h"

#include "cuda_config.h"
#include "runner.h"

#include <cuda.h>
#include <cuda_runtime.h>

/**
 * @brief Initialize the GPU context for each thread. This should be
 * called in a threaded region, e.g. runner_main_cuda.
 */
void gpu_init_thread(const struct engine* e, const int cpuid) {

  /* Find and print GPU name(s) */
  int dev_id = 0; /* gpu device name */
  struct cudaDeviceProp prop;
  int n_devices;
//  int max_blocks_SM;
//  int n_SMs;

  cudaGetDeviceCount(&n_devices);
  /* A. Nasar: If running on MPI we set code to use one MPI rank per GPU
   * This was found to work very well and simplifies writing slurm scipts */
  if (n_devices == 1) {
    cudaSetDevice(dev_id);
  }
#ifdef WITH_MPI
  else {
    cudaSetDevice(engine_rank);
    dev_id = engine_rank;
  }
#endif

  cudaFree(0);
  /* Now tell me some info about my device */
//  cudaGetDeviceProperties(&prop, dev_id);
//  cudaDeviceGetAttribute(&max_blocks_SM, cudaDevAttrMaxBlocksPerMultiprocessor,
//                         dev_id);
//  cudaDeviceGetAttribute(&n_SMs, cudaDevAttrMultiProcessorCount, dev_id);
//  char uuid_str[40];
//  cudaUUID_t uuid = prop.uuid;
//
//
//  snprintf(uuid_str, sizeof(uuid_str),
//		  "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
//		  uuid.bytes[0], uuid.bytes[1], uuid.bytes[2], uuid.bytes[3],
//		  uuid.bytes[4], uuid.bytes[5], uuid.bytes[6], uuid.bytes[7],
//		  uuid.bytes[8], uuid.bytes[9], uuid.bytes[10], uuid.bytes[11],
//		  uuid.bytes[12], uuid.bytes[13], uuid.bytes[14], uuid.bytes[15]);
//
//  size_t free_mem;
//  size_t total_mem;
//  const struct space* space = e->s;
//  cudaMemGetInfo(&free_mem, &total_mem);
//  int nPartsPerCell = space->nr_parts / space->tot_cells;
//  if (cpuid == 0 && engine_rank == 0) {
//    message("Devices available:          %i", n_devices);
//    message("Device id:                  %i", dev_id);
//    message("Device UUID:                %s", uuid_str);
//    message("Device name:                %s", prop.name);
//    message("n_SMs:                      %i", n_SMs);
//    message("max blocks per SM:          %i", max_blocks_SM);
//    message("max blocks per stream:      %i", n_SMs * max_blocks_SM);
//    message("Target n_blocks per kernel: %i",
//            N_TASKS_BUNDLE_SELF * nPartsPerCell / BLOCK_SIZE);
//    message("Target n_blocks per stream: %i",
//            N_TASKS_PER_PACK_SELF * nPartsPerCell / BLOCK_SIZE);
//    message("free mem:                   %.3g GB",
//            ((double)free_mem) / (1024. * 1024. * 1024.));
//    message("total mem:                  %.3g GB",
//            ((double)total_mem) / (1024. * 1024. * 1024.));
//  }
}

/**
 * @brief Initialize the GPU context for each thread. This should be
 * called in a threaded region, e.g. runner_main_cuda.
 */
void gpu_print_free_mem(const struct engine* e, const int cpuid) {

  /* Find and print GPU name(s) */
  int dev_id = engine_rank; /* gpu device name */
  struct cudaDeviceProp prop;

  /* Now tell me some info about my device */
  cudaGetDeviceProperties(&prop, dev_id);

  size_t free_mem;
  size_t total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  if (cpuid == 0) {
    message("After allocation: free mem: %.3g GB, total mem: %.3g GB",
            ((double)free_mem) / (1024. * 1024. * 1024.),
            ((double)total_mem) / (1024. * 1024. * 1024.));
  }
}
