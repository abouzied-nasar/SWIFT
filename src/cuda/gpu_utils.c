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
#include "gpu_pack_params.h"
#include "runner.h"

#include <cuda.h>
#include <cuda_runtime.h>

/**
 * @brief Initialize the GPU context for each thread. This should be
 * called in a threaded region, e.g. runner_main_cuda.
 */
void gpu_init_thread(const struct engine* e, const int cpuid) {

  const struct gpu_global_pack_params* gpu_pack_params = &e->gpu_pack_params;

  /* Find and print GPU name(s) */
  int dev_id = 0; /* gpu device name */
  struct cudaDeviceProp prop;
  int n_devices;
  int max_blocks_SM;
  int n_SMs;

  cudaError_t cu_error = cudaGetDeviceCount(&n_devices);
  swift_assert(cu_error == cudaSuccess);

  /* A. Nasar: If running on MPI we set code to use one MPI rank per GPU
   * This was found to work very well and simplifies writing slurm scipts */
  if (n_devices == 1) {
    cu_error = cudaSetDevice(dev_id);
    swift_assert(cu_error == cudaSuccess);
  }
#ifdef WITH_MPI
  else {
    cu_error = cudaSetDevice(engine_rank);
    swift_assert(cu_error == cudaSuccess);

    dev_id = engine_rank;
  }
#endif

  /* Now tell me some info about my device */
  cu_error = cudaGetDeviceProperties(&prop, dev_id);
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaDeviceGetAttribute(&max_blocks_SM, cudaDevAttrMaxBlocksPerMultiprocessor, dev_id);
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaDeviceGetAttribute(&n_SMs, cudaDevAttrMultiProcessorCount, dev_id);
  swift_assert(cu_error == cudaSuccess);

  size_t free_mem;
  size_t total_mem;
  const struct space* space = e->s;
  cu_error = cudaMemGetInfo(&free_mem, &total_mem);
  swift_assert(cu_error == cudaSuccess);

  int nPartsPerCell = space->nr_parts / space->tot_cells;
  if (cpuid == 0 && engine_rank == 0) {
    message("Devices available:          %i", n_devices);
    message("Device id:                  %i", dev_id);
    message("Device name:                %s", prop.name);
    message("n_SMs:                      %i", n_SMs);
    message("max blocks per SM:          %i", max_blocks_SM);
    message("max blocks per stream:      %i", n_SMs * max_blocks_SM);
    message(
        "Target n_blocks per kernel: %ld",
        gpu_pack_params->bundle_size * nPartsPerCell / GPU_THREAD_BLOCK_SIZE);
    message("Target n_blocks per stream: %ld",
            gpu_pack_params->pack_size * nPartsPerCell / GPU_THREAD_BLOCK_SIZE);
    message("free mem:                   %.3g GB",
            ((double)free_mem) / (1024. * 1024. * 1024.));
    message("total mem:                  %.3g GB",
            ((double)total_mem) / (1024. * 1024. * 1024.));
  }
}

/**
 * @brief Initialize the GPU context for each thread. This should be
 * called in a threaded region, e.g. runner_main_cuda.
 */
void gpu_print_free_mem(const struct engine* e, const int cpuid) {

  /* Find and print GPU name(s) */
  int dev_id = 0;
  int n_devices;
  cudaError_t cu_error = cudaGetDeviceCount(&n_devices);
  swift_assert(cu_error == cudaSuccess);

#ifdef WITH_MPI
  if (n_devices != 1) {
    dev_id = engine_rank;
  }
#endif

  struct cudaDeviceProp prop;

  /* Now tell me some info about my device */
  cu_error = cudaGetDeviceProperties(&prop, dev_id);
  swift_assert(cu_error == cudaSuccess);

  size_t free_mem;
  size_t total_mem;
  cu_error = cudaMemGetInfo(&free_mem, &total_mem);
  swift_assert(cu_error == cudaSuccess);

  if (cpuid == 0) {
#ifdef SWIFT_DEBUG_CHECKS
    message(
        "pciBusID %4d, After allocation: free mem: %8.3g GB, total mem: %8.3g "
        "GB",
        prop.pciBusID, ((double)free_mem) / (1024. * 1024. * 1024.),
        ((double)total_mem) / (1024. * 1024. * 1024.));
#else
    message("After allocation: free mem: %8.3g GB, total mem: %8.3g GB",
            ((double)free_mem) / (1024. * 1024. * 1024.),
            ((double)total_mem) / (1024. * 1024. * 1024.));
#endif
  }
}
