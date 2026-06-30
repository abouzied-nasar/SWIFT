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
#include "gpu_part_structs.h"

#include <cuda.h>
#include <cuda_runtime.h>

/**
 * @brief Initialize the GPU context for each thread. This should be
 * called in a threaded region, e.g. runner_main_cuda.
 */
void gpu_init_thread(struct engine *e, const int cpuid) {

  struct gpu_global_pack_params *gpu_pack_params = &e->gpu_pack_params;

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
    cu_error = cudaSetDevice(engine_rank % n_devices);
    swift_assert(cu_error == cudaSuccess);

    dev_id = engine_rank % n_devices;
    message("engine_rank %i got GPU %i", engine_rank, dev_id);
  }
#endif

  /* Now tell me some info about my device */
  cu_error = cudaGetDeviceProperties(&prop, dev_id);
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaDeviceGetAttribute(
      &max_blocks_SM, cudaDevAttrMaxBlocksPerMultiprocessor, dev_id);
  swift_assert(cu_error == cudaSuccess);

  cu_error =
      cudaDeviceGetAttribute(&n_SMs, cudaDevAttrMultiProcessorCount, dev_id);
  swift_assert(cu_error == cudaSuccess);

  size_t free_mem;
  size_t total_mem;
  const struct space *space = e->s;
  cu_error = cudaMemGetInfo(&free_mem, &total_mem);
  swift_assert(cu_error == cudaSuccess);

  int nPartsPerCell = space->nr_parts / space->tot_cells;
  if (cpuid == 0 && engine_rank == 0) {
    message("   Devices available:          %i", n_devices);
    message("   Device id:                  %i", dev_id);
    message("   Device name:                %s", prop.name);
    message("   n_SMs:                      %i", n_SMs);
    message("   max blocks per SM:          %i", max_blocks_SM);
    message("   max blocks per stream:      %i", n_SMs * max_blocks_SM);
    message(
        "   Target n_blocks per kernel: %d",
        gpu_pack_params->bundle_size * nPartsPerCell / GPU_THREAD_BLOCK_SIZE);
    message("   Target n_blocks per stream: %d",
            gpu_pack_params->pack_size * nPartsPerCell / GPU_THREAD_BLOCK_SIZE);
    message("   Leaf cell buffer size:      %i",
            gpu_pack_params->leaf_buffer_size);
    message("   Particles buffer size:      %i",
            gpu_pack_params->part_buffer_size);
    message("   Pack size:                  %i", gpu_pack_params->pack_size);
    message("   Bundle size:                %i", gpu_pack_params->bundle_size);
    message("   free mem:                   %.3g GB",
            ((double)free_mem) / (1024. * 1024. * 1024.));
    message("   total mem:                  %.3g GB",
            ((double)total_mem) / (1024. * 1024. * 1024.));
  }
  /*Based on the GPU memory available, let's calculate how much to use per CPU thread*/
  size_t safe_free_mem = free_mem;
  /*Check if we have more than 32GB. If we do, leave 5GB free to be safe.
   * Otherwise use up 90% of available memory. Pulled out of the air but
   * most decent GPUs have > 32 GB. Also a good fail-safe for when other users
   * use same GPU*/
  size_t GB2Byte = (1024 * 1024 * 1024);
  if(free_mem * GB2Byte > 32)
	  safe_free_mem =  free_mem - 5 * GB2Byte;
  else
	  safe_free_mem =  free_mem * 90 / 100;

  size_t free_mem_per_thread =  (safe_free_mem + e->nr_threads - 1)/e->nr_threads;
  if (cpuid == 0 && engine_rank == 0) {
    message("   safe to use free mem:       %.3g GB",
              ((double)safe_free_mem) / (1024. * 1024. * 1024.));
    message("   per thread:                 %.3g GB",
                ((double)free_mem_per_thread) / (1024. * 1024. * 1024.));
  }

  /*Get sizes of all the structs containing particle data*/
  size_t mem_send_d = sizeof(struct gpu_part_data_d);
  size_t mem_send_g = sizeof(struct gpu_part_data_g);
  size_t mem_send_f = sizeof(struct gpu_part_data_f);
  size_t mem_recv_d = sizeof(struct gpu_part_recv_d);
  size_t mem_recv_g = sizeof(struct gpu_part_recv_g);
  size_t mem_recv_f = sizeof(struct gpu_part_recv_f);

  /* Total mem required per thread per particle */
  double mem_req_part = mem_send_d + mem_send_g + mem_send_f
		  + mem_recv_d + mem_recv_g + mem_recv_f;

  /* Memory required per leaf computation launched */
  double mem_req_leaf_computation = sizeof(int4);

  /* Memory required per CUDA block launched, each CUDA block needs to know which cell it will work on */
  double mem_req_CUDA_block = sizeof(int2);

  /* Now we need to figure out how much memory to assign to what */
  /* We need one instance of block_ID per GPU_THREAD_BLOCK_SIZE particles */
  /* As a conservative estimate, let's say all leaf cells have a uniform
   * number of particles proportional to 2H. We therefore need one
   * cell_start_end per np particles*/

  /* Here we try to estimate average number of
   * particles per leaf-cell. */
  /* Get smoothing length/particle spacing */
                  /*1.2 is a random safety buffer*/
  double np_per_cell = 1.2 * 2 * ceil(2.0 * e->s->eta_neighbours);
  /* Apply appropriate dimensional multiplication */
#if defined(HYDRO_DIMENSION_2D)
  np_per_cell *= np_per_cell;
#elif defined(HYDRO_DIMENSION_3D)
  np_per_cell *= np_per_cell * np_per_cell;
#elif defined(HYDRO_DIMENSION_1D)
#endif

  /*Figure out how much memory we need per particle*/
  double total_memory_per_particle = mem_req_part + mem_req_leaf_computation/np_per_cell +
		  mem_req_CUDA_block/GPU_THREAD_BLOCK_SIZE;

  /*Now figure out what fraction of freemem we assign to what*/
  double fraction_of_memory_for_parts = free_mem_per_thread * mem_req_part/total_memory_per_particle;
  double fraction_of_memory_for_cell_md = free_mem_per_thread *
		  mem_req_leaf_computation/(np_per_cell * total_memory_per_particle);
  double fraction_of_memory_for_blockid = free_mem_per_thread *
		  mem_req_CUDA_block/(GPU_THREAD_BLOCK_SIZE * total_memory_per_particle);

  /* Now simply calculate how much of each data type we can into the fraction of memory allocated and assign
   * buffer sizes */
  /* TODO: part_buffer_size is currently read in from yml. For now over-write it here
   * but come back and make it so that we no longer read it in. */
  int buf_size_avail = (int)fraction_of_memory_for_parts/(int)mem_req_part;
  if(buf_size_avail < gpu_pack_params->part_buffer_size)
	  error("Only %.4gGB memory available on GPU per thread -> This fits %i particles in buffer per thread but "
			  "our minimum threshold (or size requested) is set to %i.", (double)free_mem_per_thread/(1024. * 1024. * 1024.),
			  buf_size_avail, gpu_pack_params->part_buffer_size);
  gpu_pack_params->part_buffer_size = buf_size_avail;
  gpu_pack_params->cell_start_end_buffer_size = (int)fraction_of_memory_for_cell_md/(int)mem_req_leaf_computation;
  gpu_pack_params->cuda_blockid_buffer_size = (int)fraction_of_memory_for_blockid/(int)mem_req_CUDA_block;

}

/**
 * @brief Initialize the GPU context for each thread. This should be
 * called in a threaded region, e.g. runner_main_cuda.
 */
void gpu_print_free_mem(const struct engine *e, const int cpuid) {

  /* Find and print GPU name(s) */
  int dev_id = 0;
  int n_devices;
  cudaError_t cu_error = cudaGetDeviceCount(&n_devices);
  swift_assert(cu_error == cudaSuccess);

#ifdef WITH_MPI
  if (n_devices != 1) {
    dev_id = engine_rank % n_devices;
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
        "pciBusID %4d, After allocation: free mem: %2.3g GB, total mem: %8.3g "
        "GB",
        prop.pciBusID, ((double)free_mem) / (1024. * 1024. * 1024.),
        ((double)total_mem) / (1024. * 1024. * 1024.));
#else
    message("After allocation: free mem: %2.3g GB, total mem: %8.3g GB",
            ((double)free_mem) / (1024. * 1024. * 1024.),
            ((double)total_mem) / (1024. * 1024. * 1024.));
#endif
  }
}
