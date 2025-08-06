/**
 * @file src/cuda/GPU_utils.c
 * @brief misc GPU utilities
 */

#include "GPU_utils.h"
#include "cuda_config.h"
#include "runner.h"

#ifdef WITH_MPI
#include <mpi.h>
#endif
#include <cuda.h>
#include <cuda_runtime.h>



/**
 * @brief Initialize the GPU context for each thread. This should be
 * called in a threaded region, e.g. runner_main_cuda.
 */
void gpu_init_thread(const struct engine* e, const int cpuid){

  /* Find and print GPU name(s) */
  int dev_id = 0;  /* gpu device name */
  struct cudaDeviceProp prop;
  int n_devices;
  int max_blocks_SM;
  int n_SMs;

  cudaGetDeviceCount(&n_devices);
  /* A. Nasar: If running on MPI we set code to use one MPI rank per GPU
   * This was found to work very well and simplifies writing slurm scipts */
#ifdef WITH_MPI
  /*Get my rank*/
  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif
  if (n_devices == 1) {
    cudaSetDevice(dev_id);
  }
#ifdef WITH_MPI
  else {
    cudaSetDevice(mpi_rank);
    dev_id = mpi_rank;
  }
#endif

  /* Now tell me some info about my device */
  cudaGetDeviceProperties(&prop, dev_id);
  cudaDeviceGetAttribute(&max_blocks_SM, cudaDevAttrMaxBlocksPerMultiprocessor, dev_id);
  cudaDeviceGetAttribute(&n_SMs, cudaDevAttrMultiProcessorCount, dev_id);

  size_t free_mem;
  size_t total_mem;
  const struct space* space = e->s;
  cudaMemGetInfo(&free_mem, &total_mem);
  int nPartsPerCell = space->nr_parts / space->tot_cells;
  if (cpuid == 0 && engine_rank == 0) {
    message("Devices available:          %i", n_devices);
    message("Device id:                  %i", dev_id);
    message("Device name:                %s", prop.name);
    message("n_SMs:                      %i", n_SMs);
    message("max blocks per SM:          %i", max_blocks_SM);
    message("max blocks per stream:      %i", n_SMs * max_blocks_SM);
    message("Target n_blocks per kernel: %i", N_TASKS_BUNDLE_SELF * nPartsPerCell / BLOCK_SIZE);
    message("Target n_blocks per stream: %i", N_TASKS_PER_PACK_SELF * nPartsPerCell / BLOCK_SIZE);
    message("free mem:                   %lu", free_mem);
    message("total mem:                  %lu", total_mem);
  }
}

/**
 * @brief Initialize the GPU context for each thread. This should be
 * called in a threaded region, e.g. runner_main_cuda.
 */
void gpu_print_free_mem(const struct engine* e, const int cpuid){

  /* Find and print GPU name(s) */
  int dev_id = engine_rank;  /* gpu device name */
  struct cudaDeviceProp prop;

  /* Now tell me some info about my device */
  cudaGetDeviceProperties(&prop, dev_id);

  size_t free_mem;
  size_t total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  if (cpuid == 0) {
    message("After allocation: free mem: %lu / total mem: %lu", free_mem, total_mem);
  }
}


