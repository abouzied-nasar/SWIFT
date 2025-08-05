#ifndef CUDA_CONFIG_H
#define CUDA_CONFIG_H

/**
 * @file src/cuda/cuda_config.h
 * @brief Temporary file to store all macro definitions relating to
 * configuring the cuda setup/run until we sort everything out cleanly.
 */

#define BLOCK_SIZE 64
#define N_TASKS_PER_PACK_SELF 8
#define N_TASKS_BUNDLE_SELF 2

#define BLOCK_SIZE_PAIR 64
#define N_TASKS_PER_PACK_PAIR 8
#define N_TASKS_BUNDLE_PAIR 2


#define CUDA_DEBUG

/* Config parameters. */
#define GPUOFFLOAD_DENSITY 1   /* off-load hydro density to GPU */
#define GPUOFFLOAD_GRADIENT 1  /* off-load hydro gradient to GPU */
#define GPUOFFLOAD_FORCE 1     /* off-load hydro force to GPU */


//A. Nasar: Remove as will no longer be necessary. Leaving for now during dev
#define RECURSE 1 //Allow recursion through sub-tasks before offloading

#endif
