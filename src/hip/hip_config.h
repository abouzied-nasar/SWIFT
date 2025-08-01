#ifndef HIP_CONFIG_H
#define HIP_CONFIG_H

/**
 * @file src/hip/hip_config.h
 * @brief Temporary file to store all macro definitions relating to
 * configuring the hip setup/run until we sort everything out cleanly.
 */

#define BLOCK_SIZE 64
#define N_TASKS_PER_PACK_SELF 8
#define N_TASKS_BUNDLE_SELF 2

#define BLOCK_SIZE_PAIR 64
#define N_TASKS_PER_PACK_PAIR 4
#define N_TASKS_BUNDLE_PAIR 1

#endif  // HIP_CONFIG_H
