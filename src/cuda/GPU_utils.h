#ifndef CUDA_GPU_INIT_H
#define CUDA_GPU_INIT_H

#include "engine.h"
#include "runner.h"

void gpu_init_thread(const struct engine* e, const int cpuid);
void gpu_print_free_mem(const struct engine* e, const int cpuid);


#endif
