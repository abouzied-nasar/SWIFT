#ifndef CUDA_GPU_RUNNER_FUNCTIONS_H
#define CUDA_GPU_RUNNER_FUNCTIONS_H
#define n_streams 1024

#ifdef __cplusplus
extern "C" {
#endif
#include <cuda_runtime.h>

#include "GPU_part_structs.h"

void launch_density_aos_f4(struct part_aos_f4_send_d *parts_send,
                           struct part_aos_f4_recv_d *parts_recv, float d_a,
                           float d_H, cudaStream_t stream, int numBlocks_x,
                           int numBlocks_y, int bundle_first_task,
                           int2 *d_task_first_part_f4);
void launch_gradient_aos_f4(struct part_aos_f4_send_g *parts_send,
                            struct part_aos_f4_recv_g *parts_recv, float d_a,
                            float d_H, cudaStream_t stream, int numBlocks_x,
                            int numBlocks_y, int bundle_first_task,
                            int2 *d_task_first_part_f4);
void launch_force_aos_f4(struct part_aos_f4_send_f *parts_send,
                         struct part_aos_f4_recv_f *parts_recv, float d_a,
                         float d_H, cudaStream_t stream, int numBlocks_x,
                         int numBlocks_y, int bundle_first_task,
                         int2 *d_task_first_part_f4);
void runner_dopair_branch_density_gpu_aos_f4(
    struct part_aos_f4_send_d *parts_send, struct part_aos_f4_recv_d
    *parts_recv, float d_a, float d_H, cudaStream_t stream,
    int numBlocks_x, int numBlocks_y, int bundle_first_part,
    int bundle_n_parts);
void runner_dopair_branch_gradient_gpu_aos_f4(
    struct part_aos_f4_send_g *parts_send,
    struct part_aos_f4_recv_g *parts_recv, float d_a, float d_H,
    cudaStream_t stream, int numBlocks_x, int numBlocks_y,
    int bundle_first_part, int bundle_n_parts);
void runner_dopair_branch_force_gpu_aos_f4(
    struct part_aos_f4_send_f *parts_send,
    struct part_aos_f4_recv_f *parts_recv, float d_a, float d_H,
    cudaStream_t stream, int numBlocks_x, int numBlocks_y,
    int bundle_first_part, int bundle_n_parts);
#ifdef __cplusplus
}
#endif

#endif  // GPU_RUNNER_FUNCTIONS_H
