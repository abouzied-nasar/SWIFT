#ifndef CUDA_GPU_RUNNER_FUNCTIONS_H
#define CUDA_GPU_RUNNER_FUNCTIONS_H
#define n_streams 1024

#ifdef __cplusplus
extern "C" {
#endif
#include <cuda_runtime.h>

#include "GPU_part_structs.h"

void gpu_launch_self_density(struct part_aos_f4_send_d *parts_send,
                           struct part_aos_f4_recv_d *parts_recv, float d_a,
                           float d_H, cudaStream_t stream, int numBlocks_x,
                           int numBlocks_y, int bundle_first_task,
                           int2 *d_task_first_part_f4);
void gpu_launch_self_gradient(struct part_aos_f4_send_g *parts_send,
                            struct part_aos_f4_recv_g *parts_recv, float d_a,
                            float d_H, cudaStream_t stream, int numBlocks_x,
                            int numBlocks_y, int bundle_first_task,
                            int2 *d_task_first_part_f4);
void gpu_launch_self_force(struct part_aos_f4_send_f *parts_send,
                         struct part_aos_f4_recv_f *parts_recv, float d_a,
                         float d_H, cudaStream_t stream, int numBlocks_x,
                         int numBlocks_y, int bundle_first_task,
                         int2 *d_task_first_part_f4);
void gpu_launch_pair_density(
    struct part_aos_f4_send_d *parts_send, struct part_aos_f4_recv_d
    *parts_recv, float d_a, float d_H, cudaStream_t stream,
    int numBlocks_x, int numBlocks_y, int bundle_first_part,
    int bundle_n_parts);
void gpu_launch_pair_gradient(
    struct part_aos_f4_send_g *parts_send,
    struct part_aos_f4_recv_g *parts_recv, float d_a, float d_H,
    cudaStream_t stream, int numBlocks_x, int numBlocks_y,
    int bundle_first_part, int bundle_n_parts);
void gpu_launch_pair_force(
    struct part_aos_f4_send_f *parts_send,
    struct part_aos_f4_recv_f *parts_recv, float d_a, float d_H,
    cudaStream_t stream, int numBlocks_x, int numBlocks_y,
    int bundle_first_part, int bundle_n_parts);
#ifdef __cplusplus
}
#endif

#endif  // GPU_RUNNER_FUNCTIONS_H
