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

/*******************************************************************************
 * This file contains functions used to setup and execute GPU tasks from within
 * runner_main.c. Consider this a translator allowing .cu based functions to be
 * called from within runner_main.c
 ******************************************************************************/

/* ifdef __cplusplus prevents name mangling. C code sees exact names
 of functions rather than mangled template names produced by C++ */
#ifdef __cplusplus
extern "C" {
#endif

/* Required header files */
#include "cuda_config.h"
#include "cuda_particle_kernels.cuh"
#include "gpu_launch.h"

#include <config.h>
#include <cuda.h>
/* #include <cuda_device_runtime_api.h> */
/* #include <cuda_profiler_api.h> */
/* #include <cuda_runtime.h> */

/**
 * @brief Call the particle SPH density kernel.
 *
 * @param d_parts_send array on device containing particle data
 * @param d_parts_recv array on device to write results into
 * @param d_a current cosmological scale factor
 * @param d_H current Hubble constant
 * @param bundle_first_part index of first particle of this bundle in the
 * d_parts_* arrays
 * @param bundle_n_parts nr of particles in this bundle
 */
__global__ void cuda_launch_density(
    const struct gpu_part_send_d *__restrict__ d_parts_send,
    struct gpu_part_recv_d *__restrict__ d_parts_recv, const float d_a,
    const float d_H,
    const int4 *__restrict__ d_cell_i_j_start_end,	const int2 *__restrict__ d_block_leaf_id,
    const double3 space_dim) {

  /*get my block id globally in this kernel*/
  const int bid = blockIdx.x;
  /*Get my leaf computation id to find particle data and cell pos*/
  const int leafid = d_block_leaf_id[bid].x;
  /*Get id of first block working on this leaf computation*/
  /*needed to figure out which range of parts in cell*/
  /*this block of GPU threads will work on*/
  const int bid_0 = d_block_leaf_id[bid].y;
  /* Grab handles for where cells start and end */
  int4 cell_starts_ends_read = d_cell_i_j_start_end[leafid];
  /*Assign without accounting for ci/cj_end being the
   * index of cell positions. This will be accounted for
   * in cuda_kernel_density()*/
  const int ci_start = cell_starts_ends_read.x;
  const int ci_end = cell_starts_ends_read.y;

  /*Find which block this is within the list of thread
   * blocks acting on leaf computation leafid*/
  const int b_id_local = bid - bid_0;
  /*Now find the particle this thread needs to work on*/
  const int pid = b_id_local * GPU_THREAD_BLOCK_SIZE + threadIdx.x + ci_start;
  /*Assign without accounting for ci/cj_end being the index of cell positions.
   * This will be accounted for in cuda_kernel_density()*/
  //TODO: Edit comment. This is no longer the case, we have conditions in this function preventing entry
  const int cj_start = cell_starts_ends_read.z;
  const int cj_end = cell_starts_ends_read.w;

  if(ci_end <= 0 || cj_end <= 0){
    printf("indices smaller than zero ci_end %i cj_end %i\n", ci_end, cj_end);
  }
  /*First things first. Find the shifts in case we're periodic*/
  /*Step I: Get the cell positions*/
  /*TODO: This can possibly be done with shared memory.
   * Unsure if it will give any speedup though*/
  const struct gpu_cell_pos ci_loc = d_parts_send[ci_end - 1].c_loc;
  const struct gpu_cell_pos cj_loc = d_parts_send[cj_end - 1].c_loc;

  double3 shift = {0.0, 0.0, 0.0};

  const double distx = cj_loc.x.x - ci_loc.x.x;
  const double disty = cj_loc.x.y - ci_loc.x.y;
  const double distz = cj_loc.x.z - ci_loc.x.z;

  /*Fine for now as thread divergence
   * will be three line of code max*/
  if(distx < -space_dim.x * 0.5)
    shift.x = space_dim.x;
  else if(distx > space_dim.x * 0.5)
    shift.x = -space_dim.x;

  if(disty < -space_dim.y * 0.5)
    shift.y = space_dim.y;
  else if(disty > space_dim.y * 0.5)
    shift.y = -space_dim.y;

  if(distz < -space_dim.z * 0.5)
    shift.z = space_dim.z;
  else if (distz > space_dim.z * 0.5)
    shift.z = -space_dim.z;

  const double cell_dist = sqrt(distx*distx + disty*disty + distz*distz);

  /*Note: Since shift would be zero for self tasks where ci==cj
   * this is fine as is to avoid divergence*/
  const double shift_ix = shift.x + cj_loc.x.x;
  const double shift_iy = shift.y + cj_loc.x.y;
  const double shift_iz = shift.z + cj_loc.x.z;

  const double3 shift_i_res = {shift_ix, shift_iy, shift_iz};
  const double3 shift_j_res = {cj_loc.x.x, cj_loc.x.y, cj_loc.x.z};
  /*Interact parts in ci with parts in cj.
   * Check to see if pid is in-bounds first*/
  /*Remember that the last index is used to store cell location, subtract 1 for limit*/
  if(pid < ci_end - 1)
    cuda_kernel_density_p(leafid, d_parts_send, d_parts_recv, d_a, d_H,
        cell_starts_ends_read, space_dim, shift_i_res, shift_j_res, pid);
  /*Check if this is a self interaction.
   * If it is, skip as we don't need to re-do computations
   * We could just let threads do this again to avoid
   * divergence since we only unpack ci once on host
   * (for it's ci not the dummy cj used for indexing)*/
  /*TODO: This needs re-working from host code down.
   * Self tasks are run with pairs so while threads doing selfs have finished
   * threads doing pairs will be re-doing comp.s for parts in cell j*/
  if(ci_start != cj_start){
    //      printf("Doing cj\n");
    /*We've done ci with cj, now interact cj with ci*/
    cell_starts_ends_read.x = cj_start;
    cell_starts_ends_read.y = cj_end;
    cell_starts_ends_read.z = ci_start;
    cell_starts_ends_read.w = ci_end;
    /*Re-calculate shifts*/
    const double3 shift_ii_res = {cj_loc.x.x, cj_loc.x.y, cj_loc.x.z};
    const double3 shift_jj_res = {shift_ix, shift_iy, shift_iz};
    /*Now find the particle this thread needs to work on from cell j*/
    /*TODO: Re-work the kernel so that each thread
     * only does one particle in cell i or cell j. NOT BOTH.
     * This code is anticipated to give huge load imbalance
     * when ci is smaller than cj and vice-versa*/
    const int pjd = b_id_local * GPU_THREAD_BLOCK_SIZE + threadIdx.x + cj_start;
    ///////////////////////////////////////////////////////////////////////
    /*Remember that the last index is used to store cell location*/
    if(pjd < cj_end - 1)
      cuda_kernel_density_p(leafid, d_parts_send, d_parts_recv, d_a, d_H,
          cell_starts_ends_read, space_dim, shift_ii_res, shift_jj_res, pjd);
  }
}

//Safe version////////////////
__device__ __forceinline__
void process_range_tiled_density(
    const struct gpu_part_send_d* __restrict__ d_parts_send,
    struct gpu_part_recv_d* __restrict__ d_parts_recv,
    int i_start, int i_end_excl,
    int j_start, int j_end_excl,
    const double3 shift_i_d, const double3 shift_j_d,
    int b_id_local, int tid)
{
    // Shared memory for one tile of J: positions and velocities
    extern __shared__ unsigned char smem[];
    //TODO: Check if this is safe and/or required. We're casting from float4 to float4
    //Also, we need positions to be double so this may need re-working!
    float4* s_pos4 = reinterpret_cast<float4*>(smem);                   // [TILE_J]
    float4* s_vel4 = reinterpret_cast<float4*>(s_pos4 + TILE_J);        // [TILE_J]

    // Map this thread to its i-particle
    const int i_idx = b_id_local * GPU_THREAD_BLOCK_SIZE + tid + i_start;
    const bool i_in_range = (i_idx < i_end_excl);

    // Declare i-data, initialize safely; only load if active
    float xi = 0.f, yi = 0.f, zi = 0.f, hi = 1.f;
    float vxi = 0.f, vyi = 0.f, vzi = 0.f;
    float hig2 = 0.f, hi_inv = 1.f;

    if (i_in_range) {
        const auto pi = d_parts_send[i_idx].p_data;
        xi = (float)(pi.x_h.x - shift_i_d.x);
        yi = (float)(pi.x_h.y - shift_i_d.y);
        zi = (float)(pi.x_h.z - shift_i_d.z);
        hi = (float)(pi.x_h.w);

        vxi = pi.vx_m.x;
        vyi = pi.vx_m.y;
        vzi = pi.vx_m.z;

        hig2   = (hi * hi) * kernel_gamma2;
        hi_inv = 1.0f / hi;
    }

    float4 res_rho = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 res_rot = make_float4(0.f, 0.f, 0.f, 0.f);
    constexpr float eps = 1e-24f;

    // Tile over J
    for (int base = j_start; base < j_end_excl; base += TILE_J)
    {
      /*Figure out if we're doing a full tile of j particles of
       * we only have leftovers ( < a full tile )*/
        const int tileCount = min(TILE_J, j_end_excl - base);

        // Cooperative load of this J tile into shared memory
        for (int t = tid; t < tileCount; t += GPU_THREAD_BLOCK_SIZE)
        {
            const int gj = base + t;
            s_pos4[t] = d_parts_send[gj].p_data.x_h;  // (x,y,z,hj) unshifted
            s_vel4[t] = d_parts_send[gj].p_data.vx_m; // (vx,vy,vz,m)
        }
        __syncthreads();

        // Compute only if this thread owns a valid i
        if (i_in_range)
        {
#pragma unroll 4
            for (int t = 0; t < tileCount; ++t)
            {
                const int j_idx = base + t;
                if (j_idx == i_idx) continue;  // We do not add i contribution. This is done in ghost tasks

                const float4 pj_pos = s_pos4[t]; // unshifted
                const float4 pj_vel = s_vel4[t];

                // Apply j shift on-the-fly
                const float xij = xi - (pj_pos.x - (float)shift_j_d.x);
                const float yij = yi - (pj_pos.y - (float)shift_j_d.y);
                const float zij = zi - (pj_pos.z - (float)shift_j_d.z);

                //fmaf -> fused multiply addition operator
                const float r2 = fmaf(xij, xij, fmaf(yij, yij, zij * zij));
                if (r2 >= hig2) continue;

                //Clever Co-Pilot
                const float inv_r = rsqrtf(r2 + eps);
                //Very clever Co-Pilot, multiply instead of divide
                const float r     = r2 * inv_r;
                const float ui    = r * hi_inv;

                float wi, wi_dx;
                d_kernel_deval(ui, &wi, &wi_dx);

                const float mj  = pj_vel.w;
                const float tmp = (hydro_dimension * wi + ui * wi_dx);

                // rho, rho_dh, wcount, wcount_dh
                res_rho.x += mj * wi;
                res_rho.y -= mj * tmp;
                res_rho.z += wi;
                res_rho.w -= tmp;

                const float faci = mj * wi_dx * inv_r;

                const float dvx = vxi - pj_vel.x;
                const float dvy = vyi - pj_vel.y;
                const float dvz = vzi - pj_vel.z;

                const float dvdr   = fmaf(dvx, xij, fmaf(dvy, yij, dvz * zij));
                const float curlrx = fmaf(dvy, zij, -dvz * yij);
                const float curlry = fmaf(dvz, xij, -dvx * zij);
                const float curlrz = fmaf(dvx, yij, -dvy * xij);

                res_rot.x = fmaf(faci,  curlrx, res_rot.x);
                res_rot.y = fmaf(faci,  curlry, res_rot.y);
                res_rot.z = fmaf(faci,  curlrz, res_rot.z);
                res_rot.w = fmaf(-faci, dvdr,    res_rot.w);
            }
        }

        __syncthreads(); // all threads sync before next tile
    }

    // Atomics only if active
    if (i_in_range) {
        atomicAdd(&d_parts_recv[i_idx].rho_rhodh_wcount_wcount_dh.x, res_rho.x);
        atomicAdd(&d_parts_recv[i_idx].rho_rhodh_wcount_wcount_dh.y, res_rho.y);
        atomicAdd(&d_parts_recv[i_idx].rho_rhodh_wcount_wcount_dh.z, res_rho.z);
        atomicAdd(&d_parts_recv[i_idx].rho_rhodh_wcount_wcount_dh.w, res_rho.w);

        atomicAdd(&d_parts_recv[i_idx].rot_vx_div_v.x, res_rot.x);
        atomicAdd(&d_parts_recv[i_idx].rot_vx_div_v.y, res_rot.y);
        atomicAdd(&d_parts_recv[i_idx].rot_vx_div_v.z, res_rot.z);
        atomicAdd(&d_parts_recv[i_idx].rot_vx_div_v.w, res_rot.w);
    }
}

__global__ __launch_bounds__(GPU_THREAD_BLOCK_SIZE, 2 /*min number of blocks we launch per SM*/)
void cuda_launch_tiled_density(
    const struct gpu_part_send_d* __restrict__ d_parts_send,
    struct gpu_part_recv_d* __restrict__ d_parts_recv,
    const float d_a, const float d_H,
    const int4* __restrict__ d_cell_i_j_start_end,
    const int2* __restrict__ d_block_leaf_id,
    const double3 space_dim){

    const int bid     = blockIdx.x;
    const int leafid  = d_block_leaf_id[bid].x;
    const int bid_0   = d_block_leaf_id[bid].y;
    int4 cell_se      = d_cell_i_j_start_end[leafid];

    const int ci_start = cell_se.x;
    const int ci_end   = cell_se.y; // cell pos at index [ci_end-1]
    const int cj_start = cell_se.z;
    const int cj_end   = cell_se.w; // cell pos at index [cj_end-1]

    const int b_id_local = bid - bid_0;
    const int tid = threadIdx.x;

    // Let's get the cell positions
    const auto ci_loc = d_parts_send[ci_end - 1].c_loc;
    const auto cj_loc = d_parts_send[cj_end - 1].c_loc;

    double3 shift = {0.0, 0.0, 0.0};
    const double distx = cj_loc.x.x - ci_loc.x.x;
    const double disty = cj_loc.x.y - ci_loc.x.y;
    const double distz = cj_loc.x.z - ci_loc.x.z;

    if (distx < -space_dim.x * 0.5)      shift.x =  space_dim.x;
    else if (distx >  space_dim.x * 0.5)  shift.x = -space_dim.x;
    if (disty < -space_dim.y * 0.5)      shift.y =  space_dim.y;
    else if (disty >  space_dim.y * 0.5)  shift.y = -space_dim.y;
    if (distz < -space_dim.z * 0.5)      shift.z =  space_dim.z;
    else if (distz >  space_dim.z * 0.5)  shift.z = -space_dim.z;

    const double3 shift_i_res = {shift.x + cj_loc.x.x, shift.y + cj_loc.x.y, shift.z + cj_loc.x.z};
    const double3 shift_j_res = {cj_loc.x.x, cj_loc.x.y, cj_loc.x.z};

    // Pass 1: ci <- cj
    process_range_tiled_density(
        d_parts_send, d_parts_recv,
        ci_start, ci_end - 1,
        cj_start, cj_end - 1,
        shift_i_res, shift_j_res,
        b_id_local, tid
    );

    // Pass 2: cj <- ci (only if not self)
    if (ci_start != cj_start) {
        const double3 shift_ii_res = {cj_loc.x.x, cj_loc.x.y, cj_loc.x.z};
        const double3 shift_jj_res = {shift.x + cj_loc.x.x, shift.y + cj_loc.x.y, shift.z + cj_loc.x.z};

        process_range_tiled_density(
            d_parts_send, d_parts_recv,
            cj_start, cj_end - 1,
            ci_start, ci_end - 1,
            shift_ii_res, shift_jj_res,
            b_id_local, tid
        );
    }
}


void gpu_launch_tiled_density(
    const struct gpu_part_send_d* __restrict__ d_parts_send,
    struct gpu_part_recv_d* __restrict__ d_parts_recv,
    const float d_a, const float d_H,
    int num_blocks_x,
    const int4* __restrict__ d_cell_i_j_start_end,
    const int2* __restrict__ d_block_leaf_id,
    const double3 space_dim,
    cudaStream_t stream)
{
    // Shared memory allocation
    const size_t shmem = TILE_J * (sizeof(struct gpu_part_recv_d));//(sizeof(float4) + sizeof(float4)); // 2048 bytes when TILE_J=64

    cuda_launch_tiled_density<<<num_blocks_x, GPU_THREAD_BLOCK_SIZE, shmem, stream>>>(
        d_parts_send, d_parts_recv, d_a, d_H,
        d_cell_i_j_start_end, d_block_leaf_id, space_dim);
}
///////////////////////////

/**
 * @brief Call the particle SPH gradient kernel.
 *
 * @param d_parts_send array on device containing particle data
 * @param d_parts_recv array on device to write results into
 * @param d_a current cosmological scale factor
 * @param d_H current Hubble constant
 * @param bundle_first_part index of first particle of this bundle in the
 * d_parts_* arrays
 * @param bundle_n_parts nr of particles in this bundle
 */
__global__ void cuda_launch_gradient(
    const struct gpu_part_send_g *__restrict__ d_parts_send,
    struct gpu_part_recv_g *__restrict__ d_parts_recv, float d_a, float d_H,
    int bundle_first_part, int bundle_n_parts) {

  const int threadid = blockDim.x * blockIdx.x + threadIdx.x;
  const int pid = bundle_first_part + threadid;

  if (pid < bundle_first_part + bundle_n_parts) {
    cuda_kernel_gradient(pid, d_parts_send, d_parts_recv, d_a, d_H);
  }
}

/**
 * @brief Call the particle SPH density kernel.
 *
 * @param d_parts_send array on device containing particle data
 * @param d_parts_recv array on device to write results into
 * @param d_a current cosmological scale factor
 * @param d_H current Hubble constant
 * @param bundle_first_part index of first particle of this bundle in the
 * d_parts_* arrays
 * @param bundle_n_parts nr of particles in this bundle
 */
__global__ void cuda_launch_unique_gradient(
    const struct gpu_part_send_g *__restrict__ d_parts_send,
    struct gpu_part_recv_g *__restrict__ d_parts_recv, const float d_a,
    const float d_H,
    const int4 *__restrict__ d_cell_i_j_start_end,  const int2 *__restrict__ d_block_leaf_id,
    const double3 space_dim) {

  /*get my block id globally in this kernel*/
  const int bid = blockIdx.x;
  /*Get my leaf computation id to find particle data and cell pos*/
  const int leafid = d_block_leaf_id[bid].x;
  /*Get id of first block working on this leaf computation*/
  /*needed to figure out which range of parts in cell*/
  /*this block of GPU threads will work on*/
  const int bid_0 = d_block_leaf_id[bid].y;
  /* Grab handles for where cells start and end */
  int4 cell_starts_ends_read = d_cell_i_j_start_end[leafid];
  /*Assign without accounting for ci/cj_end being the
   * index of cell positions. This will be accounted for
   * in cuda_kernel_density()*/
  const int ci_start = cell_starts_ends_read.x;
  const int ci_end = cell_starts_ends_read.y;

  /*Find which block this is within the list of thread
   * blocks acting on leaf computation leafid*/
  const int b_id_local = bid - bid_0;
  /*Now find the particle this thread needs to work on*/
  const int pid = b_id_local * GPU_THREAD_BLOCK_SIZE + threadIdx.x + ci_start;
  /*Assign without accounting for ci/cj_end being the index of cell positions.
   * This will be accounted for in cuda_kernel_density()*/
  //TODO: Edit comment. This is no longer the case, we have conditions in this function preventing entry
  const int cj_start = cell_starts_ends_read.z;
  const int cj_end = cell_starts_ends_read.w;

  if(ci_end <= 0 || cj_end <= 0){
    printf("indices smaller than zero ci_end %i cj_end %i\n", ci_end, cj_end);
  }
  /*First things first. Find the shifts in case we're periodic*/
  /*Step I: Get the cell positions*/
  /*TODO: This can possibly be done with shared memory.
   * Unsure if it will give any speedup though*/
  const struct gpu_cell_pos ci_loc = d_parts_send[ci_end - 1].c_loc;
  const struct gpu_cell_pos cj_loc = d_parts_send[cj_end - 1].c_loc;

  double3 shift = {0.0, 0.0, 0.0};

  const double distx = cj_loc.x.x - ci_loc.x.x;
  const double disty = cj_loc.x.y - ci_loc.x.y;
  const double distz = cj_loc.x.z - ci_loc.x.z;

  /*Fine for now as thread divergence
   * will be three line of code max*/
  if(distx < -space_dim.x * 0.5)
    shift.x = space_dim.x;
  else if(distx > space_dim.x * 0.5)
    shift.x = -space_dim.x;

  if(disty < -space_dim.y * 0.5)
    shift.y = space_dim.y;
  else if(disty > space_dim.y * 0.5)
    shift.y = -space_dim.y;

  if(distz < -space_dim.z * 0.5)
    shift.z = space_dim.z;
  else if (distz > space_dim.z * 0.5)
    shift.z = -space_dim.z;

  const double cell_dist = sqrt(distx*distx + disty*disty + distz*distz);

  /*Note: Since shift would be zero for self tasks where ci==cj
   * this is fine as is to avoid divergence*/
  const double shift_ix = shift.x + cj_loc.x.x;
  const double shift_iy = shift.y + cj_loc.x.y;
  const double shift_iz = shift.z + cj_loc.x.z;

  const double3 shift_i_res = {shift_ix, shift_iy, shift_iz};
  const double3 shift_j_res = {cj_loc.x.x, cj_loc.x.y, cj_loc.x.z};
  /*Interact parts in ci with parts in cj.
   * Check to see if pid is in-bounds first*/
  /*Remember that the last index is used to store cell location, subtract 1 for limit*/
  if(pid < ci_end - 1)
    cuda_kernel_gradient_p(leafid, d_parts_send, d_parts_recv, d_a, d_H,
        cell_starts_ends_read, space_dim, shift_i_res, shift_j_res, pid);
  /*Check if this is a self interaction.
   * If it is, skip as we don't need to re-do computations
   * We could just let threads do this again to avoid
   * divergence since we only unpack ci once on host
   * (for it's ci not the dummy cj used for indexing)*/
  /*TODO: This needs re-working from host code down.
   * Self tasks are run with pairs so while threads doing selfs have finished
   * threads doing pairs will be re-doing comp.s for parts in cell j (idle blocks)
   * Also, what if cj much bigger than ci? Need to re-write so we launch ci and
   * cj comp.s seperately*/
  if(ci_start != cj_start){
    //      printf("Doing cj\n");
    /*We've done ci with cj, now interact cj with ci*/
    cell_starts_ends_read.x = cj_start;
    cell_starts_ends_read.y = cj_end;
    cell_starts_ends_read.z = ci_start;
    cell_starts_ends_read.w = ci_end;
    /*Re-calculate shifts*/
    const double3 shift_ii_res = {cj_loc.x.x, cj_loc.x.y, cj_loc.x.z};
    const double3 shift_jj_res = {shift_ix, shift_iy, shift_iz};
    /*Now find the particle this thread needs to work on from cell j*/
    /*TODO: Re-work the kernel so that each thread
     * only does one particle in cell i or cell j. NOT BOTH.
     * This code is anticipated to give huge load imbalance
     * when ci is smaller than cj and vice-versa*/
    const int pjd = b_id_local * GPU_THREAD_BLOCK_SIZE + threadIdx.x + cj_start;
    ///////////////////////////////////////////////////////////////////////
    /*Remember that the last index is used to store cell location*/
    if(pjd < cj_end - 1)
      cuda_kernel_gradient_p(leafid, d_parts_send, d_parts_recv, d_a, d_H,
          cell_starts_ends_read, space_dim, shift_ii_res, shift_jj_res, pjd);
  }
}

/**
 * @brief Call the particle SPH density kernel.
 *
 * @param d_parts_send array on device containing particle data
 * @param d_parts_recv array on device to write results into
 * @param d_a current cosmological scale factor
 * @param d_H current Hubble constant
 * @param bundle_first_part index of first particle of this bundle in the
 * d_parts_* arrays
 * @param bundle_n_parts nr of particles in this bundle
 */
__global__ void cuda_launch_unique_force(
    const struct gpu_part_send_f *__restrict__ d_parts_send,
    struct gpu_part_recv_f *__restrict__ d_parts_recv, const float d_a,
    const float d_H,
    const int4 *__restrict__ d_cell_i_j_start_end,  const int2 *__restrict__ d_block_leaf_id,
    const double3 space_dim) {

  /*get my block id globally in this kernel*/
  const int bid = blockIdx.x;
  /*Get my leaf computation id to find particle data and cell pos*/
  const int leafid = d_block_leaf_id[bid].x;
  /*Get id of first block working on this leaf computation*/
  /*needed to figure out which range of parts in cell*/
  /*this block of GPU threads will work on*/
  const int bid_0 = d_block_leaf_id[bid].y;
  /* Grab handles for where cells start and end */
  int4 cell_starts_ends_read = d_cell_i_j_start_end[leafid];
  /*Assign without accounting for ci/cj_end being the
   * index of cell positions. This will be accounted for
   * in cuda_kernel_density()*/
  const int ci_start = cell_starts_ends_read.x;
  const int ci_end = cell_starts_ends_read.y;

  /*Find which block this is within the list of thread
   * blocks acting on leaf computation leafid*/
  const int b_id_local = bid - bid_0;
  /*Now find the particle this thread needs to work on*/
  const int pid = b_id_local * GPU_THREAD_BLOCK_SIZE + threadIdx.x + ci_start;
  /*Assign without accounting for ci/cj_end being the index of cell positions.
   * This will be accounted for in cuda_kernel_density()*/
  //TODO: Edit comment. This is no longer the case, we have conditions in this function preventing entry
  const int cj_start = cell_starts_ends_read.z;
  const int cj_end = cell_starts_ends_read.w;

  if(ci_end <= 0 || cj_end <= 0){
    printf("indices smaller than zero ci_end %i cj_end %i\n", ci_end, cj_end);
  }
  /*First things first. Find the shifts in case we're periodic*/
  /*Step I: Get the cell positions*/
  /*TODO: This can possibly be done with shared memory.
   * Unsure if it will give any speedup though*/
  const struct gpu_cell_pos ci_loc = d_parts_send[ci_end - 1].c_loc;
  const struct gpu_cell_pos cj_loc = d_parts_send[cj_end - 1].c_loc;

  double3 shift = {0.0, 0.0, 0.0};

  const double distx = cj_loc.x.x - ci_loc.x.x;
  const double disty = cj_loc.x.y - ci_loc.x.y;
  const double distz = cj_loc.x.z - ci_loc.x.z;

  /*Fine for now as thread divergence
   * will be three line of code max*/
  if(distx < -space_dim.x * 0.5)
    shift.x = space_dim.x;
  else if(distx > space_dim.x * 0.5)
    shift.x = -space_dim.x;

  if(disty < -space_dim.y * 0.5)
    shift.y = space_dim.y;
  else if(disty > space_dim.y * 0.5)
    shift.y = -space_dim.y;

  if(distz < -space_dim.z * 0.5)
    shift.z = space_dim.z;
  else if (distz > space_dim.z * 0.5)
    shift.z = -space_dim.z;

  const double cell_dist = sqrt(distx*distx + disty*disty + distz*distz);

  /*Note: Since shift would be zero for self tasks where ci==cj
   * this is fine as is to avoid divergence*/
  const double shift_ix = shift.x + cj_loc.x.x;
  const double shift_iy = shift.y + cj_loc.x.y;
  const double shift_iz = shift.z + cj_loc.x.z;

  const double3 shift_i_res = {shift_ix, shift_iy, shift_iz};
  const double3 shift_j_res = {cj_loc.x.x, cj_loc.x.y, cj_loc.x.z};
  /*Interact parts in ci with parts in cj.
   * Check to see if pid is in-bounds first*/
  /*Remember that the last index is used to store cell location, subtract 1 for limit*/
  if(pid < ci_end - 1)
    cuda_kernel_force_p(leafid, d_parts_send, d_parts_recv, d_a, d_H,
        cell_starts_ends_read, space_dim, shift_i_res, shift_j_res, pid);
  /*Check if this is a self interaction.
   * If it is, skip as we don't need to re-do computations
   * We could just let threads do this again to avoid
   * divergence since we only unpack ci once on host
   * (for it's ci not the dummy cj used for indexing)*/
  /*TODO: This needs re-working from host code down.
   * Self tasks are run with pairs so while threads doing selfs have finished
   * threads doing pairs will be re-doing comp.s for parts in cell j*/
  if(ci_start != cj_start){
    //      printf("Doing cj\n");
    /*We've done ci with cj, now interact cj with ci*/
    cell_starts_ends_read.x = cj_start;
    cell_starts_ends_read.y = cj_end;
    cell_starts_ends_read.z = ci_start;
    cell_starts_ends_read.w = ci_end;
    /*Re-calculate shifts*/
    const double3 shift_ii_res = {cj_loc.x.x, cj_loc.x.y, cj_loc.x.z};
    const double3 shift_jj_res = {shift_ix, shift_iy, shift_iz};
    /*Now find the particle this thread needs to work on from cell j*/
    /*TODO: Re-work the kernel so that each thread
     * only does one particle in cell i or cell j. NOT BOTH.
     * This code is anticipated to give huge load imbalance
     * when ci is smaller than cj and vice-versa*/
    const int pjd = b_id_local * GPU_THREAD_BLOCK_SIZE + threadIdx.x + cj_start;
    ///////////////////////////////////////////////////////////////////////
    /*Remember that the last index is used to store cell location*/
    if(pjd < cj_end - 1)
      cuda_kernel_force_p(leafid, d_parts_send, d_parts_recv, d_a, d_H,
          cell_starts_ends_read, space_dim, shift_ii_res, shift_jj_res, pjd);
  }
}

/**
 * @brief Call the particle SPH force kernel.
 *
 * @param d_parts_send array on device containing particle data
 * @param d_parts_recv array on device to write results into
 * @param d_a current cosmological scale factor
 * @param d_H current Hubble constant
 * @param bundle_first_part index of first particle of this bundle in the
 * d_parts_* arrays
 * @param bundle_n_parts nr of particles in this bundle
 */
__global__ void cuda_launch_force(
    const struct gpu_part_send_f *__restrict__ d_parts_send,
    struct gpu_part_recv_f *__restrict__ d_parts_recv, float d_a, float d_H,
    int bundle_first_part, int bundle_n_parts) {

  const int threadid = blockDim.x * blockIdx.x + threadIdx.x;
  const int pid = bundle_first_part + threadid;

  if (pid < bundle_first_part + bundle_n_parts) {
    cuda_kernel_force(pid, d_parts_send, d_parts_recv, d_a, d_H);
  }
}

/**
 * @brief Launch the density computation on the GPU for a bundle of leaf cells.
 *
 * @param d_parts_send array on device containing particle data
 * @param d_parts_recv array on device to write results into
 * @param d_a current cosmological scale factor
 * @param d_H current Hubble constant
 * @param stream cuda stream to use
 * @param num_blockx_x number of thread blocks to use in x-dimension
 * @param num_blockx_y number of thread blocks to use in y-dimension
 * @param bundle_first_part index of first particle of this bundle in the
 * d_parts_* arrays
 * @param bundle_n_parts nr of particles in this bundle
 */
void gpu_launch_density(const struct gpu_part_send_d *__restrict__ d_parts_send,
                        struct gpu_part_recv_d *__restrict__ d_parts_recv,
                        const float d_a, const float d_H,
                        const int num_blocks_x,
                        const int4 *__restrict__ d_cell_i_j_start_end,
                        const int2 *__restrict__ d_block_leaf_id,
                        const double3 space_dim, cudaStream_t stream) {

  /* TODO: Do we want to allocate shared memory here? */
  cuda_launch_density<<<num_blocks_x, GPU_THREAD_BLOCK_SIZE, 0, stream>>>(
      d_parts_send, d_parts_recv, d_a, d_H,
      d_cell_i_j_start_end, d_block_leaf_id, space_dim);
}

/**
 * @brief Launch the gradient computation on the GPU for a bundle of leaf cells.
 *
 * @param d_parts_send array on device containing particle data
 * @param d_parts_recv array on device to write results into
 * @param d_a current cosmological scale factor
 * @param d_H current Hubble constant
 * @param stream cuda stream to use
 * @param num_blockx_x number of thread blocks to use in x-dimension
 * @param num_blockx_y number of thread blocks to use in y-dimension
 * @param bundle_first_part index of first particle of this bundle in the
 * d_parts_* arrays
 * @param bundle_n_parts nr of particles in this bundle
 */
void gpu_launch_unique_gradient(const struct gpu_part_send_g *__restrict__ d_parts_send,
                        struct gpu_part_recv_g *__restrict__ d_parts_recv,
                        const float d_a, const float d_H,
                        const int num_blocks_x,
                        const int4 *__restrict__ d_cell_i_j_start_end,
                        const int2 *__restrict__ d_block_leaf_id,
                        const double3 space_dim, cudaStream_t stream) {

  /* TODO: Do we want to allocate shared memory here? */
  cuda_launch_unique_gradient<<<num_blocks_x, GPU_THREAD_BLOCK_SIZE, 0, stream>>>(
      d_parts_send, d_parts_recv, d_a, d_H,
      d_cell_i_j_start_end, d_block_leaf_id, space_dim);
}

/**
 * @brief Launch the gradient computation on the GPU for a bundle of leaf cells.
 *
 * @param d_parts_send array on device containing particle data
 * @param d_parts_recv array on device to write results into
 * @param d_a current cosmological scale factor
 * @param d_H current Hubble constant
 * @param stream cuda stream to use
 * @param num_blockx_x number of thread blocks to use in x-dimension
 * @param num_blockx_y number of thread blocks to use in y-dimension
 * @param bundle_first_part index of first particle of this bundle in the
 * d_parts_* arrays
 * @param bundle_n_parts nr of particles in this bundle
 */
void gpu_launch_gradient(
    const struct gpu_part_send_g *__restrict__ d_parts_send,
    struct gpu_part_recv_g *__restrict__ d_parts_recv, const float d_a,
    const float d_H, cudaStream_t stream, const int num_blocks_x,
    const int num_blocks_y, const int bundle_first_part,
    const int bundle_n_parts) {

  /* TODO: Do we want to allocate shared memory here? */
  cuda_launch_gradient<<<num_blocks_x, GPU_THREAD_BLOCK_SIZE, 0, stream>>>(
      d_parts_send, d_parts_recv, d_a, d_H, bundle_first_part, bundle_n_parts);
}

// ---- constants_compat (paste near top of the .cu) ----
#ifdef kernel_gamma2
  #define KERNEL_GAMMA2 (kernel_gamma2)
#else
  extern __device__ float kernel_gamma2;
  #define KERNEL_GAMMA2 kernel_gamma2
#endif

#ifdef const_viscosity_beta
  #define CONST_VISCOSITY_BETA (const_viscosity_beta)
#else
  extern __device__ float const_viscosity_beta;
  #define CONST_VISCOSITY_BETA const_viscosity_beta
#endif

// Your device helpers/constants already in your code base:
__device__ void  d_kernel_deval(float u, float* w, float* wdx);
__device__ float d_pow_three_gamma_minus_five_over_two(float a);

// Compute i-range against j-range using shared-memory tiles (no async prefetch)
__device__ __forceinline__
void process_range_gradient_tiled_noasync(
    const struct gpu_part_send_g* __restrict__ d_parts_send,
    struct gpu_part_recv_g*      __restrict__ d_parts_recv,
    // ranges (end_excl excludes the cell-position slot at [end-1])
    int i_start, int i_end_excl,
    int j_start, int j_end_excl,
    // periodic shifts
    const double3 shift_i_d, const double3 shift_j_d,
    // mapping
    int b_id_local, int tid,
    // cosmology
    float d_a, float d_H)
{
    // Shared memory tile for J: pos/h, vel/m, rho/avisc/u/c
    extern __shared__ unsigned char smem[];
    float4* s_pos4 = reinterpret_cast<float4*>(smem);                    // [TILE_J] (xj,yj,zj,hj)
    float4* s_vel4 = reinterpret_cast<float4*>(s_pos4 + TILE_J);         // [TILE_J] (vxj,vyj,vzj,mj)
    float4* s_rac4 = reinterpret_cast<float4*>(s_vel4 + TILE_J);         // [TILE_J] (rhoj,aviscj,energyj,cj)

    // Map this thread to its i-particle; keep all threads in lockstep (no early return)
    const int  i_idx    = b_id_local * GPU_THREAD_BLOCK_SIZE + tid + i_start;
    const bool i_in_range = (i_idx < i_end_excl);

    // Per-i data (only load if active)
    float xi=0.f, yi=0.f, zi=0.f, hi=1.f;
    float vxi=0.f, vyi=0.f, vzi=0.f;
    float energyi=0.f, ci=0.f;
    float vsigi=0.f, lapui=0.f, avisc_maxi=0.f;

    float hi_inv=1.f, hig2=0.f;

    if (i_in_range) {
        const auto pi = d_parts_send[i_idx].p_data;

        xi = (float)(pi.x_h.x - shift_i_d.x);
        yi = (float)(pi.x_h.y - shift_i_d.y);
        zi = (float)(pi.x_h.z - shift_i_d.z);
        hi = (float)(pi.x_h.w);

        vxi = pi.vx_m.x;  vyi = pi.vx_m.y;  vzi = pi.vx_m.z;

        // rho_avisc_u_c: (rho, avisc, u, c)
        energyi = pi.rho_avisc_u_c.z;
        ci      = pi.rho_avisc_u_c.w;

        // vsig_lapu_aviscmax: (vsig, laplace_u, avisc_max)
        vsigi       = pi.vsig_lapu_aviscmax.x;
        lapui       = pi.vsig_lapu_aviscmax.y;
        avisc_maxi  = pi.vsig_lapu_aviscmax.z;

        hi_inv = 1.0f / hi;
        hig2   = (hi * hi) * KERNEL_GAMMA2;
    }

    // Accumulators
    // Start from the i-side stored values (as in your original), then update over neighbors
    float3 res_vsig_lapu_avisci = {vsigi, lapui, avisc_maxi};

    // Cosmology terms
    const float fac_mu    = d_pow_three_gamma_minus_five_over_two(d_a);
    const float a2_Hubble = d_a * d_a * d_H;

    constexpr float eps = 1e-24f;

    // ---- Tile over J ----
    for (int base = j_start; base < j_end_excl; base += TILE_J)
    {
        const int tileCount = min(TILE_J, j_end_excl - base);

        // Cooperative load of this J tile into shared memory
        for (int t = tid; t < tileCount; t += GPU_THREAD_BLOCK_SIZE)
        {
            const int gj = base + t;
            const auto pj = d_parts_send[gj].p_data;
            s_pos4[t] = pj.x_h;              // (xj,yj,zj,hj)  (unshifted)
            s_vel4[t] = pj.vx_m;             // (vxj,vyj,vzj,mj)
            s_rac4[t] = pj.rho_avisc_u_c;    // (rhoj,aviscj,energyj,cj)
        }
        __syncthreads();

        if (i_in_range)
        {
#pragma unroll 4
            for (int t = 0; t < tileCount; ++t)
            {
                const int j_idx = base + t;
                if (j_idx == i_idx) continue; // self for self-pairs

                // Unpack J tile
                const float4 pj_pos = s_pos4[t];
                const float4 pj_vel = s_vel4[t];
                const float4 pj_rac = s_rac4[t];

                const float xj = (float)(pj_pos.x - (float)shift_j_d.x);
                const float yj = (float)(pj_pos.y - (float)shift_j_d.y);
                const float zj = (float)(pj_pos.z - (float)shift_j_d.z);
                // const float hj = pj_pos.w; // (hj not used in this gradient kernel)

                const float vxj = pj_vel.x, vyj = pj_vel.y, vzj = pj_vel.z, mj = pj_vel.w;

                const float rhoj    = pj_rac.x;
                const float aviscj  = pj_rac.y;
                const float energyj = pj_rac.z;
                const float cj      = pj_rac.w;

                // Geometry
                const float xij = xi - xj;
                const float yij = yi - yj;
                const float zij = zi - zj;

                const float r2  = fmaf(xij, xij, fmaf(yij, yij, zij * zij));
                if (!(r2 < hig2)) continue;  // kernel support on h_i only (matches your code)

                // Distances / normals
                const float inv_r = rsqrtf(r2 + eps);
                const float r     = r2 * inv_r;

                // Cosmology-adjusted dv·r
                const float dvx  = vxi - vxj;
                const float dvy  = vyi - vyj;
                const float dvz  = vzi - vzj;
                const float dvdr = fmaf(dvx, xij, fmaf(dvy, yij, dvz * zij));
                const float dvdr_Hubble = dvdr + a2_Hubble * r2;

                // Approaching?
                const float omega_ij = fminf(dvdr_Hubble, 0.f);
                const float mu_ij    = fac_mu * inv_r * omega_ij; // <= 0

                // Signal velocity (update running max across neighbors; init was vsigi)
                const float new_v_sig = ci + cj - CONST_VISCOSITY_BETA * mu_ij;
                res_vsig_lapu_avisci.x = fmaxf(res_vsig_lapu_avisci.x, new_v_sig);

                // Kernel (derivative wrt r/hi); only wi_dx needed here
                float wi, wi_dx;
                const float ui = r * hi_inv;
                d_kernel_deval(ui, &wi, &wi_dx);

                // Laplacian(u) accumulation
                // delta_u_factor = (u_i - u_j) / r
                const float delta_u_factor = (energyi - energyj) * inv_r;
                // + mj * (Δu / r) * wi_dx / rhoj
                res_vsig_lapu_avisci.y += mj * delta_u_factor * wi_dx * (1.0f / rhoj);

                // Max alpha_visc from neighbors (used downstream)
                res_vsig_lapu_avisci.z = fmaxf(res_vsig_lapu_avisci.z, aviscj);
            }
        }

        __syncthreads(); // all threads reach the same number of barriers
    }

    // Atomics only for active i (semantics preserved)
    if (i_in_range) {
        atomicAdd(&d_parts_recv[i_idx].vsig_lapu_aviscmax.x, res_vsig_lapu_avisci.x);
        atomicAdd(&d_parts_recv[i_idx].vsig_lapu_aviscmax.y, res_vsig_lapu_avisci.y);
        atomicMaxFloat(&d_parts_recv[i_idx].vsig_lapu_aviscmax.z, res_vsig_lapu_avisci.z);
    }
}

__global__ __launch_bounds__(GPU_THREAD_BLOCK_SIZE, 2)
void cuda_launch_gradient_tiled_noasync(
    const struct gpu_part_send_g* __restrict__ d_parts_send,
    struct gpu_part_recv_g*      __restrict__ d_parts_recv,
    const float d_a, const float d_H,
    const int4* __restrict__ d_cell_i_j_start_end,
    const int2* __restrict__ d_block_leaf_id,
    const double3 space_dim)
{
    const int bid     = blockIdx.x;
    const int leafid  = d_block_leaf_id[bid].x;
    const int bid_0   = d_block_leaf_id[bid].y;
    int4 cell_se      = d_cell_i_j_start_end[leafid];

    const int ci_start = cell_se.x;
    const int ci_end   = cell_se.y; // cell position at [ci_end-1]
    const int cj_start = cell_se.z;
    const int cj_end   = cell_se.w;

    const int b_id_local = bid - bid_0;
    const int tid        = threadIdx.x;

    // Periodic shift (same as your other kernels)
    const auto ci_loc = d_parts_send[ci_end - 1].c_loc;
    const auto cj_loc = d_parts_send[cj_end - 1].c_loc;

    double3 shift = {0.0, 0.0, 0.0};
    const double distx = cj_loc.x.x - ci_loc.x.x;
    const double disty = cj_loc.x.y - ci_loc.x.y;
    const double distz = cj_loc.x.z - ci_loc.x.z;

    if (distx < -space_dim.x * 0.5)      shift.x =  space_dim.x;
    else if (distx >  space_dim.x * 0.5)  shift.x = -space_dim.x;
    if (disty < -space_dim.y * 0.5)      shift.y =  space_dim.y;
    else if (disty >  space_dim.y * 0.5)  shift.y = -space_dim.y;
    if (distz < -space_dim.z * 0.5)      shift.z =  space_dim.z;
    else if (distz >  space_dim.z * 0.5)  shift.z = -space_dim.z;

    const double3 shift_i_res = {shift.x + cj_loc.x.x, shift.y + cj_loc.x.y, shift.z + cj_loc.x.z};
    const double3 shift_j_res = {cj_loc.x.x, cj_loc.x.y, cj_loc.x.z};

    // Pass 1: ci <- cj (exclude the cell-position slot at end-1)
    process_range_gradient_tiled_noasync(
        d_parts_send, d_parts_recv,
        ci_start, ci_end - 1,
        cj_start, cj_end - 1,
        shift_i_res, shift_j_res,
        b_id_local, tid,
        d_a, d_H
    );

    // Pass 2: cj <- ci (only if not self)
    if (ci_start != cj_start) {
        const double3 shift_ii_res = {cj_loc.x.x, cj_loc.x.y, cj_loc.x.z};
        const double3 shift_jj_res = {shift.x + cj_loc.x.x, shift.y + cj_loc.x.y, shift.z + cj_loc.x.z};

        process_range_gradient_tiled_noasync(
            d_parts_send, d_parts_recv,
            cj_start, cj_end - 1,
            ci_start, ci_end - 1,
            shift_ii_res, shift_jj_res,
            b_id_local, tid,
            d_a, d_H
        );
    }
}

void gpu_launch_gradient_tiled_noasync(
    const struct gpu_part_send_g* __restrict__ d_parts_send,
    struct gpu_part_recv_g*      __restrict__ d_parts_recv,
    const float d_a, const float d_H,
    int num_blocks_x,
    const int4* __restrict__ d_cell_i_j_start_end,
    const int2* __restrict__ d_block_leaf_id,
    const double3 space_dim,
    cudaStream_t stream)
{
    // Shared memory: pos4 + vel4 + rac4
    const size_t shmem = TILE_J * (sizeof(float4) * 3);  // 3072 bytes when TILE_J=64

    cuda_launch_gradient_tiled_noasync<<<num_blocks_x, GPU_THREAD_BLOCK_SIZE, shmem, stream>>>(
        d_parts_send, d_parts_recv, d_a, d_H,
        d_cell_i_j_start_end, d_block_leaf_id, space_dim);
}

/**
 * @brief Launch the force computation on the GPU for a bundle of leaf cells.
 *
 * @param d_parts_send array on device containing particle data
 * @param d_parts_recv array on device to write results into
 * @param d_a current cosmological scale factor
 * @param d_H current Hubble constant
 * @param stream cuda stream to use
 * @param num_blockx_x number of thread blocks to use in x-dimension
 * @param num_blockx_y number of thread blocks to use in y-dimension
 * @param bundle_first_part index of first particle of this bundle in the
 * d_parts_* arrays
 * @param bundle_n_parts nr of particles in this bundle
 */
void gpu_launch_unique_force(const struct gpu_part_send_f *__restrict__ d_parts_send,
                        struct gpu_part_recv_f *__restrict__ d_parts_recv,
                        const float d_a, const float d_H,
                        const int num_blocks_x,
                        const int4 *__restrict__ d_cell_i_j_start_end,
                        const int2 *__restrict__ d_block_leaf_id,
                        const double3 space_dim, cudaStream_t stream) {

  /* TODO: Do we want to allocate shared memory here? */
  cuda_launch_unique_force<<<num_blocks_x, GPU_THREAD_BLOCK_SIZE, 0, stream>>>(
      d_parts_send, d_parts_recv, d_a, d_H,
      d_cell_i_j_start_end, d_block_leaf_id, space_dim);
}

/**
 * @brief Launch the force computation on the GPU for a bundle of leaf cells.
 *
 * @param d_parts_send array on device containing particle data
 * @param d_parts_recv array on device to write results into
 * @param d_a current cosmological scale factor
 * @param d_H current Hubble constant
 * @param stream cuda stream to use
 * @param num_blockx_x number of thread blocks to use in x-dimension
 * @param num_blockx_y number of thread blocks to use in y-dimension
 * @param bundle_first_part index of first particle of this bundle in the
 * d_parts_* arrays
 * @param bundle_n_parts nr of particles in this bundle
 */
void gpu_launch_force(const struct gpu_part_send_f *__restrict__ d_parts_send,
                      struct gpu_part_recv_f *__restrict__ d_parts_recv,
                      const float d_a, const float d_H, cudaStream_t stream,
                      const int num_blocks_x, const int num_blocks_y,
                      const int bundle_first_part, const int bundle_n_parts) {

  /* TODO: Do we want to allocate shared memory here? */
  cuda_launch_force<<<num_blocks_x, GPU_THREAD_BLOCK_SIZE, 0, stream>>>(
      d_parts_send, d_parts_recv, d_a, d_H, bundle_first_part, bundle_n_parts);
}

//Safe version////////////////

// ===== Device functions / constants you already have =====
__device__ void d_kernel_deval(float u, float* w, float* wdx);
__device__ float d_pow_dimension_plus_one(float x);
__device__ float d_pow_three_gamma_minus_five_over_two(float a);

__device__ __forceinline__
void process_range_force_tiled_noasync(
    const struct gpu_part_send_f* __restrict__ d_parts_send,
    struct gpu_part_recv_f*      __restrict__ d_parts_recv,
    // ranges (end_excl excludes the cell-position slot at [end-1])
    int i_start, int i_end_excl,
    int j_start, int j_end_excl,
    // periodic shifts
    const double3 shift_i_d, const double3 shift_j_d,
    // mapping
    int b_id_local, int tid,
    // cosmology
    float d_a, float d_H)
{
    // Shared memory tile for J: pos/h, vel/m, f/bals/rho/p, c/u/avisc/adiff
    extern __shared__ unsigned char smem[];
    float4* s_pos4  = reinterpret_cast<float4*>(smem);                      // [TILE_J]
    float4* s_vel4  = reinterpret_cast<float4*>(s_pos4  + TILE_J);          // [TILE_J]
    float4* s_fbrp4 = reinterpret_cast<float4*>(s_vel4  + TILE_J);          // [TILE_J]
    float4* s_cuid4 = reinterpret_cast<float4*>(s_fbrp4 + TILE_J);          // [TILE_J]

    // Map this thread to its i-particle (keep all threads in lockstep → no early return)
    const int  i_idx     = b_id_local * GPU_THREAD_BLOCK_SIZE + tid + i_start;
    const bool i_in_range  = (i_idx < i_end_excl);

    // Per-i data (only loaded if active)
    float xi=0.f, yi=0.f, zi=0.f, hi=1.f;
    float vxi=0.f, vyi=0.f, vzi=0.f, mi=1.f;
    float fi=0.f, balsi=0.f, rhoi=1.f, pressurei=0.f;
    float ci=0.f, energyi=0.f, avisci=0.f, adiffi=0.f;
    int   tbj=0, min_ngb_tbi=0;     // follows your original variable usage

    float hi_inv=1.f, hid_inv=1.f, mi_inv=1.f, rhoi_inv=1.f, rhoi_inv2=1.f, hig2=0.f;

    if (i_in_range) {
        const auto pi = d_parts_send[i_idx].p_data;

        xi = (float)(pi.x_h.x - shift_i_d.x);
        yi = (float)(pi.x_h.y - shift_i_d.y);
        zi = (float)(pi.x_h.z - shift_i_d.z);
        hi = (float)(pi.x_h.w);

        vxi = pi.vx_m.x;  vyi = pi.vx_m.y;  vzi = pi.vx_m.z;  mi  = pi.vx_m.w;

        fi = pi.f_bals_rho_p.x;  balsi = pi.f_bals_rho_p.y;
        rhoi = pi.f_bals_rho_p.z; pressurei = pi.f_bals_rho_p.w;

        ci = pi.c_u_avisc_adiff.x;     energyi = pi.c_u_avisc_adiff.y;
        avisci = pi.c_u_avisc_adiff.z; adiffi  = pi.c_u_avisc_adiff.w;

        tbj         = pi.timebin_minngbtimebin_pjs_pje.x;
        min_ngb_tbi = pi.timebin_minngbtimebin_pjs_pje.y;

        hi_inv   = 1.0f / hi;
        hid_inv  = d_pow_dimension_plus_one(hi_inv);
        mi_inv   = 1.0f / mi;
        rhoi_inv = 1.0f / rhoi;
        rhoi_inv2= rhoi_inv * rhoi_inv;
        hig2     = (hi * hi) * kernel_gamma2;
    }

    // Accumulators
    float3 res_ahydro  = {0.f, 0.f, 0.f};
    float2 res_udt_hdt = {0.f, 0.f};
    int    res_min_ngb_timebin = i_in_range ? min_ngb_tbi : 0;

    // Cosmology (same for all threads; computing per-thread is fine)
    const float fac_mu    = d_pow_three_gamma_minus_five_over_two(d_a);
    const float a2_Hubble = d_a * d_a * d_H;

    constexpr float eps = 1e-24f;

    // ---- Tile over J ----
    for (int base = j_start; base < j_end_excl; base += TILE_J)
    {
        const int tileCount = min(TILE_J, j_end_excl - base);

        // Cooperative load of this J tile into shared memory
        for (int t = tid; t < tileCount; t += GPU_THREAD_BLOCK_SIZE)
        {
            const int gj = base + t;
            const auto pj = d_parts_send[gj].p_data;
            s_pos4[t]  = pj.x_h;             // (xj,yj,zj,hj) — unshifted
            s_vel4[t]  = pj.vx_m;            // (vxj,vyj,vzj,mj)
            s_fbrp4[t] = pj.f_bals_rho_p;    // (fj,balsj,rhoj,pressurej)
            s_cuid4[t] = pj.c_u_avisc_adiff; // (cj,energyj,aviscj,adiffj)
        }
        __syncthreads();

        if (i_in_range)
        {
#pragma unroll 4
            for (int t = 0; t < tileCount; ++t)
            {
                const int j_idx = base + t;
                if (j_idx == i_idx) continue; // self for self-pairs

                // Unpack J tile
                const float4 pj_pos  = s_pos4[t];
                const float4 pj_vel  = s_vel4[t];
                const float4 pj_fbrp = s_fbrp4[t];
                const float4 pj_cuid = s_cuid4[t];

                const float xj = (float)(pj_pos.x - (float)shift_j_d.x);
                const float yj = (float)(pj_pos.y - (float)shift_j_d.y);
                const float zj = (float)(pj_pos.z - (float)shift_j_d.z);
                const float hj = pj_pos.w;

                const float vxj = pj_vel.x, vyj = pj_vel.y, vzj = pj_vel.z, mj = pj_vel.w;

                const float fj = pj_fbrp.x, balsj = pj_fbrp.y;
                const float rhoj = pj_fbrp.z, pressurej = pj_fbrp.w;

                const float cj  = pj_cuid.x, energyj = pj_cuid.y;
                const float aviscj = pj_cuid.z, adiffj = pj_cuid.w;

                // Geometry
                const float xij = xi - xj;
                const float yij = yi - yj;
                const float zij = zi - zj;

                const float r2  = fmaf(xij, xij, fmaf(yij, yij, zij * zij));
                const float hjg2= (hj * hj) * kernel_gamma2;

                if (!((r2 < hig2) || (r2 < hjg2))) continue;

                const float inv_r = rsqrtf(r2 + eps);
                const float r     = r2 * inv_r;

                // Kernels for i and j
                float wi, wi_dx, wj, wj_dx;

                const float ui = r * hi_inv;   // r / hi
                d_kernel_deval(ui, &wi, &wi_dx);
                const float wi_dr = hid_inv * wi_dx;

                const float hj_inv  = 1.0f / hj;
                const float hjd_inv = d_pow_dimension_plus_one(hj_inv);
                const float uj      = r * hj_inv; // r / hj
                d_kernel_deval(uj, &wj, &wj_dx);
                const float wj_dr = hjd_inv * wj_dx;

                // Velocity diffs
                const float dvx = vxi - vxj;
                const float dvy = vyi - vyj;
                const float dvz = vzi - vzj;

                const float dvdr = fmaf(dvx, xij, fmaf(dvy, yij, dvz * zij)); // dv · r

                // Hubble augmentation for dv·r
                const float dvdr_Hubble = dvdr + a2_Hubble * r2;

                // Are they approaching?
                const float omega_ij = fminf(dvdr_Hubble, 0.f);
                const float mu_ij    = fac_mu * inv_r * omega_ij; // <= 0

                // Signal velocity and grad-h terms
                const float v_sig = ci + cj - const_viscosity_beta * mu_ij;

                // NOTE: f_ij = 1 - fi/mj ; f_ji = 1 - fj/mi
                const float f_ij = 1.f - fi * (1.f / mj);
                const float f_ji = 1.f - fj * mi_inv;

                // Viscosity
                const float rhoij      = rhoi + rhoj;
                const float rhoij_inv  = 1.f / rhoij;
                const float alpha      = avisci + aviscj;
                const float visc       = -0.25f * alpha * v_sig * mu_ij * (balsi + balsj) * rhoij_inv;

                const float visc_acc_term = 0.5f * visc * (wi_dr * f_ij + wj_dr * f_ji) * inv_r;

                // Pressure
                const float rhoj2         = rhoj * rhoj;
                const float rhoj_inv      = 1.f / rhoj;
                const float P_over_rho2_i = pressurei * rhoi_inv2 * f_ij;
                const float P_over_rho2_j = pressurej * (1.f / rhoj2) * f_ji;

                const float sph_acc_term  = (P_over_rho2_i * wi_dr + P_over_rho2_j * wj_dr) * inv_r;

                const float acc = sph_acc_term + visc_acc_term;

                // Acceleration accumulation
                res_ahydro.x -= mj * acc * xij;
                res_ahydro.y -= mj * acc * yij;
                res_ahydro.z -= mj * acc * zij;

                // du/dt terms
                const float sph_du_term_i = P_over_rho2_i * dvdr * inv_r * wi_dr;
                const float visc_du_term  = 0.5f * visc_acc_term * dvdr_Hubble;

                // Diffusion
                float alpha_diff = (pressurei * adiffi + pressurej * adiffj) / (pressurei + pressurej);
                // if (fabsf(pressurei + pressurej) < 1e-10f) alpha_diff = 0.f; // optional

                const float v_diff = alpha_diff * 0.5f *
                    (sqrtf(2.f * fabsf(pressurei - pressurej) * rhoij_inv) +
                     fabsf(fac_mu * inv_r * dvdr_Hubble));

                const float diff_du_term = v_diff * (energyi - energyj) *
                    (f_ij * wi_dr * rhoi_inv + f_ji * wj_dr * rhoj_inv);

                const float du_dt_i = sph_du_term_i + visc_du_term + diff_du_term;

                // Accumulate energy & h-derivative
                res_udt_hdt.x += du_dt_i * mj;
                res_udt_hdt.y -= mj * dvdr * inv_r * rhoj_inv * wi_dr;
            }
        }

        __syncthreads(); // all threads must reach the same number of barriers
    }

    // Min neighbor timebin (your original logic used tbj from i)
    if (i_in_range && tbj > 0) {
        res_min_ngb_timebin = min(res_min_ngb_timebin, tbj);
    }

    // Atomics only for active i
    if (i_in_range) {
        atomicAdd(&d_parts_recv[i_idx].udt_hdt.x, res_udt_hdt.x);
        atomicAdd(&d_parts_recv[i_idx].udt_hdt.y, res_udt_hdt.y);

        // If timebin is zero, set it; then take min
        atomicCAS(&d_parts_recv[i_idx].minngbtb, 0, res_min_ngb_timebin);
        atomicMin(&d_parts_recv[i_idx].minngbtb, res_min_ngb_timebin);

        atomicAdd(&d_parts_recv[i_idx].a_hydro.x, res_ahydro.x);
        atomicAdd(&d_parts_recv[i_idx].a_hydro.y, res_ahydro.y);
        atomicAdd(&d_parts_recv[i_idx].a_hydro.z, res_ahydro.z);
    }
}

__global__ __launch_bounds__(GPU_THREAD_BLOCK_SIZE, 2)
void cuda_launch_force_tiled_noasync(
    const struct gpu_part_send_f* __restrict__ d_parts_send,
    struct gpu_part_recv_f*      __restrict__ d_parts_recv,
    const float d_a, const float d_H,
    const int4* __restrict__ d_cell_i_j_start_end,
    const int2* __restrict__ d_block_leaf_id,
    const double3 space_dim)
{
    const int bid     = blockIdx.x;
    const int leafid  = d_block_leaf_id[bid].x;
    const int bid_0   = d_block_leaf_id[bid].y;
    int4 cell_se      = d_cell_i_j_start_end[leafid];

    const int ci_start = cell_se.x;
    const int ci_end   = cell_se.y; // cell position at [ci_end-1]
    const int cj_start = cell_se.z;
    const int cj_end   = cell_se.w;

    const int b_id_local = bid - bid_0;
    const int tid = threadIdx.x;

    // Periodic shift (same as your density wrapper)
    const auto ci_loc = d_parts_send[ci_end - 1].c_loc;
    const auto cj_loc = d_parts_send[cj_end - 1].c_loc;

    double3 shift = {0.0, 0.0, 0.0};
    const double distx = cj_loc.x.x - ci_loc.x.x;
    const double disty = cj_loc.x.y - ci_loc.x.y;
    const double distz = cj_loc.x.z - ci_loc.x.z;

    if (distx < -space_dim.x * 0.5)      shift.x =  space_dim.x;
    else if (distx >  space_dim.x * 0.5)  shift.x = -space_dim.x;
    if (disty < -space_dim.y * 0.5)      shift.y =  space_dim.y;
    else if (disty >  space_dim.y * 0.5)  shift.y = -space_dim.y;
    if (distz < -space_dim.z * 0.5)      shift.z =  space_dim.z;
    else if (distz >  space_dim.z * 0.5)  shift.z = -space_dim.z;

    const double3 shift_i_res = {shift.x + cj_loc.x.x, shift.y + cj_loc.x.y, shift.z + cj_loc.x.z};
    const double3 shift_j_res = {cj_loc.x.x, cj_loc.x.y, cj_loc.x.z};

    // Pass 1: ci <- cj (exclude cell-position slot at end-1)
    process_range_force_tiled_noasync(
        d_parts_send, d_parts_recv,
        ci_start, ci_end - 1,
        cj_start, cj_end - 1,
        shift_i_res, shift_j_res,
        b_id_local, tid,
        d_a, d_H
    );

    // Pass 2: cj <- ci (only if not self)
    if (ci_start != cj_start) {
        const double3 shift_ii_res = {cj_loc.x.x, cj_loc.x.y, cj_loc.x.z};
        const double3 shift_jj_res = {shift.x + cj_loc.x.x, shift.y + cj_loc.x.y, shift.z + cj_loc.x.z};

        process_range_force_tiled_noasync(
            d_parts_send, d_parts_recv,
            cj_start, cj_end - 1,
            ci_start, ci_end - 1,
            shift_ii_res, shift_jj_res,
            b_id_local, tid,
            d_a, d_H
        );
    }
}

void gpu_launch_force_tiled_noasync(
    const struct gpu_part_send_f* __restrict__ d_parts_send,
    struct gpu_part_recv_f*      __restrict__ d_parts_recv,
    const float d_a, const float d_H,
    int num_blocks_x,
    const int4* __restrict__ d_cell_i_j_start_end,
    const int2* __restrict__ d_block_leaf_id,
    const double3 space_dim,
    cudaStream_t stream)
{
    // Shared memory: pos4 + vel4 + fbrp4 + cuid4
    const size_t shmem = TILE_J * (sizeof(float4) * 4); // 4096 B when TILE_J=64

    cuda_launch_force_tiled_noasync<<<num_blocks_x, GPU_THREAD_BLOCK_SIZE, shmem, stream>>>(
        d_parts_send, d_parts_recv, d_a, d_H,
        d_cell_i_j_start_end, d_block_leaf_id, space_dim);
}

#ifdef __cplusplus
}
#endif
