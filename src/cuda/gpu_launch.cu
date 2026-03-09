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


// Tunables: start with BLOCK_SIZE=128..256, TILE_J=128..256 depending on SM resources.
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef TILE_J
#define TILE_J 16
#endif

//Safe version////////////////
__device__ __forceinline__
void process_range_tiled_noasync(
    const struct gpu_part_send_d* __restrict__ d_parts_send,
    struct gpu_part_recv_d* __restrict__ d_parts_recv,
    int i_start, int i_end_excl,
    int j_start, int j_end_excl,
    const double3 shift_i_d, const double3 shift_j_d,
    int b_id_local, int tid)
{
    // Shared memory for one tile of J: positions and velocities
    extern __shared__ __align__(16) unsigned char smem[];
    float4* s_pos4 = reinterpret_cast<float4*>(smem);                   // [TILE_J]
    float4* s_vel4 = reinterpret_cast<float4*>(s_pos4 + TILE_J);        // [TILE_J]

    // Map this thread to its i-particle
    const int i_idx = b_id_local * BLOCK_SIZE + tid + i_start;
    const bool i_active = (i_idx < i_end_excl);

    // Declare i-data, initialize safely; only load if active
    float xi = 0.f, yi = 0.f, zi = 0.f, hi = 1.f;
    float vxi = 0.f, vyi = 0.f, vzi = 0.f;
    float hig2 = 0.f, hi_inv = 1.f;

    if (i_active) {
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
        const int tileCount = min(TILE_J, j_end_excl - base);

        // Cooperative load of this J tile into shared memory
        for (int t = tid; t < tileCount; t += BLOCK_SIZE)
        {
            const int gj = base + t;
            s_pos4[t] = d_parts_send[gj].p_data.x_h;  // (x,y,z,hj) unshifted
            s_vel4[t] = d_parts_send[gj].p_data.vx_m; // (vx,vy,vz,m)
        }
        __syncthreads();

        // Compute only if this thread owns a valid i
        if (i_active)
        {
#pragma unroll 4
            for (int t = 0; t < tileCount; ++t)
            {
                const int j_idx = base + t;
                if (j_idx == i_idx) continue;  // self for self-pairs

                const float4 pj_pos = s_pos4[t]; // unshifted
                const float4 pj_vel = s_vel4[t];

                // Apply j shift on-the-fly
                const float xij = xi - (pj_pos.x - (float)shift_j_d.x);
                const float yij = yi - (pj_pos.y - (float)shift_j_d.y);
                const float zij = zi - (pj_pos.z - (float)shift_j_d.z);

                const float r2 = fmaf(xij, xij, fmaf(yij, yij, zij * zij));
                if (r2 >= hig2) continue;

                const float inv_r = rsqrtf(r2 + eps);
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
    if (i_active) {
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

__global__ __launch_bounds__(BLOCK_SIZE, 2)
void cuda_launch_density_tiled_noasync(
    const struct gpu_part_send_d* __restrict__ d_parts_send,
    struct gpu_part_recv_d* __restrict__ d_parts_recv,
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
    const int ci_end   = cell_se.y; // cell pos at index [ci_end-1]
    const int cj_start = cell_se.z;
    const int cj_end   = cell_se.w;

    const int b_id_local = bid - bid_0;
    const int tid = threadIdx.x;

    // Periodic shift (as in your original code)
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
    process_range_tiled_noasync(
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

        process_range_tiled_noasync(
            d_parts_send, d_parts_recv,
            cj_start, cj_end - 1,
            ci_start, ci_end - 1,
            shift_ii_res, shift_jj_res,
            b_id_local, tid
        );
    }
}


void gpu_launch_density_tiled_noasync(
    const struct gpu_part_send_d* __restrict__ d_parts_send,
    struct gpu_part_recv_d* __restrict__ d_parts_recv,
    const float d_a, const float d_H,
    int num_blocks_x,
    const int4* __restrict__ d_cell_i_j_start_end,
    const int2* __restrict__ d_block_leaf_id,
    const double3 space_dim,
    cudaStream_t stream)
{
    // Shared memory: one buffer for pos4 + one buffer for vel4
    const size_t shmem = TILE_J * (sizeof(float4) + sizeof(float4)); // 2048 bytes when TILE_J=64

    cuda_launch_density_tiled_noasync<<<num_blocks_x, BLOCK_SIZE, shmem, stream>>>(
        d_parts_send, d_parts_recv, d_a, d_H,
        d_cell_i_j_start_end, d_block_leaf_id, space_dim);
}
///////////////////////////

// Kernel with shared-memory tiling of the J cell for both passes (ci<-cj) and (cj<-ci)
__global__ void cuda_launch_density_tiled(
    const struct gpu_part_send_d* __restrict__ d_parts_send,
    struct gpu_part_recv_d* __restrict__ d_parts_recv,
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
    const int ci_end   = cell_se.y; // last index stores cell loc; effective end is ci_end-1
    const int cj_start = cell_se.z;
    const int cj_end   = cell_se.w;

    const int b_id_local = bid - bid_0;

    // Compute shifts (periodic) once per block
    // Load cell positions (stored at last slot of each cell range)
    const struct gpu_cell_pos ci_loc = d_parts_send[ci_end - 1].c_loc;
    const struct gpu_cell_pos cj_loc = d_parts_send[cj_end - 1].c_loc;

    double3 shift = {0.0, 0.0, 0.0};
    const double distx = cj_loc.x.x - ci_loc.x.x;
    const double disty = cj_loc.x.y - ci_loc.x.y;
    const double distz = cj_loc.x.z - ci_loc.x.z;

    if (distx < -space_dim.x * 0.5) shift.x =  space_dim.x;
    else if (distx >  space_dim.x * 0.5) shift.x = -space_dim.x;

    if (disty < -space_dim.y * 0.5) shift.y =  space_dim.y;
    else if (disty >  space_dim.y * 0.5) shift.y = -space_dim.y;

    if (distz < -space_dim.z * 0.5) shift.z =  space_dim.z;
    else if (distz >  space_dim.z * 0.5) shift.z = -space_dim.z;

    // Shifts for each pass
    const double3 shift_i_res = {shift.x + cj_loc.x.x, shift.y + cj_loc.x.y, shift.z + cj_loc.x.z};
    const double3 shift_j_res = {cj_loc.x.x, cj_loc.x.y, cj_loc.x.z};

    // Shared memory tiles for J particles
    extern __shared__ unsigned char smem[];
    float4* s_pos4 = reinterpret_cast<float4*>(smem);                   // size TILE_J
    float4* s_vel4 = reinterpret_cast<float4*>(s_pos4 + TILE_J);        // size TILE_J

    const int tid = threadIdx.x;

    auto process_i_range_against_j_range = [&](int i_start, int i_end_excl,
                                               int j_start, int j_end_excl,
                                               const double3 shift_i_d, const double3 shift_j_d)
    {
        const int i_idx = b_id_local * BLOCK_SIZE + tid + i_start;
        if (i_idx >= i_end_excl) return;

        // Load i once
        const auto pi = d_parts_send[i_idx].p_data;
        const float xi = (float)(pi.x_h.x - shift_i_d.x);
        const float yi = (float)(pi.x_h.y - shift_i_d.y);
        const float zi = (float)(pi.x_h.z - shift_i_d.z);
        const float hi = (float)(pi.x_h.w);

        const float vxi = pi.vx_m.x;
        const float vyi = pi.vx_m.y;
        const float vzi = pi.vx_m.z;

        const float hig2   = (hi * hi) * kernel_gamma2;
        const float hi_inv = 1.0f / hi;

        float4 res_rho = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 res_rot = make_float4(0.f, 0.f, 0.f, 0.f);

        constexpr float eps = 1e-24f;

        // Tile over J cell
        for (int base = j_start; base < j_end_excl; base += TILE_J)
        {
            const int tileCount = min(TILE_J, j_end_excl - base);

            // Cooperative load of J tile into shared memory
            for (int t = tid; t < tileCount; t += BLOCK_SIZE)
            {
                const auto pj = d_parts_send[base + t].p_data;
                // Pack pos - shift_j and h in w
                s_pos4[t] = make_float4(
                    (float)(pj.x_h.x - shift_j_d.x),
                    (float)(pj.x_h.y - shift_j_d.y),
                    (float)(pj.x_h.z - shift_j_d.z),
                    pj.x_h.w
                );
                // Pack (vx, vy, vz, m)
                s_vel4[t] = pj.vx_m;
            }
            __syncthreads();

            // Compute interactions with staged tile
#pragma unroll 4
            for (int t = 0; t < tileCount; ++t)
            {
                const int j_idx = base + t;

                // Self-skip only relevant for self tasks
                if (j_idx == i_idx) continue;

                const float4 pj_pos = s_pos4[t];
                const float4 pj_vel = s_vel4[t];

                const float xij = xi - pj_pos.x;
                const float yij = yi - pj_pos.y;
                const float zij = zi - pj_pos.z;

                const float r2 = fmaf(xij, xij, fmaf(yij, yij,   zij * zij));
                if (r2 >= hig2) continue;

                const float vxj = pj_vel.x;
                const float vyj = pj_vel.y;
                const float vzj = pj_vel.z;
                const float mj  = pj_vel.w;

                const float inv_r = rsqrtf(r2 + eps);
                const float r     = r2 * inv_r;

                const float ui = r * hi_inv;

                float wi, wi_dx;
                d_kernel_deval(ui, &wi, &wi_dx);

                const float tmp = (hydro_dimension * wi + ui * wi_dx);

                res_rho.x += mj * wi;
                res_rho.y -= mj * tmp;
                res_rho.z += wi;
                res_rho.w -= tmp;

                const float faci = mj * wi_dx * inv_r;

                const float dvx = vxi - vxj;
                const float dvy = vyi - vyj;
                const float dvz = vzi - vzj;

                const float dvdr   = fmaf(dvx, xij, fmaf(dvy, yij, dvz * zij));
                const float curlrx = fmaf(dvy, zij, -dvz * yij);
                const float curlry = fmaf(dvz, xij, -dvx * zij);
                const float curlrz = fmaf(dvx, yij, -dvy * xij);

                res_rot.x = fmaf(faci,  curlrx, res_rot.x);
                res_rot.y = fmaf(faci,  curlry, res_rot.y);
                res_rot.z = fmaf(faci,  curlrz, res_rot.z);
                res_rot.w = fmaf(-faci, dvdr,    res_rot.w);
            }
            __syncthreads();
        }

        // Atomically add results for this i
        atomicAdd(&d_parts_recv[i_idx].rho_rhodh_wcount_wcount_dh.x, res_rho.x);
        atomicAdd(&d_parts_recv[i_idx].rho_rhodh_wcount_wcount_dh.y, res_rho.y);
        atomicAdd(&d_parts_recv[i_idx].rho_rhodh_wcount_wcount_dh.z, res_rho.z);
        atomicAdd(&d_parts_recv[i_idx].rho_rhodh_wcount_wcount_dh.w, res_rho.w);

        atomicAdd(&d_parts_recv[i_idx].rot_vx_div_v.x, res_rot.x);
        atomicAdd(&d_parts_recv[i_idx].rot_vx_div_v.y, res_rot.y);
        atomicAdd(&d_parts_recv[i_idx].rot_vx_div_v.z, res_rot.z);
        atomicAdd(&d_parts_recv[i_idx].rot_vx_div_v.w, res_rot.w);
    };

    // Pass 1: ci <- cj
    {
        const int i_end_excl = ci_end - 1;
        const int j_end_excl = cj_end - 1;
        process_i_range_against_j_range(ci_start, i_end_excl, cj_start, j_end_excl,
                                        shift_i_res, shift_j_res);
    }

    // Pass 2: cj <- ci (only if not self)
    if (ci_start != cj_start)
    {
        // swap roles
        const double3 shift_ii_res = {cj_loc.x.x, cj_loc.x.y, cj_loc.x.z};
        const double3 shift_jj_res = {shift.x + cj_loc.x.x, shift.y + cj_loc.x.y, shift.z + cj_loc.x.z};

        const int i_end_excl = cj_end - 1;
        const int j_end_excl = ci_end - 1;

        // Reuse the same mapping for thread → i in the swapped range
        process_i_range_against_j_range(cj_start, i_end_excl, ci_start, j_end_excl,
                                        shift_ii_res, shift_jj_res);
    }
}

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

void gpu_launch_density_tiled(const struct gpu_part_send_d* __restrict__ d_parts_send,
                        struct gpu_part_recv_d* __restrict__ d_parts_recv,
                        const float d_a, const float d_H,
                        const int num_blocks_x,
                        const int4* __restrict__ d_cell_i_j_start_end,
                        const int2* __restrict__ d_block_leaf_id,
                        const double3 space_dim,
                        cudaStream_t stream)
{
    // Shared memory size: two float4 tiles of size TILE_J
    const size_t shmem = (sizeof(float4) * TILE_J) * 2;

    cuda_launch_density_tiled<<<num_blocks_x, BLOCK_SIZE, shmem, stream>>>(
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

#ifdef __cplusplus
}
#endif
