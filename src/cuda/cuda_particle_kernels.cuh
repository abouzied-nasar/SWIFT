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
#ifndef CUDA_PARTICLE_KERNELS_CUH
#define CUDA_PARTICLE_KERNELS_CUH

/**
 * @file cuda/cuda_particle_kernels.cuh
 * @brief contains the actual particle interaction kernels executed on device
 * TODO: This needs to become SPH flavour specific. Currently contains SPHENIX.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "cuda_config.h"
#include "device_functions.cuh"
#include "gpu_part_structs.h"
#include "inline.h"

#include <config.h>

/**
 * @brief Naive kernel computing the density interactions of a single particle
 *
 * @param pid index of particle to compute density for in the data arrays
 * @param d_pars_send array of particle data received from CPU
 * @param d_parts_recv array of particle data to write results into
 * @param d_a current cosmological expansion factor
 * @param d_H current Hubble constant
 */
//TODO: When changing the file cuda_particle_kernels.cuh and then recompiling the compiler doesn't realise the file has changed
__device__ __forceinline__ void neighbour_interactions_density(
    const struct gpu_part_send_d* __restrict__ d_parts_send,
    struct gpu_part_recv_d* __restrict__ d_parts_recv,
    int i_start, int i_end,
    int j_start, int j_end,
    const double3 shift_i_d, const double3 shift_j_d,
    int b_id_local, int tid){

  // Shared memory for one tile of J: positions and velocities
  extern __shared__ unsigned char smem[];
  //TODO: Check if this is safe and/or required. We're casting from float4 to float4
  //Also, we need positions to be double down tne road so this may need re-working!
  float4* s_x_h = reinterpret_cast<float4*>(smem);                   // [0 to 2 * GPU_THREAD_BLOCK_SIZE] (xj,yj,zj,hj)
  float4* s_vx_m = reinterpret_cast<float4*>(s_x_h + 2 * GPU_THREAD_BLOCK_SIZE);        // [2 * GPU_THREAD_BLOCK_SIZE to 4 * GPU_THREAD_BLOCK_SIZE] (vx,vy,vz,m)

  // Map this thread to its i-particle
  const int i_id = b_id_local * GPU_THREAD_BLOCK_SIZE + tid + i_start;
  const bool i_in_range = (i_id < i_end);

  // Declare i-data, initialize safely; only load if active
  float xi = 0.f, yi = 0.f, zi = 0.f, hi = 1.f;
  float vxi = 0.f, vyi = 0.f, vzi = 0.f;
  float hig2 = 0.f, hi_inv = 1.f;

  if (i_in_range) {
      const struct gpu_part_data_d pi = d_parts_send[i_id].p_data;
      xi = (float)(pi.x_h.x - shift_i_d.x);
      yi = (float)(pi.x_h.y - shift_i_d.y);
      zi = (float)(pi.x_h.z - shift_i_d.z);
      hi = (float)(pi.x_h.w);

      vxi = pi.vx_m.x;
      vyi = pi.vx_m.y;
      vzi = pi.vx_m.z;

      /* Do some auxiliary computations */
      hig2   = (hi * hi) * kernel_gamma2;
      hi_inv = 1.0f / hi;
  }
  /* Prep output */
  /* rho, rho_dh, wcount, wcount_dh */
  float4 res_rho = make_float4(0.f, 0.f, 0.f, 0.f);
  /* curl of velocity (3 coordinates), velocity divergence */
  float4 res_rot = make_float4(0.f, 0.f, 0.f, 0.f);
  constexpr float eps = 1e-24f;

  // Number of tiles
  const int numTiles = (j_end - j_start + GPU_THREAD_BLOCK_SIZE - 1) / GPU_THREAD_BLOCK_SIZE;

  // === Prefetch tile 0 into buffer 0 ===
  if (numTiles > 0) {
      const int base0      = j_start;
      const int tileCount0 = min(GPU_THREAD_BLOCK_SIZE, j_end - base0);

      for (int t = tid; t < tileCount0; t += GPU_THREAD_BLOCK_SIZE) {
          const int gj = base0 + t;
          __pipeline_memcpy_async(&s_x_h[0 * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.x_h, sizeof(float4));
          __pipeline_memcpy_async(&s_vx_m[0 * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.vx_m, sizeof(float4));
      }
      __pipeline_commit();
  }

  // === Tile over J with prefetch of "next" tile while computing "current" ===
  for (int tile = 0; tile < numTiles; ++tile)
  {
      const int buf       = tile & 1;  // 0 or 1 (ping-pong)
      const int base      = j_start + tile * GPU_THREAD_BLOCK_SIZE;
      const int tileCount = min(GPU_THREAD_BLOCK_SIZE, j_end - base);

      // Make sure the current tile (already committed) is resident in shared memory
      __pipeline_wait_prior(0);
      __syncthreads();

      // --- Kick off prefetch for the next tile (overlaps with compute below) ---
      const int nextTile = tile + 1;
      if (nextTile < numTiles)
      {
        /*If nextTile & 1 == 0. nextTile is even. If nextTile & 1 == 1, nextTile is odd*/
        const int nextBuf  = nextTile & 1;
        const int nextBase = j_start + nextTile * GPU_THREAD_BLOCK_SIZE;
        const int nextCnt  = min(GPU_THREAD_BLOCK_SIZE, j_end - nextBase);

        for (int t = tid; t < nextCnt; t += GPU_THREAD_BLOCK_SIZE) {
          const int gj = nextBase + t;
          __pipeline_memcpy_async(&s_x_h[nextBuf * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.x_h, sizeof(float4));
          __pipeline_memcpy_async(&s_vx_m[nextBuf * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.vx_m, sizeof(float4));
        }
        __pipeline_commit();
      }

      // --- Compute on the current tile (buf) ---
      if (i_in_range)
      {
      /* Start the neighbour interactions */
#pragma unroll 4
          for (int t = 0; t < tileCount; ++t)
          {
              const int j_idx = base + t;
              /* j != pid: Exclude self contribution. This happens at a later step. */
              if (j_idx == i_id) continue;

              /* First, grab handles. */
              const float4 pj_x_h = s_x_h[buf * GPU_THREAD_BLOCK_SIZE + t]; // unshifted
              const float4 pj_vel = s_vx_m[buf * GPU_THREAD_BLOCK_SIZE + t];

              /* Now get stuff done. */
              // Apply j shift on-the-fly
              const float xij = xi - (pj_x_h.x - (float)shift_j_d.x);
              const float yij = yi - (pj_x_h.y - (float)shift_j_d.y);
              const float zij = zi - (pj_x_h.z - (float)shift_j_d.z);

              // fmaf -> fused multiply-add
              const float r2 = fmaf(xij, xij, fmaf(yij, yij, zij * zij));
              if (r2 >= hig2) continue;

              // Clever Co-Pilot
              const float inv_r = rsqrtf(r2 + eps);
              // Very clever Co-Pilot, multiply instead of divide
              /* Recover some data */
              const float r     = r2 * inv_r;
              /* Get the kernel for hi. */
              const float ui    = r * hi_inv;
              float wi, wi_dx;
              d_kernel_deval(ui, &wi, &wi_dx);

              const float mj  = pj_vel.w;
              const float tmp = (hydro_dimension * wi + ui * wi_dx);

              /* Add to sums of rho, rho_dh, wcount and wcount_dh */
              res_rho.x += mj * wi;
              res_rho.y -= mj * tmp;
              res_rho.z += wi;
              res_rho.w -= tmp;

              const float faci = mj * wi_dx * inv_r;

              /* Compute dv dot r */
              const float dvx = vxi - pj_vel.x;
              const float dvy = vyi - pj_vel.y;
              const float dvz = vzi - pj_vel.z;
              const float dvdr   = fmaf(dvx, xij, fmaf(dvy, yij, dvz * zij));
              /* Compute dv cross r */
              const float curlrx = fmaf(dvy, zij, -dvz * yij);
              const float curlry = fmaf(dvz, xij, -dvx * zij);
              const float curlrz = fmaf(dvx, yij, -dvy * xij);

              res_rot.x = fmaf(faci,  curlrx, res_rot.x);
              res_rot.y = fmaf(faci,  curlry, res_rot.y);
              res_rot.z = fmaf(faci,  curlrz, res_rot.z);
              res_rot.w = fmaf(-faci, dvdr,    res_rot.w);
          }
      }/*Loop through parts in cell j and in current tile*/
      // Ensure no thread is still reading from the current buffer before it may be overwritten next
      __syncthreads();
  }

  /* Write results. */
  if (i_in_range) {
      atomicAdd(&d_parts_recv[i_id].rho_rhodh_wcount_wcount_dh.x, res_rho.x);
      atomicAdd(&d_parts_recv[i_id].rho_rhodh_wcount_wcount_dh.y, res_rho.y);
      atomicAdd(&d_parts_recv[i_id].rho_rhodh_wcount_wcount_dh.z, res_rho.z);
      atomicAdd(&d_parts_recv[i_id].rho_rhodh_wcount_wcount_dh.w, res_rho.w);

      atomicAdd(&d_parts_recv[i_id].rot_vx_div_v.x, res_rot.x);
      atomicAdd(&d_parts_recv[i_id].rot_vx_div_v.y, res_rot.y);
      atomicAdd(&d_parts_recv[i_id].rot_vx_div_v.z, res_rot.z);
      atomicAdd(&d_parts_recv[i_id].rot_vx_div_v.w, res_rot.w);
  }

}

__global__ void cuda_kernel_density(
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
    neighbour_interactions_density(
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

        neighbour_interactions_density(
            d_parts_send, d_parts_recv,
            cj_start, cj_end - 1,
            ci_start, ci_end - 1,
            shift_ii_res, shift_jj_res,
            b_id_local, tid
        );
    }
}

/**
 * @brief Naive kernel computing the gradient interactions of a single particle
 *
 * @param pid index of particle to compute density for in the data arrays
 * @param d_pars_send array of particle data received from CPU
 * @param d_parts_recv array of particle data to write results into
 * @param i_start first particle in cell i
 * @param i_end last particle in cell i
 * @param j_start first particle in cell j
 * @param j_end last particle in cell j
 * @param shift_i shifts for particles in cell i
 * @param shift_j shifts for particles in cell j
 * @param b_id_local within the GPU thread blocks acting on this cell what is my id.
 *        Needed to figure out which range of particles each CUDA block will work on
 * @param t_id the current threads id in the list of threads in the block
 * @param d_a current cosmological expansion factor
 * @param d_H current Hubble constant
 */
__device__ __forceinline__ void neighbour_interactions_gradient(
    const struct gpu_part_send_g* __restrict__ d_parts_send,
    struct gpu_part_recv_g*      __restrict__ d_parts_recv,
    int i_start, int i_end, int j_start, int j_end,
    const double3 shift_i_d, const double3 shift_j_d, int b_id_local,
    int tid, float d_a, float d_H){

  /*Declare a variable to use up allocated shared memory*/
  extern __shared__ unsigned char smem[];
  /* TODO: we need positions to be double so this needs re-working in the near future!*/
  /*Assign range of memory to use for x, y, z and h*/
  float4* s_x_h = reinterpret_cast<float4*>(smem);
  /*Assign range of memory to use for velocity (u, v, w) and mass*/
  float4* s_vx_m = reinterpret_cast<float4*>(s_x_h + 2 * GPU_THREAD_BLOCK_SIZE);
  /*Assign range of memory to use for energy u, density rho, speed of sound c, alpha visc*/
  float4* s_u_rho_c_aviscmax = reinterpret_cast<float4*>(s_vx_m + 2 * GPU_THREAD_BLOCK_SIZE);

  /*Map this thread to its i-particle*/
  const int  i_id    = b_id_local * GPU_THREAD_BLOCK_SIZE + tid + i_start;
  /*Is the id in the cell we need to work on?*/
  const bool i_in_range = (i_id < i_end);

  /* Initialise particle i's data */
  float xi=0.f, yi=0.f, zi=0.f, hi=1.f;
  float vxi=0.f, vyi=0.f, vzi=0.f;
  float energyi=0.f, ci=0.f;
  float vsigi=0.f, lapui=0.f, avisc_maxi=0.f;
  float hi_inv=1.f, hig2=0.f;

  /*Do not do any calculation if i_id is not in cell i range of particles*/
  if (i_in_range) {

    /*Load particle i*/
    const struct gpu_part_data_g pi = d_parts_send[i_id].p_data;

    /*Calculate i's position local to the cell*/
    xi = (float)(pi.x_h.x - shift_i_d.x);
    yi = (float)(pi.x_h.y - shift_i_d.y);
    zi = (float)(pi.x_h.z - shift_i_d.z);
    /*i smoothing length*/
    hi = (float)(pi.x_h.w);
    /*Find my velocities, mass not needed for particle i*/
    vxi = pi.vx_m.x;
    vyi = pi.vx_m.y;
    vzi = pi.vx_m.z;

    /*Now get energy i and speed of sound from u_rho_c_aviscmax (u, rho, c, avisc)*/
    energyi = pi.u_rho_c_aviscmax.x;
    ci      = pi.rho_avisc_u_c.w;

    /* Prep output */
    /* Get previous value of avisc_vsig_lapu. vsig and lapu will be incremented while
     * for avisc we want to find the maximum*/
    avisc_maxi  = pi.avisc_vsig_lapu.x;
    /*TODO: Double check whether I've introduced a bug here and if we actually need this.
     * lapui should be set to zero I think since we atomically add them at the end*/
    vsigi       = pi.avisc_vsig_lapu.y;
    lapui       = 0.f//pi.avisc_vsig_lapu.z;

    hi_inv = 1.0f / hi;
    hig2   = (hi * hi) * kernel_gamma2;
  }

  // Accumulators
  // Start from the i-side stored values (as in your original), then update over neighbours
  float3 res_aviscmax_vsig_lapui = {avisc_maxi, vsigi, lapui};

  // Cosmology terms
  const float fac_mu    = d_pow_three_gamma_minus_five_over_two(d_a);
  const float a2_Hubble = d_a * d_a * d_H;

  constexpr float eps = 1e-24f;

  // Number of tiles
  const int numTiles = (j_end - j_start + GPU_THREAD_BLOCK_SIZE - 1) / GPU_THREAD_BLOCK_SIZE;

  // === Prefetch tile 0 into buffer 0 ===
  if (numTiles > 0){
    const int base0      = j_start;
    const int tileCount0 = min(GPU_THREAD_BLOCK_SIZE, j_end - base0);

    for (int t = tid; t < tileCount0; t += GPU_THREAD_BLOCK_SIZE) {
      const int gj = base0 + t;
      __pipeline_memcpy_async(&s_x_h[0 * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.x_h, sizeof(float4));
      __pipeline_memcpy_async(&s_vx_m[0 * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.vx_m, sizeof(float4));
      __pipeline_memcpy_async(&s_u_rho_c_aviscmax[0 * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.u_rho_c_aviscmax, sizeof(float4));
    }
    __pipeline_commit();
  }
  // === Tile over J with prefetch of "next" tile while computing "current" ===
  for (int tile = 0; tile < numTiles; ++tile){

    const int buf       = tile & 1;  // 0 or 1 (ping-pong)
    const int base      = j_start + tile * GPU_THREAD_BLOCK_SIZE;
    const int tileCount = min(GPU_THREAD_BLOCK_SIZE, j_end - base);

    // Make sure the current tile (already committed) is resident in shared memory
    __pipeline_wait_prior(0);
    __syncthreads();

    // --- Kick off prefetch for the next tile (overlaps with compute below) ---
    const int nextTile = tile + 1;
    if (nextTile < numTiles){
      /*If nextTile & 1 == 0. nextTile is even. If nextTile & 1 == 1, nextTile is odd*/
      const int nextBuf  = nextTile & 1;
      const int nextBase = j_start + nextTile * GPU_THREAD_BLOCK_SIZE;
      const int nextCnt  = min(GPU_THREAD_BLOCK_SIZE, j_end - nextBase);

      for (int t = tid; t < nextCnt; t += GPU_THREAD_BLOCK_SIZE) {
        const int gj = nextBase + t;
        __pipeline_memcpy_async(&s_x_h[nextBuf * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.x_h, sizeof(float4));
        __pipeline_memcpy_async(&s_vx_m[nextBuf * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.vx_m, sizeof(float4));
        __pipeline_memcpy_async(&s_u_rho_c_aviscmax[nextBuf * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.u_rho_c_aviscmax, sizeof(float4));
      }
      __pipeline_commit();
    }
    // --- Compute on the current tile (buf) ---
    if (i_in_range){
      /* Start the neighbour interactions */
#pragma unroll 4
      for (int t = 0; t < tileCount; ++t){
        const int j_idx = base + t;
        /* j != pid: Exclude self contribution. This happens at a later step. */
        if (j_idx == i_id) continue;

        /* First, grab handles. */
        const float4 pj_x_h = s_x_h[buf * GPU_THREAD_BLOCK_SIZE + t]; // unshifted
        const float4 pj_vx_m = s_vx_m[buf * GPU_THREAD_BLOCK_SIZE + t];
        /*Get rho, visc, energy and speed of sound*/
        const float4 pj_u_rho_c_aviscmax = s_u_rho_c_aviscmax[buf * GPU_THREAD_BLOCK_SIZE + t];
        const float xj = (float)(pj_x_h.x - (float)shift_j_d.x);
        const float yj = (float)(pj_x_h.y - (float)shift_j_d.y);
        const float zj = (float)(pj_x_h.z - (float)shift_j_d.z);
        // const float hj = pj_x_h.w; // (hj not used in this gradient kernel)

//        const float aviscj  = pj_u_rho_c_aviscmax.w;

        /*Find particle distances*/
        const float xij = xi - xj;
        const float yij = yi - yj;
        const float zij = zi - zj;

        const float r2  = fmaf(xij, xij, fmaf(yij, yij, zij * zij));
        if (!(r2 < hig2)) continue;

        const float vxj = pj_vx_m.x, vyj = pj_vx_m.y, vzj = pj_vx_m.z, mj = pj_vx_m.w;

        const float energyj = pj_u_rho_c_aviscmax.x;
        const float rhoj    = pj_u_rho_c_aviscmax.y;
        const float cj      = pj_u_rho_c_aviscmax.z;
        /*TODO: avisc_vsig_lapu should be changed to float 2 since we only need avisc and vsig for comparison. Lapu is accumulator*/
        const float aviscj  = pj.avisc_vsig_lapu.x;
        res_aviscmax_vsig_lapui.x = fmaxf(res_aviscmax_vsig_lapui.x, aviscj);
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


        // Signal velocity (update running max across neighbors; initialised as vsigi)
        const float new_v_sig = ci + cj - const_viscosity_beta * mu_ij;
        res_aviscmax_vsig_lapui.y = fmaxf(res_aviscmax_vsig_lapui.y, new_v_sig);

        // Kernel (derivative wrt r/hi); only wi_dx needed here
        float wi, wi_dx;
        const float ui = r * hi_inv;
        d_kernel_deval(ui, &wi, &wi_dx);

        // Laplacian(u) accumulation
        // delta_u_factor = (u_i - u_j) / r
        /*Energy calculations*/
        const float delta_u_factor = (energyi - energyj) * inv_r;
        res_aviscmax_vsig_lapui.z += mj * delta_u_factor * wi_dx * (1.0f / rhoj);

      }
    }

    __syncthreads();
  }

  /*Conditional to prevent writing out of bounds of this computation*/
  if (i_in_range) {
    //aviscmax
    atomicMaxFloat(&d_parts_recv[i_id].avisc_vsig_lapu.y, vsigi);
    //vsig
    atomicMaxFloat(&d_parts_recv[i_id].avisc_vsig_lapu.x, res_aviscmax_vsig_lapui.x);
    //lapu
    atomicAdd(&d_parts_recv[i_id].avisc_vsig_lapu.z, res_aviscmax_vsig_lapui.z);
  }

}

__global__ void cuda_kernel_gradient(
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
    neighbour_interactions_gradient(
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

        neighbour_interactions_gradient(
            d_parts_send, d_parts_recv,
            cj_start, cj_end - 1,
            ci_start, ci_end - 1,
            shift_ii_res, shift_jj_res,
            b_id_local, tid,
            d_a, d_H
        );
    }
}

__global__ void cuda_kernel_force(
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
    neighbour_interactions_force(
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

        neighbour_interactions_force(
            d_parts_send, d_parts_recv,
            cj_start, cj_end - 1,
            ci_start, ci_end - 1,
            shift_ii_res, shift_jj_res,
            b_id_local, tid,
            d_a, d_H
        );
    }
}

/**
 * @brief Naive kernel computing the force interactions of a single particle
 *
 * @param pid index of particle to compute density for in the data arrays
 * @param d_pars_send array of particle data received from CPU
 * @param d_parts_recv array of particle data to write results into
 * @param d_a current cosmological expansion factor
 * @param d_H current Hubble constant
 */
__device__ __attribute__((always_inline)) INLINE void neighbour_interactions_force(
    int pid, const struct gpu_part_send_f *__restrict__ d_parts_send,
    struct gpu_part_recv_f *__restrict__ d_parts_recv, float d_a, float d_H) {

  /* First, grab handles */
  const struct gpu_part_data_f pi = d_parts_send[pid];

  const float xi = pi.x_h.x;
  const float yi = pi.x_h.y;
  const float zi = pi.x_h.z;
  const float hi = pi.x_h.w;

  const float vxi = pi.vx_m.x;
  const float vyi = pi.vx_m.y;
  const float vzi = pi.vx_m.z;
  const float mi = pi.vx_m.w;

  const float energyi = pi.u_rho_f_p.x;
  const float rhoi = pi.u_rho_f_p.y;
  const float fi = pi.u_rho_f_p.z;
  const float pressurei = pi.u_rho_f_p.w;

  const float balsi = pi.bals_c_avisc_adiff.x;
  const float ci = pi.bals_c_avisc_adiff.y;
  const float avisci = pi.bals_c_avisc_adiff.z;
  const float adiffi = pi.bals_c_avisc_adiff.w;

  /* const int tbi = pi.timebin_minngbtimebin_pjs_pje.x; */
  const int min_ngb_tbi = pi.timebin_minngbtimebin_pjs_pje.y;
  const int pj_start = pi.timebin_minngbtimebin_pjs_pje.z;
  const int pj_end = pi.timebin_minngbtimebin_pjs_pje.w;

  /* Some auxiliary computations */
  const float hig2 = hi * hi * kernel_gamma2;
  const float hi_inv = 1.f / hi;
  const float hid_inv = d_pow_dimension_plus_one(hi_inv); /* 1/h^(d+1) */
  const float mi_inv = 1.f / mi;
  const float rhoi_inv = 1.f / rhoi;
  const float rhoi_inv2 = rhoi_inv * rhoi_inv;

  /* Prep output */
  float3 res_ahydro = {0.f, 0.f, 0.f};
  float2 res_udt_hdt = {0.f, 0.f};
  int res_min_ngb_timebin = min_ngb_tbi;

  /* Start the neighbour interactions */
  for (int j = pj_start; j < pj_end; j++) {

    /* First, grab handles. */
    const struct gpu_part_data_f pj = d_parts_send[j];

    const float xj = pj.x_h.x;
    const float yj = pj.x_h.y;
    const float zj = pj.x_h.z;
    const float hj = pj.x_h.w;

    const float vxj = pj.vx_m.x;
    const float vyj = pj.vx_m.y;
    const float vzj = pj.vx_m.z;
    const float mj = pj.vx_m.w;

    const float energyj = pj.u_rho_f_p.x;
    const float rhoj = pj.u_rho_f_p.y;
    const float fj = pj.u_rho_f_p.z;
    const float pressurej = pj.u_rho_f_p.w;

    const float balsj = pj.bals_c_avisc_adiff.x;
    const float cj = pj.bals_c_avisc_adiff.y;
    const float aviscj = pj.bals_c_avisc_adiff.z;
    const float adiffj = pj.bals_c_avisc_adiff.w;

    const int tbj = pj.timebin_minngbtimebin_pjs_pje.x;
    /* const int min_ngb_tbj = pj.timebin_minngbtimebin_pjs_pje.y; */

    /* Now get stuff done. */
    const float xij = xi - xj;
    const float yij = yi - yj;
    const float zij = zi - zj;
    const float r2 = xij * xij + yij * yij + zij * zij;
    const float hjg2 = hj * hj * kernel_gamma2;

    /* (j != pid): Exclude self contribution. This happens at a later step. */
    const bool iact_condition = ((r2 < hig2) || (r2 < hjg2)) && (j != pid);
    const float mask = iact_condition ? 1.f : 0.f;
    const int tbj_masked =
        iact_condition && (tbj > 0) ? tbj : res_min_ngb_timebin;

    /* Cosmology terms for the signal velocity */
    const float fac_mu = d_pow_three_gamma_minus_five_over_two(d_a);
    const float a2_Hubble = d_a * d_a * d_H;

    const float r = sqrtf(r2);
    /* r == 0 can happen for self-interaction, which we're masking out,
     * but it'll produce NaNs through division by zero, so handle that. */
    const float r_inv = r > 0.f ? (1.f / r) : 1.f;

    /* Get the kernel for hi. */
    const float qi = r * hi_inv;
    float wi;
    float wi_dx;
    d_kernel_deval(qi, &wi, &wi_dx);
    const float wi_dr = hid_inv * wi_dx;

    /* Get the kernel for hj. */
    const float hj_inv = 1.0f / hj;
    const float hjd_inv = d_pow_dimension_plus_one(hj_inv); /* 1/h^(d+1) */
    const float qj = r * hj_inv;
    float wj;
    float wj_dx;
    d_kernel_deval(qj, &wj, &wj_dx);
    const float wj_dr = hjd_inv * wj_dx;

    /* Compute dv dot r */
    float dvx = vxi - vxj;
    float dvy = vyi - vyj;
    float dvz = vzi - vzj;
    const float dvdr = dvx * xij + dvy * yij + dvz * zij;

    /* Add Hubble flow; not used for du/dt */
    const float dvdr_Hubble = dvdr + a2_Hubble * r2;

    /* Are the particles moving towards each others ? */
    const float omega_ij = min(dvdr_Hubble, 0.f);
    const float mu_ij = fac_mu * r_inv * omega_ij; /* This is 0 or negative */

    /* Signal velocity */
    const float v_sig = ci + cj - const_viscosity_beta * mu_ij;

    /* Variable smoothing length term */
    const float f_ij = 1.f - fi / mj;
    const float f_ji = 1.f - fj * mi_inv;

    /* Construct the full viscosity term */
    const float rhoij = rhoi + rhoj;
    const float rhoij_inv = 1.f / rhoij;
    const float alpha = avisci + aviscj;
    const float visc =
        -0.25f * alpha * v_sig * mu_ij * (balsi + balsj) * rhoij_inv;

    /* Convolve with the kernel */
    const float visc_acc_term =
        0.5f * visc * (wi_dr * f_ij + wj_dr * f_ji) * r_inv;

    /* Compute gradient terms */
    const float rhoj2 = rhoj * rhoj;
    const float rhoj_inv = 1.f / rhoj;
    const float P_over_rho2_i = pressurei * rhoi_inv2 * f_ij;
    const float P_over_rho2_j = pressurej / (rhoj2) * f_ji;

    /* SPH acceleration term */
    const float sph_acc_term =
        (P_over_rho2_i * wi_dr + P_over_rho2_j * wj_dr) * r_inv;

    /* Assemble the acceleration */
    const float acc = sph_acc_term + visc_acc_term;

    /* Use the force Luke ! */
    res_ahydro.x -= mj * acc * xij * mask;
    res_ahydro.y -= mj * acc * yij * mask;
    res_ahydro.z -= mj * acc * zij * mask;

    /* Get the time derivative for u. */
    const float sph_du_term_i = P_over_rho2_i * dvdr * r_inv * wi_dr;

    /* Viscosity term */
    const float visc_du_term = 0.5f * visc_acc_term * dvdr_Hubble;

    /* Diffusion term */
    /* Combine the alpha_diff into a pressure-based switch -- this allows the
     * alpha from the highest pressure particle to dominate, so that the
     * diffusion limited particles always take precedence - another trick to
     * allow the scheme to work with thermal feedback. */
    float alpha_diff =
        (pressurei * adiffi + pressurej * adiffj) / (pressurei + pressurej);
    /* if (fabsf(pressurei + pressurej) < 1e-10) alpha_diff = 0.f; */

    const float v_diff =
        alpha_diff * 0.5f *
        (sqrtf(2.f * fabsf(pressurei - pressurej) * rhoij_inv) +
         fabsf(fac_mu * r_inv * dvdr_Hubble));

    /* wi_dx + wj_dx / 2 is F_ij */
    const float diff_du_term =
        v_diff * (energyi - energyj) *
        (f_ij * wi_dr * rhoi_inv + f_ji * wj_dr * rhoj_inv);

    /* Assemble the energy equation term */
    const float du_dt_i = sph_du_term_i + visc_du_term + diff_du_term;

    /* Internal energy time derivative */
    res_udt_hdt.x += du_dt_i * mj * mask;

    /* Get the time derivative for h. */
    res_udt_hdt.y -= mj * dvdr * r_inv * rhoj_inv * wi_dr * mask;

    /* tbj > 0 check is included in mask */
    res_min_ngb_timebin = min(res_min_ngb_timebin, tbj_masked);
  }

  d_parts_recv[pid].udt_hdt_minngbtb = {res_udt_hdt.x, res_udt_hdt.y,
                                        (float)res_min_ngb_timebin};
  d_parts_recv[pid].a_hydro = res_ahydro;
}

#ifdef __cplusplus
}
#endif

#endif /* CUDA_PARTICLE_KERNELS_CUH */
