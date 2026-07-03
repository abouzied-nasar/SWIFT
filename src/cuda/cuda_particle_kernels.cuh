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
#include <stdio.h>

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

	/*Declare a variable to use up allocated shared memory*/
  extern __shared__ unsigned char smem[];
  /* TODO: we need positions to be double so this needs re-working in the near future!*/
  /*Assign range of memory to use for x, y, z and h*/
  double2* s_x_y = reinterpret_cast<double2*>(smem);
  double2* s_z_h = reinterpret_cast<double2*>(s_x_y + 2 * GPU_THREAD_BLOCK_SIZE);
  /*Assign range of memory to use for velocity (u, v, w) and mass*/
  float4* s_vx_m = reinterpret_cast<float4*>(s_z_h + 2 * GPU_THREAD_BLOCK_SIZE);

  /*Map this thread to its i-particle*/
  const int i_id = b_id_local * GPU_THREAD_BLOCK_SIZE + tid + i_start;
  /*Is the particle i_id in the cell we need to work on?*/
  const bool i_in_range = (i_id < i_end);

  /* Initialise particle i's data. Needed since we require definition
     * before checking if i_in_range below */
  float xi = 0.f, yi = 0.f, zi = 0.f, hi = 0.f;
  float vxi = 0.f, vyi = 0.f, vzi = 0.f;
  float hig2 = 0.f, hi_inv = 0.f;

  /*Do not do any calculation if i_id is not in cell i range of particles*/
  if (i_in_range) {

	  /* First, grab handles. */
      const struct gpu_part_data_d pi = d_parts_send[i_id].p_data;
	  /*Calculate i's position local to the cell*/
	  xi = (pi.x_y.x - shift_i_d.x);
	  yi = (pi.x_y.y - shift_i_d.y);
	  zi = (pi.z_h.x - shift_i_d.z);
	  /*Get particle i smoothing length*/
	  hi = (pi.z_h.y);
      /*Find my velocities, mass not needed for particle i*/
      vxi = pi.vx_m.x;
      vyi = pi.vx_m.y;
      vzi = pi.vx_m.z;

      /*Calculate smoothing length related parameters*/
      hig2   = (hi * hi) * kernel_gamma2;
      hi_inv = 1.0f / hi;
  }

  /* Prep output */
  /* rho, rho_dh, wcount, wcount_dh */
  float4 res_rho = make_float4(0.f, 0.f, 0.f, 0.f);
  /* curl of velocity (3 coordinates), velocity divergence */
  float4 res_rot = make_float4(0.f, 0.f, 0.f, 0.f);

  /*const to avoid div by zero*/
//  constexpr float eps = 1e-24f;

  /* Number of tiles. How many times to de we need to load GPU_
     * THREAD_BLOCK_SIZE particles to get through this cell? */
  const int numTiles = (j_end - j_start + GPU_THREAD_BLOCK_SIZE - 1) / GPU_THREAD_BLOCK_SIZE;

  /* Prefetch tile 0 into buffer 0 */
  if (numTiles > 0) {
      const int base0      = j_start;
      const int tileCount0 = min(GPU_THREAD_BLOCK_SIZE, j_end - base0);

      for (int t = tid; t < tileCount0; t += GPU_THREAD_BLOCK_SIZE) {
          const int gj = base0 + t;
          __pipeline_memcpy_async(&s_x_y[0 * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.x_y, sizeof(double2));
          __pipeline_memcpy_async(&s_z_h[0 * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.z_h, sizeof(double2));
          __pipeline_memcpy_async(&s_vx_m[0 * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.vx_m, sizeof(float4));
      }
      __pipeline_commit();
  }
  /* Loop over tiles prefetching of "next" tile while computing "current" */
  for (int tile = 0; tile < numTiles; ++tile){

	  /*Is this tile 0 or 1 (ping-pong tile execution and prefetching)*/
      const int buf       = tile & 1;  // 0 or 1 (ping-pong)
      const int base      = j_start + tile * GPU_THREAD_BLOCK_SIZE;
      const int tileCount = min(GPU_THREAD_BLOCK_SIZE, j_end - base);

      /* Make sure the current tile (already committed) is resident in shared memory */
      __pipeline_wait_prior(0);
      __syncthreads();

      /* Initiate prefetch for the next tile (overlaps with compute further in the loop) */
      const int nextTile = tile + 1;
      if (nextTile < numTiles){

    	/*If nextTile & 1 == 0. nextTile is even. If nextTile & 1 == 1, nextTile is odd*/
        const int nextBuf  = nextTile & 1;
        /*Where does the next tile begin in the buffer array?*/
        const int nextBase = j_start + nextTile * GPU_THREAD_BLOCK_SIZE;
        /*What is the number of required threads in the next tile? If we are at the end of
               * cell j's range in the buffer only read to the end of the range*/
        const int nextCnt  = min(GPU_THREAD_BLOCK_SIZE, j_end - nextBase);

        /*Now issue pre-fetch for next data set*/
        for (int t = tid; t < nextCnt; t += GPU_THREAD_BLOCK_SIZE) {
          const int gj = nextBase + t;
		  __pipeline_memcpy_async(&s_x_y[nextBuf * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.x_y, sizeof(double2));
		  __pipeline_memcpy_async(&s_z_h[nextBuf * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.z_h, sizeof(double2));
          __pipeline_memcpy_async(&s_vx_m[nextBuf * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.vx_m, sizeof(float4));
        }
        /*Commit but don't sync, syncing is done at the end of computations*/
        __pipeline_commit();
      }
      /* Run computations on the current tile (buf)*/
      if (i_in_range)
      {
      /* Start the neighbour interactions */
#pragma unroll 4
          for (int t = 0; t < tileCount; ++t){

        	  /*grab the particle's index in the buffer array*/
              const int j_idx = base + t;
              /* Exclude self contribution. This happens at a later step. */
              if (j_idx == i_id) continue;

              /* First, grab handles. */
              const double2 pj_x_y = s_x_y[buf * GPU_THREAD_BLOCK_SIZE + t]; // unshifted
              const double2 pj_z_h = s_z_h[buf * GPU_THREAD_BLOCK_SIZE + t]; // unshifted
              const float4 pj_vel = s_vx_m[buf * GPU_THREAD_BLOCK_SIZE + t];

			  const float xj = (pj_x_y.x - shift_j_d.x);
			  const float yj = (pj_x_y.y - shift_j_d.y);
			  const float zj = (pj_z_h.x - shift_j_d.z);

              /* Now get stuff done*/
              const float xij = xi - xj;
              const float yij = yi - yj;
              const float zij = zi - zj;

              const float r2 = fmaf(xij, xij, fmaf(yij, yij, zij * zij));
              if (r2 >= hig2) continue;

              const float inv_r = rsqrtf(r2);
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
      /* Ensure no thread is still reading from the current buffer before it may be
       * overwritten next */
      __syncthreads();
  }

  /*Conditional to prevent writing out of bounds of this computation*/
  if (i_in_range) {
	  /* Write results. */
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

	/*Figure out which range of particles threads in this block will work on*/
    const int bid     = blockIdx.x;
    /*What is the leaf computation this block will work on*/
    const int leafid  = d_block_leaf_id[bid].x;
    /*In case we need more than one block to run this leaf computation we need to know
     * where in the group of blocks acting on a cell we are.
     * bid_0 is the id of the first block acting on this cell*/
    const int bid_0   = d_block_leaf_id[bid].y;
    /*Get the start and end positions of cells i and j*/
    int4 cell_se      = d_cell_i_j_start_end[leafid];

    const int ci_start = cell_se.x;
    const int ci_end   = cell_se.y; // cell pos at index [ci_end-1]
    const int cj_start = cell_se.z;
    const int cj_end   = cell_se.w; // cell pos at index [cj_end-1]

    /*We can now find our block index in a local reference to this cell*/
    const int b_id_local = bid - bid_0;
    const int tid = threadIdx.x;

    /* Get the cell positions. We store them as the last entry in the
     * particle array for each leaf computation */
    const auto ci_loc = d_parts_send[ci_end - 1].c_loc;
    const auto cj_loc = d_parts_send[cj_end - 1].c_loc;

    /*Calculate the shifted cell positions if we have periodics*/
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
  /* TODO: Should add float2 in shared for avisc and vsig*/
  /*Assign range of memory to use for x, y, z and h*/
  double2* s_x_y = reinterpret_cast<double2*>(smem);
  double2* s_z_h = reinterpret_cast<double2*>(s_x_y + 2 * GPU_THREAD_BLOCK_SIZE);
  /*Assign range of memory to use for velocity (u, v, w) and mass*/
  float4* s_vx_m = reinterpret_cast<float4*>(s_z_h + 2 * GPU_THREAD_BLOCK_SIZE);
  /*Assign range of memory to use for energy u, density rho, speed of sound c, alpha visc*/
  float4* s_u_rho_c_aviscmax = reinterpret_cast<float4*>(s_vx_m + 2 * GPU_THREAD_BLOCK_SIZE);
  /*Assign range of memory to use for energy u, density rho, speed of sound c, alpha visc*/
  float4* s_avisc_vsig = reinterpret_cast<float4*>(s_u_rho_c_aviscmax + 2 * GPU_THREAD_BLOCK_SIZE);

  /*Map this thread to its i-particle*/
  const int  i_id = b_id_local * GPU_THREAD_BLOCK_SIZE + tid + i_start;
  /*Is the id in the cell we need to work on?*/
  const bool i_in_range = (i_id < i_end);

  /* Initialise particle i's data. Needed since we require definition
   * before checking if i_in_range below */
  float xi=0.f, yi=0.f, zi=0.f, hi=0.f;
  float vxi=0.f, vyi=0.f, vzi=0.f;
  float energyi=0.f, ci=0.f;
  float vsigi=0.f, lapui=0.f, avisc_maxi=0.f;
  float hi_inv=0.f, hig2=0.f;

  /*Do not do any calculation if i_id is not in cell i range of particles*/
  if (i_in_range) {

	/* First, grab handles. */
    const struct gpu_part_data_g pi = d_parts_send[i_id].p_data;

    /*Calculate i's position local to the cell*/
	xi = (pi.x_y.x - shift_i_d.x);
	yi = (pi.x_y.y - shift_i_d.y);
	zi = (pi.z_h.x - shift_i_d.z);
    /*Get particle i smoothing length*/
	hi = (pi.z_h.y);
    /*Find my velocities, mass not needed for particle i*/
    vxi = pi.vx_m.x;
    vyi = pi.vx_m.y;
    vzi = pi.vx_m.z;

    /*Now get energy i and speed of sound from u_rho_c_aviscmax (u, rho, c, aviscmax)
     * rho not needed*/
    energyi = pi.u_rho_c_aviscmax.x;
    /*rhoi  = pi.u_rho_c_aviscmax.y;*/
    ci      = pi.u_rho_c_aviscmax.z;
    /* Get previous value of avisc and vsig. lapu will be incremented while
     * for avisc we want to find the maximum. Set lapu to zero. We read the actula avisc for i here
     * not aviscmax*/
    avisc_maxi  = pi.u_rho_c_aviscmax.w;
    /*Now get vsigi. avisc i not needed since we use aviscmax above*/
    vsigi       = pi.avisc_vsig.y;
    /*lapui incremeneted at the end. Initialise to zero for j loop*/
    lapui       = 0.f;
    /*Calculate smoothing length related parameters*/
    hi_inv = 1.0f / hi;
    hig2   = (hi * hi) * kernel_gamma2;
  }

  /* Cosmology terms for the signal velocity */
  const float fac_mu    = d_pow_three_gamma_minus_five_over_two(d_a);
  const float a2_Hubble = d_a * d_a * d_H;

  /*const to avoid div by zero*/
//  constexpr float eps = 1e-24f;

  /* Number of tiles. How many times to de we need to load GPU_
   * THREAD_BLOCK_SIZE particles to get through this cell? */
  const int numTiles = (j_end - j_start + GPU_THREAD_BLOCK_SIZE - 1) / GPU_THREAD_BLOCK_SIZE;

  /* Prefetch tile 0 into buffer 0 */
  if (numTiles > 0){
    const int base0      = j_start;
    const int tileCount0 = min(GPU_THREAD_BLOCK_SIZE, j_end - base0);

    for (int t = tid; t < tileCount0; t += GPU_THREAD_BLOCK_SIZE) {
      const int gj = base0 + t;
      __pipeline_memcpy_async(&s_x_y[0 * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.x_y, sizeof(double2));
      __pipeline_memcpy_async(&s_z_h[0 * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.z_h, sizeof(double2));
      __pipeline_memcpy_async(&s_vx_m[0 * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.vx_m, sizeof(float4));
      __pipeline_memcpy_async(&s_u_rho_c_aviscmax[0 * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.u_rho_c_aviscmax, sizeof(float4));
      __pipeline_memcpy_async(&s_avisc_vsig[0 * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.avisc_vsig, sizeof(float4));
    }
    __pipeline_commit();
  }
  /* Loop over tiles prefetching of "next" tile while computing "current" */
  for (int tile = 0; tile < numTiles; ++tile){

	/*Is this tile 0 or 1 (ping-pong tile execution and prefetching)*/
    const int buf       = tile & 1;
    const int base      = j_start + tile * GPU_THREAD_BLOCK_SIZE;
    const int tileCount = min(GPU_THREAD_BLOCK_SIZE, j_end - base);

    /* Make sure the current tile (already committed) is resident in shared memory */
    __pipeline_wait_prior(0);
    __syncthreads();

    /* Initiate prefetch for the next tile (overlaps with compute further in the loop) */
    const int nextTile = tile + 1;
    if (nextTile < numTiles){
      /*If nextTile & 1 == 0. nextTile is even. If nextTile & 1 == 1, nextTile is odd*/
      const int nextBuf  = nextTile & 1;
      /*Where does the next tile begin in the buffer array?*/
      const int nextBase = j_start + nextTile * GPU_THREAD_BLOCK_SIZE;
      /*What is the number of required threads in the next tile? If we are at the end of
       * cell j's range in the buffer only read to the end of the range*/
      const int nextCnt  = min(GPU_THREAD_BLOCK_SIZE, j_end - nextBase);

      /*Now issue pre-fetch for next data set*/
      for (int t = tid; t < nextCnt; t += GPU_THREAD_BLOCK_SIZE) {
        const int gj = nextBase + t;
		__pipeline_memcpy_async(&s_x_y[nextBuf * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.x_y, sizeof(double2));
		__pipeline_memcpy_async(&s_z_h[nextBuf * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.z_h, sizeof(double2));
        __pipeline_memcpy_async(&s_vx_m[nextBuf * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.vx_m, sizeof(float4));
        __pipeline_memcpy_async(&s_u_rho_c_aviscmax[nextBuf * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.u_rho_c_aviscmax, sizeof(float4));
        __pipeline_memcpy_async(&s_avisc_vsig[nextBuf * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.avisc_vsig, sizeof(float4));
      }
      /*Commit but don't sync, syncing is done at the end of computations*/
      __pipeline_commit();
    }
    /* Run computations on the current tile (buf)*/
    if (i_in_range){
      /* Start the neighbour interactions */
#pragma unroll 4
      for (int t = 0; t < tileCount; ++t){

    	/* We need to construct the maximal signal velocity between our particle
    	 * and all of it's neighbours */
    	/*grab the particle's index in the buffer array*/
        const int j_idx = base + t;
        /* Exclude self contribution. This happens at a later step. */
        if (j_idx == i_id) continue;

        /* First, grab handles. */
        const double2 pj_x_y = s_x_y[buf * GPU_THREAD_BLOCK_SIZE + t]; // unshifted
        const double2 pj_z_h = s_z_h[buf * GPU_THREAD_BLOCK_SIZE + t]; // unshifted
        const float4 pj_vx_m = s_vx_m[buf * GPU_THREAD_BLOCK_SIZE + t];
        /*Get rho, visc, energy and speed of sound*/
        const float4 pj_u_rho_c_aviscmax = s_u_rho_c_aviscmax[buf * GPU_THREAD_BLOCK_SIZE + t];
        /*TODO: This is over-kill for since we only need avisc for particles j
         *  but leave as-is for now until I work out a better data arrangement:
         *  We need vsig from CPU for particle i but not it's neighbours
         *  and we need aviscmax and vsig for particle i and not it's neighbours*/
        const float4 pj_avisc_vsig = s_avisc_vsig[buf * GPU_THREAD_BLOCK_SIZE + t];
		const float xj = (pj_x_y.x - shift_j_d.x);
		const float yj = (pj_x_y.y - shift_j_d.y);
		const float zj = (pj_z_h.x - shift_j_d.z);

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

        const float aviscj  = pj_avisc_vsig.x;

        avisc_maxi = fmaxf(avisc_maxi, aviscj);
        const float inv_r = rsqrtf(r2);
        const float r     = r2 * inv_r;

        const float dvx  = vxi - vxj;
        const float dvy  = vyi - vyj;
        const float dvz  = vzi - vzj;

        const float dvdr = fmaf(dvx, xij, fmaf(dvy, yij, dvz * zij));
        const float dvdr_Hubble = dvdr + a2_Hubble * r2;

        /* Are the particles moving towards each others ? */
        const float omega_ij = fminf(dvdr_Hubble, 0.f);
        const float mu_ij    = fac_mu * inv_r * omega_ij; /* This is 0 or negative */


        /* Signal velocity (update running max across neighbors; initialised as vsigi) */
        const float new_v_sig = ci + cj - const_viscosity_beta * mu_ij;
        /* Update if we need to */
        vsigi = fmaxf(vsigi, new_v_sig);

        /* Calculate Del^2 u for the thermal diffusion coefficient. */
        /* Need to get some kernel values F_ij = wi_dx */
        float wi, wi_dx;
        const float ui = r * hi_inv;
        d_kernel_deval(ui, &wi, &wi_dx);

        const float delta_u_factor = (energyi - energyj) * inv_r;
        lapui += mj * delta_u_factor * wi_dx * (1.0f / rhoj);

      }
    }

    __syncthreads();
  }

  /*Conditional to prevent writing out of bounds of this computation*/
  if (i_in_range) {
    /*aviscmax*/
	atomicMaxFloat(&d_parts_recv[i_id].aviscmax_vsig_lapu.x, avisc_maxi);
	/*vsig*/
    atomicMaxFloat(&d_parts_recv[i_id].aviscmax_vsig_lapu.y, vsigi);
    /*lapu*/
    atomicAdd(&d_parts_recv[i_id].aviscmax_vsig_lapu.z, lapui);
  }

}

__global__ void cuda_kernel_gradient(
    const struct gpu_part_send_g* __restrict__ d_parts_send,
    struct gpu_part_recv_g*      __restrict__ d_parts_recv,
    const float d_a, const float d_H,
    const int4* __restrict__ d_cell_i_j_start_end,
    const int2* __restrict__ d_block_leaf_id,
    const double3 space_dim){

	/*Figure out which range of particles threads in this block will work on*/
    const int bid     = blockIdx.x;
    /*What is the leaf computation this block will work on*/
    const int leafid  = d_block_leaf_id[bid].x;
    /*In case we need more than one block to run this leaf computation we need to know
     * where in the group of blocks acting on a cell we are.
     * bid_0 is the id of the first block acting on this cell*/
    const int bid_0   = d_block_leaf_id[bid].y;
    /*Get the start and end positions of cells i and j*/
    int4 cell_se      = d_cell_i_j_start_end[leafid];

    const int ci_start = cell_se.x;
    const int ci_end   = cell_se.y; // cell position at [ci_end-1]
    const int cj_start = cell_se.z;
    const int cj_end   = cell_se.w;

    /*We can now find our block index in a local reference to this cell*/
    const int b_id_local = bid - bid_0;
    const int tid        = threadIdx.x;

    /* Get the cell positions. We store them as the last entry in the
     * particle array for each leaf computation */
    const auto ci_loc = d_parts_send[ci_end - 1].c_loc;
    const auto cj_loc = d_parts_send[cj_end - 1].c_loc;

    /*Calculate the shifted cell positions if we have periodics*/
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
__device__ __forceinline__ void neighbour_interactions_force(
    const struct gpu_part_send_f* __restrict__ d_parts_send,
    struct gpu_part_recv_f*      __restrict__ d_parts_recv,
    int i_start, int i_end, int j_start, int j_end,
    const double3 shift_i_d, const double3 shift_j_d, int b_id_local,
	int tid, float d_a, float d_H) {

	/*Declare a variable to use up allocated shared memory*/
	extern __shared__ __align__(16) unsigned char smem[];
	/* TODO: we need positions to be double so this needs re-working in the near future!*/
	/*Assign range of memory to use for x, y, z and h*/
	double2* s_x_y = reinterpret_cast<double2*>(smem);
	double2* s_z_h = reinterpret_cast<double2*>(s_x_y + 2 * GPU_THREAD_BLOCK_SIZE);
	/*Assign range of memory to use for velocity (u, v, w) and mass*/
	float4* s_vx_m = reinterpret_cast<float4*>(s_z_h + 2 * GPU_THREAD_BLOCK_SIZE);
	/*Assign range of memory to use for energy u, density rho, smoothing length constant f, pressure p*/
	float4* s_u_r_f_p = reinterpret_cast<float4*>(s_vx_m  + 2 * GPU_THREAD_BLOCK_SIZE);
	/*Assign range of memory to use for balsara b, speed of sound c, alpha visc av and diffusion ad*/
	float4* s_b_c_av_ad = reinterpret_cast<float4*>(s_u_r_f_p + 2 * GPU_THREAD_BLOCK_SIZE);
	/*Assign range of memory to use for balsara b, speed of sound c, alpha visc av and diffusion ad*/
	int2* s_tb_min_ngb_tb = reinterpret_cast<int2*>(s_b_c_av_ad + 2 * GPU_THREAD_BLOCK_SIZE);

	/*Map this thread to its i-particle*/
	const int  i_id = b_id_local * GPU_THREAD_BLOCK_SIZE + tid + i_start;
	/*Is the id in the cell we need to work on?*/
	const bool i_in_range  = (i_id < i_end);

	/* Initialise particle i's data. Needed since we require definition
	 * before checking if i_in_range below */
	float xi=0.f, yi=0.f, zi=0.f, hi=0.f;
	float vxi=0.f, vyi=0.f, vzi=0.f, mi=0.f;
	float fi=0.f, balsi=0.f, rhoi=0.f, pressurei=0.f;
	float ci=0.f, energyi=0.f, avisci=0.f, adiffi=0.f;
	int   min_ngb_tbi=INT_MAX;

	float hi_inv=0.f, hid_inv=0.f, mi_inv=0.f, rhoi_inv=0.f, rhoi_inv2=0.f, hig2=0.f;
	/*Do not do any calculation if i_id is not in cell i range of particles*/
	if (i_in_range) {
		/* First, grab handles. */
		const struct gpu_part_data_f pi = d_parts_send[i_id].p_data;

	    /*Calculate i's position local to the cell*/
		xi = (pi.x_y.x - shift_i_d.x);
		yi = (pi.x_y.y - shift_i_d.y);
		zi = (pi.z_h.x - shift_i_d.z);
	    /*Get particle i smoothing length*/
		hi = (pi.z_h.y);

	    /*Find my velocities, mass not needed for particle i*/
		vxi = pi.vx_m.x;
		vyi = pi.vx_m.y;
		vzi = pi.vx_m.z;
		mi  = pi.vx_m.w;

	    /*Now get energy i and the rest of the variables we need*/
		energyi = pi.u_rho_f_p.x;
		rhoi = pi.u_rho_f_p.y;
		fi = pi.u_rho_f_p.z;
		pressurei = pi.u_rho_f_p.w;

		balsi = pi.bals_c_avisc_adiff.x;
		ci = pi.bals_c_avisc_adiff.y;
		avisci = pi.bals_c_avisc_adiff.z;
		adiffi  = pi.bals_c_avisc_adiff.w;

//		min_ngb_tbi = pi.timebin_minngbtimebin.y;

		const int old_min_ngb_tbi = pi.timebin_minngbtimebin.y;
		min_ngb_tbi = old_min_ngb_tbi > 0 ? old_min_ngb_tbi : INT_MAX;


		/*If no CUDA thread has written to it yet, the result will be zero.
		 * So, initialise to the value we got from the CPU. Otherwise, leave as-is.
		 * TODO: Do we need to check if min_ngb_tbi > 0?*/
	    atomicCAS(&d_parts_recv[i_id].minngbtb, 0, min_ngb_tbi);

		/* Get the kernel for hi. */
		hi_inv   = 1.0f / hi;
		hid_inv  = d_pow_dimension_plus_one(hi_inv); /* 1/h^(d+1) */
		mi_inv   = 1.0f / mi;
		rhoi_inv = 1.0f / rhoi;
		rhoi_inv2= rhoi_inv * rhoi_inv;
		hig2     = (hi * hi) * kernel_gamma2;
	}

	/*Accumulators (results) for acceleration and everything else*/
	float3 res_ahydro  = {0.f, 0.f, 0.f};
	float2 res_udt_hdt = {0.f, 0.f};

	/* Cosmology terms for the signal velocity */
	const float fac_mu    = d_pow_three_gamma_minus_five_over_two(d_a);
	const float a2_Hubble = d_a * d_a * d_H;
	/*const to avoid div by zero*/
//	constexpr float eps = 1e-24f;

	/* Number of tiles. How many times to de we need to load GPU_
	 * THREAD_BLOCK_SIZE particles to get through this cell? */
	const int numTiles = (j_end - j_start + GPU_THREAD_BLOCK_SIZE - 1) / GPU_THREAD_BLOCK_SIZE;

	/* Prefetch tile 0 into buffer 0 */
	if (numTiles > 0){
		const int base0      = j_start;
		const int tileCount0 = min(GPU_THREAD_BLOCK_SIZE, j_end - base0);

		for (int t = tid; t < tileCount0; t += GPU_THREAD_BLOCK_SIZE) {
			const int gj = base0 + t;
	        __pipeline_memcpy_async(&s_x_y[0 * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.x_y, sizeof(double2));
	        __pipeline_memcpy_async(&s_z_h[0 * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.z_h, sizeof(double2));
			__pipeline_memcpy_async(&s_vx_m[0 * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.vx_m, sizeof(float4));
			__pipeline_memcpy_async(&s_u_r_f_p[0 * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.u_rho_f_p, sizeof(float4));
			__pipeline_memcpy_async(&s_b_c_av_ad[0 * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.bals_c_avisc_adiff, sizeof(float4));
			__pipeline_memcpy_async(&s_tb_min_ngb_tb[0 * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.timebin_minngbtimebin, sizeof(int2));

		}
		__pipeline_commit();
	}
	/* Loop over tiles prefetching of "next" tile while computing "current" */
	for (int tile = 0; tile < numTiles; ++tile){

		/*Is this tile 0 or 1 (ping-pong tile execution and prefetching)*/
		const int buf       = tile & 1;  // 0 or 1 (ping-pong)
		const int base      = j_start + tile * GPU_THREAD_BLOCK_SIZE;
		const int tileCount = min(GPU_THREAD_BLOCK_SIZE, j_end - base);

	    /* Make sure the current tile (already committed) is resident in shared memory */
		__pipeline_wait_prior(0);
		__syncthreads();


	    /* Initiate prefetch for the next tile (overlaps with compute further in the loop) */
		const int nextTile = tile + 1;
		if (nextTile < numTiles){
			/*If nextTile & 1 == 0. nextTile is even. If nextTile & 1 == 1, nextTile is odd*/
			const int nextBuf  = nextTile & 1;
			/*Where does the next tile begin in the buffer array?*/
			const int nextBase = j_start + nextTile * GPU_THREAD_BLOCK_SIZE;
			/*What is the number of required threads in the next tile? If we are at the end of
			 * cell j's range in the buffer only read to the end of the range*/
			const int nextCnt  = min(GPU_THREAD_BLOCK_SIZE, j_end - nextBase);

		    /*Now issue pre-fetch for next data set*/
			for (int t = tid; t < nextCnt; t += GPU_THREAD_BLOCK_SIZE) {
				const int gj = nextBase + t;
				__pipeline_memcpy_async(&s_x_y[nextBuf * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.x_y, sizeof(double2));
				__pipeline_memcpy_async(&s_z_h[nextBuf * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.z_h, sizeof(double2));
				__pipeline_memcpy_async(&s_vx_m[nextBuf * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.vx_m, sizeof(float4));
				__pipeline_memcpy_async(&s_u_r_f_p[nextBuf * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.u_rho_f_p, sizeof(float4));
				__pipeline_memcpy_async(&s_b_c_av_ad[nextBuf * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.bals_c_avisc_adiff, sizeof(float4));
				__pipeline_memcpy_async(&s_tb_min_ngb_tb[nextBuf * GPU_THREAD_BLOCK_SIZE + t], &d_parts_send[gj].p_data.timebin_minngbtimebin, sizeof(int2));
			}
		    /*Commit but don't sync, syncing is done at the end of computations*/
			__pipeline_commit();
		}

	    /* Run computations on the current tile (buf)*/
		if (i_in_range){
		  /* Start the neighbour interactions */
#pragma unroll 4
			for (int t = 0; t < tileCount; ++t){

		    	/* We need to construct the maximal signal velocity between our particle
		    	 * and all of it's neighbours */
		    	/*grab the particle's index in the buffer array*/
				const int j_idx = base + t;
				/* Exclude self contribution. This happens at a later step. */
				if (j_idx == i_id) continue; // self for self-pairs

				/* First, grab handles. */
	            const double2 pj_x_y = s_x_y[buf * GPU_THREAD_BLOCK_SIZE + t]; // unshifted
	            const double2 pj_z_h = s_z_h[buf * GPU_THREAD_BLOCK_SIZE + t]; // unshifted

				/*Calculate particle position relative to cell position and get smoothing length*/
				const float xj = (pj_x_y.x - shift_j_d.x);
				const float yj = (pj_x_y.y - shift_j_d.y);
				const float zj = (pj_z_h.x - shift_j_d.z);
				const float hj = pj_z_h.y;

				/*Find particle distances*/
				const float xij = xi - xj;
				const float yij = yi - yj;
				const float zij = zi - zj;

				const float r2  = fmaf(xij, xij, fmaf(yij, yij, zij * zij));
				const float hjg2= (hj * hj) * kernel_gamma2;

				if (!((r2 < hig2) || (r2 < hjg2))) continue;

				/* Grab remaining required handles. */
				const float4 pj_vx_m  = s_vx_m[buf * GPU_THREAD_BLOCK_SIZE + t];
				const float4 pj_u_r_f_p = s_u_r_f_p[buf * GPU_THREAD_BLOCK_SIZE + t];
				const float4 pj_b_c_av_ad = s_b_c_av_ad[buf * GPU_THREAD_BLOCK_SIZE + t];
				const int2 pj_tb_min_ngb_tb = s_tb_min_ngb_tb[buf * GPU_THREAD_BLOCK_SIZE + t];

				const float vxj = pj_vx_m.x;
				const float vyj = pj_vx_m.y;
				const float vzj = pj_vx_m.z, mj = pj_vx_m.w;

				const float energyj = pj_u_r_f_p.x;
				const float rhoj = pj_u_r_f_p.y;
				const float fj = pj_u_r_f_p.z;
                const float pressurej = pj_u_r_f_p.w;

                const float balsj = pj_b_c_av_ad.x;
				const float cj  = pj_b_c_av_ad.y;
				const float aviscj = pj_b_c_av_ad.z;
				const float adiffj = pj_b_c_av_ad.w;

				/*second condition is a fix for if min_ngb_tbi is zero.
				 * Unsure why that would be but hey ho*/

				if (pj_tb_min_ngb_tb.x > 0)
				  min_ngb_tbi = min(pj_tb_min_ngb_tb.x, min_ngb_tbi);

				const float inv_r = rsqrtf(r2);
				const float r     = r2 * inv_r;

				/* Get the kernel for hj */
				float wi, wi_dx, wj, wj_dx;

				const float ui = r * hi_inv;   // r / hi
				d_kernel_deval(ui, &wi, &wi_dx);
				const float wi_dr = hid_inv * wi_dx;

				const float hj_inv  = 1.0f / hj;
				const float hjd_inv = d_pow_dimension_plus_one(hj_inv); /* 1/h^(d+1) */
				const float uj      = r * hj_inv; // r / hj
				d_kernel_deval(uj, &wj, &wj_dx);
				const float wj_dr = hjd_inv * wj_dx;

				/* Compute dv dot r. */
				const float dvx = vxi - vxj;
				const float dvy = vyi - vyj;
				const float dvz = vzi - vzj;
				const float dvdr = fmaf(dvx, xij, fmaf(dvy, yij, dvz * zij)); // dv · r

				/* Includes the hubble flow term; not used for du/dt */
				const float dvdr_Hubble = dvdr + a2_Hubble * r2;

				/* Are the particles moving towards each others ? */
				const float omega_ij = fminf(dvdr_Hubble, 0.f);
				const float mu_ij    = fac_mu * inv_r * omega_ij; /* This is 0 or negative */

				/* Compute sound speeds and signal velocity */
				const float v_sig = ci + cj - const_viscosity_beta * mu_ij;

				/* Variable smoothing length term */
				const float f_ij = 1.f - fi * (1.f / mj);
				const float f_ji = 1.f - fj * mi_inv;

				/* Construct the full viscosity term */
				const float rhoij      = rhoi + rhoj;
				const float rhoij_inv  = 1.f / rhoij;
				const float alpha      = avisci + aviscj;
				const float visc       = -0.25f * alpha * v_sig * mu_ij * (balsi + balsj) * rhoij_inv;

				/* Convolve with the kernel */
				const float visc_acc_term = 0.5f * visc * (wi_dr * f_ij + wj_dr * f_ji) * inv_r;

				/* Compute gradient terms */
				const float rhoj2         = rhoj * rhoj;
				const float rhoj_inv      = 1.f / rhoj;
				const float P_over_rho2_i = pressurei * rhoi_inv2 * f_ij;
				const float P_over_rho2_j = pressurej * (1.f / rhoj2) * f_ji;

				const float sph_acc_term  = (P_over_rho2_i * wi_dr + P_over_rho2_j * wj_dr) * inv_r;

				/* Adaptive softening acceleration term */
				const float acc = sph_acc_term + visc_acc_term;

				/* Assemble the acceleration */
				res_ahydro.x -= mj * acc * xij;
				res_ahydro.y -= mj * acc * yij;
				res_ahydro.z -= mj * acc * zij;

				/* Get the time derivative for u. */
				const float sph_du_term_i = P_over_rho2_i * dvdr * inv_r * wi_dr;

				/* Viscosity term */
				const float visc_du_term  = 0.5f * visc_acc_term * dvdr_Hubble;

				/* Diffusion term */
				/* Combine the alpha_diff into a pressure-based switch -- this allows the
				 * alpha from the highest pressure particle to dominate, so that the
				 * diffusion limited particles always take precedence - another trick to
				 * allow the scheme to work with thermal feedback. */
				float alpha_diff = (pressurei * adiffi + pressurej * adiffj) / (pressurei + pressurej);
				// if (fabsf(pressurei + pressurej) < 1e-10f) alpha_diff = 0.f; // optional safe guard
				const float v_diff = alpha_diff * 0.5f *
						(sqrtf(2.f * fabsf(pressurei - pressurej) * rhoij_inv) +
								fabsf(fac_mu * inv_r * dvdr_Hubble));
				/* wi_dx + wj_dx / 2 is F_ij */
				const float diff_du_term = v_diff * (energyi - energyj) *
						(f_ij * wi_dr * rhoi_inv + f_ji * wj_dr * rhoj_inv);

				/* Assemble the energy equation term */
				const float du_dt_i = sph_du_term_i + visc_du_term + diff_du_term;

				/* Internal energy time derivative */
				res_udt_hdt.x += du_dt_i * mj;

				/* Get the time derivative for h. */
				res_udt_hdt.y -= mj * dvdr * inv_r * rhoj_inv * wi_dr;
			}
		}

		__syncthreads();
	}

	/*Conditional to prevent writing out of bounds */
	if (i_in_range) {

		atomicAdd(&d_parts_recv[i_id].a_hydro.x, res_ahydro.x);
		atomicAdd(&d_parts_recv[i_id].a_hydro.y, res_ahydro.y);
		atomicAdd(&d_parts_recv[i_id].a_hydro.z, res_ahydro.z);
		atomicAdd(&d_parts_recv[i_id].udt_hdt.x, res_udt_hdt.x);
		atomicAdd(&d_parts_recv[i_id].udt_hdt.y, res_udt_hdt.y);
		/* If minimum timebin calculated in this loop is not zero,
		 * compare with the global result and set to the minimum */
		if(min_ngb_tbi > 0)
			atomicMin(&d_parts_recv[i_id].minngbtb, min_ngb_tbi);

	}
}


__global__ void cuda_kernel_force(
    const struct gpu_part_send_f* __restrict__ d_parts_send,
    struct gpu_part_recv_f*      __restrict__ d_parts_recv,
    const float d_a, const float d_H,
    const int4* __restrict__ d_cell_i_j_start_end,
    const int2* __restrict__ d_block_leaf_id,
    const double3 space_dim){

	/*Figure out which range of particles threads in this block will work on*/
    const int bid     = blockIdx.x;
    /*What is the leaf computation this block will work on*/
    const int leafid  = d_block_leaf_id[bid].x;
    /*In case we need more than one block to run this leaf computation we need to know
     * where in the group of blocks acting on a cell we are.
     * bid_0 is the id of the first block acting on this cell*/
    const int bid_0   = d_block_leaf_id[bid].y;
    /*Get the start and end positions of cells i and j*/
    int4 cell_se      = d_cell_i_j_start_end[leafid];

    const int ci_start = cell_se.x;
    const int ci_end   = cell_se.y; // cell position at [ci_end-1]
    const int cj_start = cell_se.z;
    const int cj_end   = cell_se.w;

    /*We can now find our block index in a local reference to this cell*/
    const int b_id_local = bid - bid_0;
    const int tid = threadIdx.x;

    /* Get the cell positions. We store them as the last entry in the
     * particle array for each leaf computation */
    const auto ci_loc = d_parts_send[ci_end - 1].c_loc;
    const auto cj_loc = d_parts_send[cj_end - 1].c_loc;

    /*Calculate the shifted cell positions if we have periodics*/
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
        d_a, d_H);

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
            d_a, d_H);
    }
}

#ifdef __cplusplus
}
#endif

#endif /* CUDA_PARTICLE_KERNELS_CUH */
