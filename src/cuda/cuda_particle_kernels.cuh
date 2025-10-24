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
 * @brief Does the self density computation on device
 * TODO: parameter documentation
 */
__global__ void cuda_kernel_self_density(
    const struct gpu_part_send_d *__restrict__ parts_send,
    struct gpu_part_recv_d *__restrict__ parts_recv, const float d_a,
    const float d_H, const int bundle_first_task,
    const int2 *__restrict__ d_task_first_part_f4) {

  extern __shared__ float4 vars_f4[];

  const int threadid = blockDim.x * blockIdx.x + threadIdx.x;
  const int task_id = bundle_first_task + blockIdx.y;

  int2 first_last_parts = d_task_first_part_f4[task_id];
  int first_part_in_task_blocks = first_last_parts.x;
  int last_part_in_task_blocks = first_last_parts.y;

  const int pid = threadid + first_part_in_task_blocks;

  float4 res_rho = {0.0, 0.0, 0.0, 0.0};
  float4 res_rot = {0.0, 0.0, 0.0, 0.0};
  const struct gpu_part_send_d pi = parts_send[pid];
  const float4 x_pi = pi.x_p_h;
  const float4 ux_pi = pi.ux_m;
  const float hi = x_pi.w;
  const float hig2 = hi * hi * kernel_gamma2;
  /* Here we use different pointers "x_p_tmp", etc. to point to different
   * regions of the single shared memory space "vars" which we allocate in
   * kernel invocation */
  float4 *__restrict__ x_p_h_tmp = (float4 *)&vars_f4[0];
  float4 *__restrict__ ux_m_tmp = (float4 *)&vars_f4[GPU_THREAD_BLOCK_SIZE];
  /* Particles copied in blocks to shared memory */
  for (int b = first_part_in_task_blocks; b < last_part_in_task_blocks;
       b += GPU_THREAD_BLOCK_SIZE) {

    int j = b + threadIdx.x;
    struct gpu_part_send_d pj = parts_send[j];
    x_p_h_tmp[threadIdx.x] = pj.x_p_h;
    ux_m_tmp[threadIdx.x] = pj.ux_m;
    __syncthreads();
    for (int j_block = 0; j_block < GPU_THREAD_BLOCK_SIZE; j_block++) {
      j = j_block + b;
      if (j < last_part_in_task_blocks) {
        /* Compute the pairwise distance. */
        const float4 x_p_h_j = x_p_h_tmp[j_block];
        const float4 ux_m_j = ux_m_tmp[j_block];
        const float xij = x_pi.x - x_p_h_j.x;
        const float yij = x_pi.y - x_p_h_j.y;
        const float zij = x_pi.z - x_p_h_j.z;
        const float r2 = xij * xij + yij * yij + zij * zij;
        if (r2 < hig2 && r2 > (0.01f / 128.f) * (0.01f / 128.f)) {
          const float r = sqrtf(r2);
          /* Recover some data */
          const float mj = ux_m_j.w;
          /* Get the kernel for hi. */
          const float h_inv = 1.f / hi;
          const float ui = r * h_inv;
          float wi, wi_dx;

          d_kernel_deval(ui, &wi, &wi_dx);
          /*Add to sums of rho, rho_dh, wcount and wcount_dh*/
          res_rho.x += mj * wi;
          res_rho.y -= mj * (hydro_dimension * wi + ui * wi_dx);
          res_rho.z += wi;
          res_rho.w -= (hydro_dimension * wi + ui * wi_dx);

          const float r_inv = 1.f / r;
          const float faci = mj * wi_dx * r_inv;

          /* Compute dv dot r */
          const float dvx = ux_pi.x - ux_m_j.x;
          const float dvy = ux_pi.y - ux_m_j.y;
          const float dvz = ux_pi.z - ux_m_j.z;
          const float dvdr = dvx * xij + dvy * yij + dvz * zij;

          /* Compute dv cross r */
          const float curlvrx = dvy * zij - dvz * yij;
          const float curlvry = dvz * xij - dvx * zij;
          const float curlvrz = dvx * yij - dvy * xij;
          /*Add to sums of rot_u and div_v*/
          res_rot.x += faci * curlvrx;
          res_rot.y += faci * curlvry;
          res_rot.z += faci * curlvrz;
          res_rot.w -= faci * dvdr;
        }
      }
    }
    __syncthreads();
  }
  if (pid < last_part_in_task_blocks) {
    parts_recv[pid].rho_dh_wcount = res_rho;
    parts_recv[pid].rot_ux_div_v = res_rot;
  }
}

/**
 * Does the self gradient computation on device
 * TODO: parameter documentation
 */
__global__ void cuda_kernel_self_gradient(
    const struct gpu_part_send_g *__restrict__ parts_send,
    struct gpu_part_recv_g *__restrict__ parts_recv, const float d_a,
    const float d_H, const int bundle_first_task,
    const int2 *__restrict__ d_task_first_part_f4) {

  extern __shared__ float4 varsf4_g[];

  const int threadid = blockDim.x * blockIdx.x + threadIdx.x;
  const int task_id = bundle_first_task + blockIdx.y;

  int2 first_last_parts = d_task_first_part_f4[task_id];
  int first_part_in_task_blocks = first_last_parts.x;
  int last_part_in_task_blocks = first_last_parts.y;
  const int pid = threadid + first_part_in_task_blocks;

  struct gpu_part_send_g pi = parts_send[pid];
  float4 x_h_i = pi.x_h;
  float4 ux_m_i = pi.ux_m;
  float4 rho_avisc_u_c_i = pi.rho_avisc_u_c;
  float3 vsig_lapu_aviscmax_i = {0.f, 0.f, 0.f};

  const float hi = x_h_i.w;
  const float hig2 = hi * hi * kernel_gamma2;

  /* Here we use different pointers "x_p_tmp", etc. to point to different
   * regions of the single shared memory space "vars" which we allocate in
   * kernel invocation */
  float4 *__restrict__ x_h_tmp = (float4 *)&varsf4_g[0];
  float4 *__restrict__ ux_m_tmp = (float4 *)&varsf4_g[GPU_THREAD_BLOCK_SIZE];
  float4 *__restrict__ rho_avisc_u_c_tmp =
      (float4 *)&varsf4_g[GPU_THREAD_BLOCK_SIZE * 2];

  /*Particles copied in blocks to shared memory*/
  for (int b = first_part_in_task_blocks; b < last_part_in_task_blocks;
       b += GPU_THREAD_BLOCK_SIZE) {

    int j = b + threadIdx.x;

    struct gpu_part_send_g pj = parts_send[j];
    x_h_tmp[threadIdx.x] = pj.x_h;
    ux_m_tmp[threadIdx.x] = pj.ux_m;
    rho_avisc_u_c_tmp[threadIdx.x] = pj.rho_avisc_u_c;

    __syncthreads();
    for (int j_block = 0; j_block < GPU_THREAD_BLOCK_SIZE; j_block++) {
      j = j_block + b;
      if (j < last_part_in_task_blocks) {
        float4 x_h_j = x_h_tmp[j_block];
        float4 ux_m_j = ux_m_tmp[j_block];
        float4 rho_avisc_u_c_j = rho_avisc_u_c_tmp[j_block];
        /* Compute the pairwise distance. */
        const float xij = x_h_i.x - x_h_j.x;
        const float yij = x_h_i.y - x_h_j.y;
        const float zij = x_h_i.z - x_h_j.z;
        const float r2 = xij * xij + yij * yij + zij * zij;

        if (r2 < hig2 && r2 > (0.01f / 128.f) * (0.01f / 128.f)) {
          const float r = sqrt(r2);
          const float r_inv = 1.f / r;
          /* Recover some data */
          const float mj = ux_m_j.w;
          /* Get the kernel for hi. */
          const float h_inv = 1.f / hi;
          float wi, wi_dx;
          /* Cosmology terms for the signal velocity */
          const float fac_mu = d_pow_three_gamma_minus_five_over_two(d_a);
          const float a2_Hubble = d_a * d_a * d_H;
          /* Compute dv dot r */
          float dvx = ux_m_i.x - ux_m_j.x;
          float dvy = ux_m_i.y - ux_m_j.y;
          float dvz = ux_m_i.z - ux_m_j.z;
          const float dvdr = dvx * xij + dvy * yij + dvz * zij;
          /* Add Hubble flow */
          const float dvdr_Hubble = dvdr + a2_Hubble * r2;
          /* Are the particles moving towards each others ? */
          const float omega_ij = min(dvdr_Hubble, 0.f);
          const float mu_ij =
              fac_mu * r_inv * omega_ij; /* This is 0 or negative */

          /* Signal velocity */
          const float new_v_sig = rho_avisc_u_c_i.w + rho_avisc_u_c_j.w -
                                  const_viscosity_beta * mu_ij;
          /* Update if we need to */
          vsig_lapu_aviscmax_i.x = fmaxf(vsig_lapu_aviscmax_i.x, new_v_sig);
          /* Calculate Del^2 u for the thermal diffusion coefficient. */
          /* Need to get some kernel values F_ij = wi_dx */
          const float ui = r * h_inv;
          d_kernel_deval(ui, &wi, &wi_dx);

          const float delta_u_factor =
              (rho_avisc_u_c_i.z - rho_avisc_u_c_j.z) * r_inv;
          vsig_lapu_aviscmax_i.y +=
              mj * delta_u_factor * wi_dx / rho_avisc_u_c_j.x;

          /* Set the maximal alpha from the previous step over the neighbours
           * (this is used to limit the diffusion in hydro_prepare_force) */
          const float alpha_j = rho_avisc_u_c_j.y;
          vsig_lapu_aviscmax_i.z = fmaxf(vsig_lapu_aviscmax_i.z, alpha_j);
        }
      }
    }
    __syncthreads();
  }
  if (pid < last_part_in_task_blocks) {
    parts_recv[pid].vsig_lapu_aviscmax = vsig_lapu_aviscmax_i;
  }
}

/**
 * @brief Does the self force computation on device
 * TODO: parameter documentation
 */
__global__ void cuda_kernel_self_force(
    const struct gpu_part_send_f *__restrict__ parts_send,
    struct gpu_part_recv_f *__restrict__ parts_recv, const float d_a,
    const float d_H, const int bundle_first_task,
    const int2 *__restrict__ d_task_first_part_f4) {

  extern __shared__ float4 varsf4_f[];

  const int threadid = blockDim.x * blockIdx.x + threadIdx.x;
  const int task_id = bundle_first_task + blockIdx.y;

  int2 first_last_parts = d_task_first_part_f4[task_id];
  int first_part_in_task_blocks = first_last_parts.x;
  int last_part_in_task_blocks = first_last_parts.y;

  const int pid = threadid + first_part_in_task_blocks;

  const gpu_part_send_f pi = parts_send[pid];
  float4 x_h_i = pi.x_h;
  float4 ux_m_i = pi.ux_m;
  float4 f_b_t_mintbinngb_i = pi.f_bals_timebin_mintimebin_ngb;
  float4 rho_p_c_vsig_i = pi.rho_p_c_vsigi;
  float3 u_avisc_adiff_i = pi.u_alphavisc_alphadiff;

  const float mi = ux_m_i.w;
  float pressurei = rho_p_c_vsig_i.y;
  const float ci = rho_p_c_vsig_i.z;
  float3 ahydro = {0.0, 0.0, 0.0};
  float4 udt_hdt_vsig_mintbinngb = {0.0, 0.0, 0.0, 0.0};
  udt_hdt_vsig_mintbinngb.z = rho_p_c_vsig_i.w;
  udt_hdt_vsig_mintbinngb.w = f_b_t_mintbinngb_i.w;

  float hi = x_h_i.w;
  float hig2 = hi * hi * kernel_gamma2;

  /*Here we use different pointers "x_p_tmp", etc. to point to different regions
   * of the single shared memory space "vars" which we allocate in kernel
   * invocation*/
  float4 *__restrict__ x_h_tmp = (float4 *)&varsf4_f[0];
  float4 *__restrict__ ux_m_tmp = (float4 *)&varsf4_f[GPU_THREAD_BLOCK_SIZE];
  float4 *__restrict__ f_b_t_mintbinngb_tmp =
      (float4 *)&varsf4_f[GPU_THREAD_BLOCK_SIZE * 2];
  float4 *__restrict__ rho_p_c_vsig_tmp =
      (float4 *)&varsf4_f[GPU_THREAD_BLOCK_SIZE * 3];
  float3 *__restrict__ u_avisc_adiff_tmp =
      (float3 *)&varsf4_f[GPU_THREAD_BLOCK_SIZE * 4];

  /*Particles copied in blocks to shared memory*/
  for (int b = first_part_in_task_blocks; b < last_part_in_task_blocks;
       b += GPU_THREAD_BLOCK_SIZE) {
    int j = b + threadIdx.x;
    struct gpu_part_send_f pj = parts_send[j];
    x_h_tmp[threadIdx.x] = pj.x_h;
    ux_m_tmp[threadIdx.x] = pj.ux_m;
    f_b_t_mintbinngb_tmp[threadIdx.x] = pj.f_bals_timebin_mintimebin_ngb;
    rho_p_c_vsig_tmp[threadIdx.x] = pj.rho_p_c_vsigi;
    //    alpha_tmp[threadIdx.x] = parts_aos[j].visc_alpha;
    u_avisc_adiff_tmp[threadIdx.x] = pj.u_alphavisc_alphadiff;
    __syncthreads();
    for (int j_block = 0; j_block < GPU_THREAD_BLOCK_SIZE; j_block++) {
      j = j_block + b;
      if (j < last_part_in_task_blocks) {
        /* Compute the pairwise distance. */
        float4 x_h_j = x_h_tmp[j_block];
        float4 ux_m_j = ux_m_tmp[j_block];
        float4 f_b_t_mintbinngb_j = f_b_t_mintbinngb_tmp[j_block];
        float4 rho_p_c_vsig_j = rho_p_c_vsig_tmp[j_block];
        float3 u_avisc_adiff_j = u_avisc_adiff_tmp[j_block];
        const float xij = x_h_i.x - x_h_j.x;
        const float yij = x_h_i.y - x_h_j.y;
        const float zij = x_h_i.z - x_h_j.z;
        const float hj = x_h_j.w;
        const float hjg2 = hj * hj * kernel_gamma2;
        const float r2 = xij * xij + yij * yij + zij * zij;
        if ((r2 < hig2 || r2 < hjg2) &&
            r2 > (0.01f / 128.f) * (0.01f / 128.f)) {
          //          /* Cosmology terms for the signal velocity */
          const float fac_mu = d_pow_three_gamma_minus_five_over_two(d_a);
          const float a2_Hubble = d_a * d_a * d_H;
          const float r = sqrt(r2);
          const float r_inv = 1.f / r;
          //          /* Recover some data */
          const float mj = ux_m_j.w;
          //          /* Get the kernel for hi. */
          const float hi_inv = 1.f / hi;
          const float hid_inv =
              d_pow_dimension_plus_one(hi_inv); /* 1/h^(d+1) */
          const float xi = r * hi_inv;
          float wi, wi_dx;
          d_kernel_deval(xi, &wi, &wi_dx);
          const float wi_dr = hid_inv * wi_dx;
          /* Get the kernel for hj. */
          const float hj = x_h_j.w;
          const float hj_inv = 1.0f / hj;
          const float hjd_inv =
              d_pow_dimension_plus_one(hj_inv); /* 1/h^(d+1) */
          const float xj = r * hj_inv;
          float wj, wj_dx;
          d_kernel_deval(xj, &wj, &wj_dx);
          const float wj_dr = hjd_inv * wj_dx;
          //          /* Compute dv dot r */
          float dvx = ux_m_i.x - ux_m_j.x;
          float dvy = ux_m_i.y - ux_m_j.y;
          float dvz = ux_m_i.z - ux_m_j.z;
          const float dvdr = dvx * xij + dvy * yij + dvz * zij;
          //          /* Add Hubble flow */
          const float dvdr_Hubble = dvdr + a2_Hubble * r2;
          //          /* Are the particles moving towards each others ? */
          const float omega_ij = min(dvdr_Hubble, 0.f);
          const float mu_ij =
              fac_mu * r_inv * omega_ij; /* This is 0 or negative */
                                         //
                                         //          /* Signal velocity */
          const float cj = rho_p_c_vsig_j.z;
          const float v_sig = ci + cj - const_viscosity_beta * mu_ij;

          /* Variable smoothing length term */
          const float f_ij = 1.f - f_b_t_mintbinngb_i.x / mj;
          const float f_ji = 1.f - f_b_t_mintbinngb_j.x / mi;

          /* Construct the full viscosity term */
          const float pressurej = rho_p_c_vsig_j.y;
          const float rho_ij = rho_p_c_vsig_i.x + rho_p_c_vsig_j.x;
          const float alpha = u_avisc_adiff_i.y + u_avisc_adiff_j.y;
          const float visc = -0.25f * alpha * v_sig * mu_ij *
                             (f_b_t_mintbinngb_i.y + f_b_t_mintbinngb_j.y) /
                             rho_ij;
          /* Convolve with the kernel */
          const float visc_acc_term =
              0.5f * visc * (wi_dr * f_ij + wj_dr * f_ji) * r_inv;
          /* Compute gradient terms */
          const float rhoi2 = rho_p_c_vsig_i.x * rho_p_c_vsig_i.x;
          const float rhoj2 = rho_p_c_vsig_j.x * rho_p_c_vsig_j.x;
          const float P_over_rho2_i = pressurei / (rhoi2)*f_ij;
          const float P_over_rho2_j = pressurej / (rhoj2)*f_ji;

          /* SPH acceleration term */
          const float sph_acc_term =
              (P_over_rho2_i * wi_dr + P_over_rho2_j * wj_dr) * r_inv;

          /* Assemble the acceleration */
          const float acc = sph_acc_term + visc_acc_term;
          /* Use the force Luke ! */
          ahydro.x -= mj * acc * xij;
          ahydro.y -= mj * acc * yij;
          ahydro.z -= mj * acc * zij;
          /* Get the time derivative for u. */
          const float sph_du_term_i = P_over_rho2_i * dvdr * r_inv * wi_dr;

          /* Viscosity term */
          const float visc_du_term = 0.5f * visc_acc_term * dvdr_Hubble;
          /* Diffusion term */
          /* Combine the alpha_diff into a pressure-based switch -- this allows
           * the alpha from the highest pressure particle to dominate, so that
           * the diffusion limited particles always take precedence - another
           * trick to allow the scheme to work with thermal feedback. */
          float alpha_diff =
              (pressurei * u_avisc_adiff_i.z + pressurej * u_avisc_adiff_j.z) /
              (pressurei + pressurej);
          if (fabsf(pressurei + pressurej) < 1e-10) alpha_diff = 0.f;
          const float v_diff =
              alpha_diff * 0.5f *
              (sqrtf(2.f * fabsf(pressurei - pressurej) / rho_ij) +
               fabsf(fac_mu * r_inv * dvdr_Hubble));
          /* wi_dx + wj_dx / 2 is F_ij */
          const float diff_du_term = v_diff *
                                     (u_avisc_adiff_i.x - u_avisc_adiff_j.x) *
                                     (f_ij * wi_dr / rho_p_c_vsig_i.x +
                                      f_ji * wj_dr / rho_p_c_vsig_j.x);

          /* Assemble the energy equation term */
          const float du_dt_i = sph_du_term_i + visc_du_term + diff_du_term;

          /* Internal energy time derivative */
          udt_hdt_vsig_mintbinngb.x += du_dt_i * mj;

          /* Get the time derivative for h. */
          udt_hdt_vsig_mintbinngb.y -=
              mj * dvdr * r_inv / rho_p_c_vsig_j.x * wi_dr;

          /* Update if we need to; this should be guaranteed by the gradient
           * loop but due to some possible synchronisation problems this is here
           * as a _quick fix_. Added: 14th August 2019. To be removed by 1st Jan
           * 2020. (JB) */
          udt_hdt_vsig_mintbinngb.z = fmaxf(udt_hdt_vsig_mintbinngb.z, v_sig);
          unsigned int time_bin_j = (f_b_t_mintbinngb_j.z + 0.5f);
          unsigned int min_tb_i = (f_b_t_mintbinngb_i.w + 0.5f);
          if (time_bin_j > 0) f_b_t_mintbinngb_i.w = min(min_tb_i, time_bin_j);
          //          printf("Got in\n");
        }
      }
    }
    __syncthreads();
  }
  if (pid < last_part_in_task_blocks) {
    udt_hdt_vsig_mintbinngb.w = f_b_t_mintbinngb_i.w;
    parts_recv[pid].udt_hdt_vsig_mintimebin_ngb = udt_hdt_vsig_mintbinngb;
    parts_recv[pid].a_hydro = ahydro;
  }
}

/**
 * @brief Naive kernel computing density interaction for pair task
 * TODO: parameter documentation
 */
__device__ __attribute__((always_inline)) INLINE void cuda_kernel_density(
    const struct gpu_part_send_d pi,
    const struct gpu_part_send_d *__restrict__ d_parts_send,
    struct gpu_part_recv_d *__restrict__ d_parts_recv, int pid,
    const int cj_start, const int cj_end, float d_a, float d_H) {

  float hi = 0.0;
  float hig2 = 0.0;

  float4 res_rho = {0.0, 0.0, 0.0, 0.0};
  float4 res_rot = {0.0, 0.0, 0.0, 0.0};
  const float4 x_pi = pi.x_p_h;
  const float4 ux_pi = pi.ux_m;
  hi = x_pi.w, hig2 = hi * hi * kernel_gamma2;

  /* Particles copied in blocks to shared memory */
  for (int j = cj_start; j < cj_end; j++) {
    struct gpu_part_send_d pj = d_parts_send[j];

    const float4 x_p_h_j = pj.x_p_h;
    const float4 ux_m_j = pj.ux_m;

    const float xij = x_pi.x - x_p_h_j.x;
    const float yij = x_pi.y - x_p_h_j.y;
    const float zij = x_pi.z - x_p_h_j.z;
    const float r2 = xij * xij + yij * yij + zij * zij;
    /*Small & constant distance to ensure we don't interact a particle with itself*/
    const float epsilon = 1e-8;
    if (r2 < hig2 && r2 > epsilon) {
      /* Recover some data */
      const float mj = ux_m_j.w;
      const float r = sqrt(r2);
      /* Get the kernel for hi. */
      const float h_inv = 1.f / hi;
      const float ui = r * h_inv;
      float wi, wi_dx;
      d_kernel_deval(ui, &wi, &wi_dx);

      /*Add to sums of rho, rho_dh, wcount and wcount_dh*/
      res_rho.x += mj * wi;
      res_rho.y -= mj * (hydro_dimension * wi + ui * wi_dx);
      res_rho.z += wi;
      res_rho.w -= (hydro_dimension * wi + ui * wi_dx);

      const float r_inv = 1.f / r;
      const float faci = mj * wi_dx * r_inv;
      /* Compute dv dot r */
      const float dvx = ux_pi.x - ux_m_j.x;
      const float dvy = ux_pi.y - ux_m_j.y;
      const float dvz = ux_pi.z - ux_m_j.z;
      const float dvdr = dvx * xij + dvy * yij + dvz * zij;
      /* Compute dv cross r */
      const float curlvrx = dvy * zij - dvz * yij;
      const float curlvry = dvz * xij - dvx * zij;
      const float curlvrz = dvx * yij - dvy * xij;

      res_rot.x += faci * curlvrx;
      res_rot.y += faci * curlvry;
      res_rot.z += faci * curlvrz;
      res_rot.w -= faci * dvdr;
    }
  } /*Loop through parts in cell j one GPU_THREAD_BLOCK_SIZE at a time*/

  d_parts_recv[pid].rho_dh_wcount = res_rho;
  d_parts_recv[pid].rot_ux_div_v = res_rot;
}

/**
 * Naive kernel computing gradient interaction for pair task
 * TODO: parameter documentation
 */
__device__ __attribute__((always_inline)) INLINE void cuda_kernel_pair_gradient(
    const struct gpu_part_send_g pi,
    const struct gpu_part_send_g *__restrict__ d_parts_send,
    struct gpu_part_recv_g *__restrict__ d_parts_recv, int pid,
    const int cj_start, const int cj_end, float d_a, float d_H) {

  float hi = 0.0;
  float hig2 = 0.0;

  const float4 x_h_i = pi.x_h;
  const float4 ux_m_i = pi.ux_m;
  const float4 rho_avisc_u_c_i = pi.rho_avisc_u_c;
  float3 vsig_lapu_aviscmax_i = {0.f, 0.f, 0.f};

  hi = x_h_i.w, hig2 = hi * hi * kernel_gamma2;

  /* Particles copied in blocks to shared memory */
  for (int j = cj_start; j < cj_end; j++) {
    struct gpu_part_send_g pj = d_parts_send[j];

    const float4 x_h_j = pj.x_h;
    const float4 ux_m_j = pj.ux_m;
    const float4 rho_avisc_u_c_j = pj.rho_avisc_u_c;
    const float xij = x_h_i.x - x_h_j.x;
    const float yij = x_h_i.y - x_h_j.y;
    const float zij = x_h_i.z - x_h_j.z;
    const float r2 = xij * xij + yij * yij + zij * zij;

    if (r2 < hig2) {
      const float r = sqrt(r2);
      const float r_inv = 1.f / r;
      /* Recover some data */
      const float mj = ux_m_j.w;
      /* Get the kernel for hi. */
      const float h_inv = 1.f / hi;
      float wi, wi_dx;
      /* Cosmology terms for the signal velocity */
      const float fac_mu = d_pow_three_gamma_minus_five_over_two(d_a);
      const float a2_Hubble = d_a * d_a * d_H;
      /* Compute dv dot r */
      float dvx = ux_m_i.x - ux_m_j.x;
      float dvy = ux_m_i.y - ux_m_j.y;
      float dvz = ux_m_i.z - ux_m_j.z;
      const float dvdr = dvx * xij + dvy * yij + dvz * zij;
      /* Add Hubble flow */
      const float dvdr_Hubble = dvdr + a2_Hubble * r2;
      /* Are the particles moving towards each others ? */
      const float omega_ij = min(dvdr_Hubble, 0.f);
      const float mu_ij = fac_mu * r_inv * omega_ij; /* This is 0 or negative */

      /* Signal velocity */
      const float new_v_sig =
          rho_avisc_u_c_i.w + rho_avisc_u_c_j.w - const_viscosity_beta * mu_ij;
      /* Update if we need to */
      vsig_lapu_aviscmax_i.x = fmaxf(vsig_lapu_aviscmax_i.x, new_v_sig);
      /* Calculate Del^2 u for the thermal diffusion coefficient. */
      /* Need to get some kernel values F_ij = wi_dx */
      const float ui = r * h_inv;
      d_kernel_deval(ui, &wi, &wi_dx);

      const float delta_u_factor =
          (rho_avisc_u_c_i.z - rho_avisc_u_c_j.z) * r_inv;
      vsig_lapu_aviscmax_i.y += mj * delta_u_factor * wi_dx / rho_avisc_u_c_j.x;

      /* Set the maximal alpha from the previous step over the neighbours
       * (this is used to limit the diffusion in hydro_prepare_force) */
      const float alpha_j = rho_avisc_u_c_j.y;
      vsig_lapu_aviscmax_i.z = fmaxf(vsig_lapu_aviscmax_i.z, alpha_j);
    }
  } /*Loop through parts in cell j one GPU_THREAD_BLOCK_SIZE at a time*/

  d_parts_recv[pid].vsig_lapu_aviscmax = vsig_lapu_aviscmax_i;
}

/**
 * Naive kernel computing force interaction for pair task
 * TODO: parameter documentation
 */
__device__ __attribute__((always_inline)) INLINE void cuda_kernel_pair_force(
    const struct gpu_part_send_f pi,
    const struct gpu_part_send_f *__restrict__ d_parts_send,
    struct gpu_part_recv_f *__restrict__ d_parts_recv, int pid,
    const int cj_start, const int cj_end, float d_a, float d_H) {

  const float4 x_h_i = pi.x_h;
  const float4 ux_m_i = pi.ux_m;

  float4 f_b_t_mintbinngb_i = pi.f_bals_timebin_mintimebin_ngb;
  const float4 rho_p_c_vsig_i = pi.rho_p_c_vsigi;
  const float3 u_avisc_adiff_i = pi.u_alphavisc_alphadiff;

  const float mi = ux_m_i.w;
  const float pressurei = rho_p_c_vsig_i.y;
  const float ci = rho_p_c_vsig_i.z;
  float3 ahydro = {0.0, 0.0, 0.0};
  float4 udt_hdt_vsig_mintbinngb = {0.0, 0.0, 0.0, 0.0};
  udt_hdt_vsig_mintbinngb.z = rho_p_c_vsig_i.w;
  udt_hdt_vsig_mintbinngb.w = f_b_t_mintbinngb_i.w;

  const float hi = x_h_i.w;
  const float hig2 = hi * hi * kernel_gamma2;

  /* Particles copied in blocks to shared memory */
  for (int j = cj_start; j < cj_end; j++) {
    struct gpu_part_send_f pj = d_parts_send[j];
    const float4 x_h_j = pj.x_h;
    const float4 ux_m_j = pj.ux_m;
    const float4 f_b_t_mintbinngb_j = pj.f_bals_timebin_mintimebin_ngb;
    const float4 rho_p_c_vsig_j = pj.rho_p_c_vsigi;
    //    alpha_tmp[threadIdx.x] = parts_aos[j].visc_alpha;
    const float3 u_avisc_adiff_j = pj.u_alphavisc_alphadiff;
    const float xij = x_h_i.x - x_h_j.x;
    const float yij = x_h_i.y - x_h_j.y;
    const float zij = x_h_i.z - x_h_j.z;
    const float hj = x_h_j.w;
    const float hjg2 = hj * hj * kernel_gamma2;
    const float r2 = xij * xij + yij * yij + zij * zij;

    if (r2 < hig2 || r2 < hjg2) {
      //          /* Cosmology terms for the signal velocity */
      const float fac_mu = d_pow_three_gamma_minus_five_over_two(d_a);
      const float a2_Hubble = d_a * d_a * d_H;
      const float r = sqrt(r2);
      const float r_inv = 1.f / r;
      //          /* Recover some data */
      const float mj = ux_m_j.w;
      //          /* Get the kernel for hi. */
      const float hi_inv = 1.f / hi;
      const float hid_inv = d_pow_dimension_plus_one(hi_inv); /* 1/h^(d+1) */
      const float xi = r * hi_inv;
      float wi, wi_dx;
      d_kernel_deval(xi, &wi, &wi_dx);
      const float wi_dr = hid_inv * wi_dx;
      /* Get the kernel for hj. */
      const float hj = x_h_j.w;
      const float hj_inv = 1.0f / hj;
      const float hjd_inv = d_pow_dimension_plus_one(hj_inv); /* 1/h^(d+1) */
      const float xj = r * hj_inv;
      float wj, wj_dx;
      d_kernel_deval(xj, &wj, &wj_dx);
      const float wj_dr = hjd_inv * wj_dx;
      //          /* Compute dv dot r */
      float dvx = ux_m_i.x - ux_m_j.x;
      float dvy = ux_m_i.y - ux_m_j.y;
      float dvz = ux_m_i.z - ux_m_j.z;
      const float dvdr = dvx * xij + dvy * yij + dvz * zij;
      //          /* Add Hubble flow */
      const float dvdr_Hubble = dvdr + a2_Hubble * r2;
      //          /* Are the particles moving towards each others ? */
      const float omega_ij = min(dvdr_Hubble, 0.f);
      const float mu_ij = fac_mu * r_inv * omega_ij; /* This is 0 or negative */
                                                     //
      //          /* Signal velocity */
      const float cj = rho_p_c_vsig_j.z;
      const float v_sig = ci + cj - const_viscosity_beta * mu_ij;

      /* Variable smoothing length term */
      const float f_ij = 1.f - f_b_t_mintbinngb_i.x / mj;
      const float f_ji = 1.f - f_b_t_mintbinngb_j.x / mi;

      /* Construct the full viscosity term */
      const float pressurej = rho_p_c_vsig_j.y;
      const float rho_ij = rho_p_c_vsig_i.x + rho_p_c_vsig_j.x;
      const float alpha = u_avisc_adiff_i.y + u_avisc_adiff_j.y;
      const float visc = -0.25f * alpha * v_sig * mu_ij *
                         (f_b_t_mintbinngb_i.y + f_b_t_mintbinngb_j.y) / rho_ij;
      /* Convolve with the kernel */
      const float visc_acc_term =
          0.5f * visc * (wi_dr * f_ij + wj_dr * f_ji) * r_inv;
      /* Compute gradient terms */
      const float rhoi2 = rho_p_c_vsig_i.x * rho_p_c_vsig_i.x;
      const float rhoj2 = rho_p_c_vsig_j.x * rho_p_c_vsig_j.x;
      const float P_over_rho2_i = pressurei / (rhoi2)*f_ij;
      const float P_over_rho2_j = pressurej / (rhoj2)*f_ji;

      /* SPH acceleration term */
      const float sph_acc_term =
          (P_over_rho2_i * wi_dr + P_over_rho2_j * wj_dr) * r_inv;

      /* Assemble the acceleration */
      const float acc = sph_acc_term + visc_acc_term;
      /* Use the force Luke ! */
      ahydro.x -= mj * acc * xij;
      ahydro.y -= mj * acc * yij;
      ahydro.z -= mj * acc * zij;
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
          (pressurei * u_avisc_adiff_i.z + pressurej * u_avisc_adiff_j.z) /
          (pressurei + pressurej);
      if (fabsf(pressurei + pressurej) < 1e-10) alpha_diff = 0.f;
      const float v_diff = alpha_diff * 0.5f *
                           (sqrtf(2.f * fabsf(pressurei - pressurej) / rho_ij) +
                            fabsf(fac_mu * r_inv * dvdr_Hubble));
      /* wi_dx + wj_dx / 2 is F_ij */
      const float diff_du_term =
          v_diff * (u_avisc_adiff_i.x - u_avisc_adiff_j.x) *
          (f_ij * wi_dr / rho_p_c_vsig_i.x + f_ji * wj_dr / rho_p_c_vsig_j.x);

      /* Assemble the energy equation term */
      const float du_dt_i = sph_du_term_i + visc_du_term + diff_du_term;

      /* Internal energy time derivative */
      udt_hdt_vsig_mintbinngb.x += du_dt_i * mj;

      /* Get the time derivative for h. */
      udt_hdt_vsig_mintbinngb.y -= mj * dvdr * r_inv / rho_p_c_vsig_j.x * wi_dr;

      /* Update if we need to; this should be guaranteed by the gradient loop
       * but due to some possible synchronisation problems this is here as a
       * _quick fix_. Added: 14th August 2019. To be removed by 1st Jan 2020.
       * (JB) */
      udt_hdt_vsig_mintbinngb.z = fmaxf(udt_hdt_vsig_mintbinngb.z, v_sig);
      unsigned int time_bin_j = (f_b_t_mintbinngb_j.z + 0.5f);
      unsigned int min_tb_i = (f_b_t_mintbinngb_i.w + 0.5f);
      if (time_bin_j > 0) f_b_t_mintbinngb_i.w = min(min_tb_i, time_bin_j);
    }
  } /*Loop through parts in cell j one GPU_THREAD_BLOCK_SIZE at a time*/

  udt_hdt_vsig_mintbinngb.w = f_b_t_mintbinngb_i.w;
  d_parts_recv[pid].udt_hdt_vsig_mintimebin_ngb = udt_hdt_vsig_mintbinngb;
  d_parts_recv[pid].a_hydro = ahydro;
}

#ifdef __cplusplus
}
#endif

#endif /* CUDA_PARTICLE_KERNELS_CUH */
