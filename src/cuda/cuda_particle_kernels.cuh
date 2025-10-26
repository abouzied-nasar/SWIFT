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
__device__ __attribute__((always_inline)) INLINE void cuda_kernel_density(
    int pid, const struct gpu_part_send_d *__restrict__ d_parts_send,
    struct gpu_part_recv_d *__restrict__ d_parts_recv, float d_a, float d_H) {

  /* First, grab handles. */
  const struct gpu_part_send_d pi = d_parts_send[pid];

  const float xi = pi.x_h.x;
  const float yi = pi.x_h.y;
  const float zi = pi.x_h.z;
  const float hi = pi.x_h.w;

  const float vxi = pi.vx_m.x;
  const float vyi = pi.vx_m.y;
  const float vzi = pi.vx_m.z;
  /* const float mi = pi.vx_m.w; */

  const int pj_start = pi.pjs_pje.x;
  const int pj_end = pi.pjs_pje.y;

  /* Do some auxiliary computations */
  const float hig2 = hi * hi * kernel_gamma2;
  const float hi_inv = 1.f / hi;

  /* Prep output */
  /* rho, rho_dh, wcount, wcount_dh */
  float4 res_rho = {0.0, 0.0, 0.0, 0.0};
  /* curl of velocity (3 coordinates), velocity divergence */
  float4 res_rot = {0.0, 0.0, 0.0, 0.0};

  /* Start the neighbour interactions */
  for (int j = pj_start; j < pj_end; j++) {

    /* First, grab handles. */
    const struct gpu_part_send_d pj = d_parts_send[j];

    const float xj = pj.x_h.x;
    const float yj = pj.x_h.y;
    const float zj = pj.x_h.z;
    /* const float hj = pj.x_p_h.w; */

    const float vxj = pj.vx_m.x;
    const float vyj = pj.vx_m.y;
    const float vzj = pj.vx_m.z;
    const float mj = pj.vx_m.w;

    /* Now get stuff done. */
    const float xij = xi - xj;
    const float yij = yi - yj;
    const float zij = zi - zj;
    const float r2 = xij * xij + yij * yij + zij * zij;

    if ((r2 < hig2) && (j != pid)) {
      /* j != pid: Exclude self contribution. This happens at a later step. */

      /* Recover some data */
      const float r = sqrtf(r2);

      /* Get the kernel for hi. */
      const float ui = r * hi_inv;
      float wi;
      float wi_dx;
      d_kernel_deval(ui, &wi, &wi_dx);

      /* Add to sums of rho, rho_dh, wcount and wcount_dh */
      res_rho.x += mj * wi;
      res_rho.y -= mj * (hydro_dimension * wi + ui * wi_dx);
      res_rho.z += wi;
      res_rho.w -= (hydro_dimension * wi + ui * wi_dx);

      const float r_inv = 1.f / r;
      const float faci = mj * wi_dx * r_inv;

      /* Compute dv dot r */
      const float dvx = vxi - vxj;
      const float dvy = vyi - vyj;
      const float dvz = vzi - vzj;
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

  /* Write results. */
  d_parts_recv[pid].rho_rhodh_wcount_wcount_dh = res_rho;
  d_parts_recv[pid].rot_vx_div_v = res_rot;
}

/**
 * @brief Naive kernel computing the gradient interactions of a single particle
 *
 * @param pid index of particle to compute density for in the data arrays
 * @param d_pars_send array of particle data received from CPU
 * @param d_parts_recv array of particle data to write results into
 * @param d_a current cosmological expansion factor
 * @param d_H current Hubble constant
 */
__device__ __attribute__((always_inline)) INLINE void cuda_kernel_gradient(
    int pid, const struct gpu_part_send_g *__restrict__ d_parts_send,
    struct gpu_part_recv_g *__restrict__ d_parts_recv, float d_a, float d_H) {

  /* First, grab handles. */
  const struct gpu_part_send_g pi = d_parts_send[pid];

  const float xi = pi.x_h.x;
  const float yi = pi.x_h.y;
  const float zi = pi.x_h.z;
  const float hi = pi.x_h.w;

  const float vxi = pi.vx_m.x;
  const float vyi = pi.vx_m.y;
  const float vzi = pi.vx_m.z;
  /* const float mi = pi.vx_m.w; */

  /* const float rhoi = pi.rho_avisc_u_c.x; */
  /* const float avisci = pi.rho_avisc_u_c.y; */
  const float energyi = pi.rho_avisc_u_c.z;
  const float ci = pi.rho_avisc_u_c.w;

  const float vsigi = pi.vsig_lapu_aviscmax.x;
  const float lapui = pi.vsig_lapu_aviscmax.y;
  const float avisc_maxi = pi.vsig_lapu_aviscmax.z;

  const int pj_start = pi.pjs_pje.x;
  const int pj_end = pi.pjs_pje.y;

  /* Do some auxiliary computations */
  const float hig2 = hi * hi * kernel_gamma2;
  const float hi_inv = 1.f / hi;

  /* Prep output */
  /* v_sig, laplace_u, a_viscosity_max */
  float3 res_vsig_lapu_avisci = {vsigi, lapui, avisc_maxi};

  /* Start the neighbour interactions */
  for (int j = pj_start; j < pj_end; j++) {

    /* First, grab handles. */
    const struct gpu_part_send_g pj = d_parts_send[pid];

    const float xj = pj.x_h.x;
    const float yj = pj.x_h.y;
    const float zj = pj.x_h.z;
    /* const float hj = pj.x_h.w; */

    const float vxj = pj.vx_m.x;
    const float vyj = pj.vx_m.y;
    const float vzj = pj.vx_m.z;
    const float mj = pj.vx_m.w;

    const float rhoj = pj.rho_avisc_u_c.x;
    const float aviscj = pj.rho_avisc_u_c.y;
    const float energyj = pj.rho_avisc_u_c.z;
    const float cj = pj.rho_avisc_u_c.w;

    /* const float vsigj = pj.vsig_lapu_aviscmax.x; */
    /* const float lapuj = pj.vsig_lapu_aviscmax.y; */
    /* const float avisc_maxj = pj.vsig_lapu_aviscmax.z; */

    /* Now get stuff done. */
    const float xij = xi - xj;
    const float yij = yi - yj;
    const float zij = zi - zj;

    const float r2 = xij * xij + yij * yij + zij * zij;

    if ((r2 < hig2) && (j != pid)) {
      /* (j != pid): Exclude self contribution. This happens at a later step. */

      const float r = sqrtf(r2);
      const float r_inv = 1.f / r;

      /* Cosmology terms for the signal velocity */
      const float fac_mu = d_pow_three_gamma_minus_five_over_two(d_a);
      const float a2_Hubble = d_a * d_a * d_H;

      /* Compute dv dot r */
      float dvx = vxi - vxj;
      float dvy = vyi - vyj;
      float dvz = vzi - vzj;
      const float dvdr = dvx * xij + dvy * yij + dvz * zij;

      /* Add Hubble flow */
      const float dvdr_Hubble = dvdr + a2_Hubble * r2;

      /* Are the particles moving towards each others ? */
      const float omega_ij = fminf(dvdr_Hubble, 0.f);
      const float mu_ij = fac_mu * r_inv * omega_ij; /* This is 0 or negative */

      /* Signal velocity */
      const float new_v_sig = ci + cj - const_viscosity_beta * mu_ij;

      /* Update if we need to */
      res_vsig_lapu_avisci.x = fmaxf(vsigi, new_v_sig);

      /* Calculate Del^2 u for the thermal diffusion coefficient. */
      /* Need to get some kernel values F_ij = wi_dx */
      float wi;
      float wi_dx;
      const float ui = r * hi_inv;
      d_kernel_deval(ui, &wi, &wi_dx);

      const float delta_u_factor = (energyi - energyj) * r_inv;
      res_vsig_lapu_avisci.y += mj * delta_u_factor * wi_dx / rhoj;

      /* Set the maximal alpha from the previous step over the neighbours
       * (this is used to limit the diffusion in hydro_prepare_force) */
      res_vsig_lapu_avisci.z = fmaxf(res_vsig_lapu_avisci.z, aviscj);
    }
  } /*Loop through parts in cell j one GPU_THREAD_BLOCK_SIZE at a time*/

  /* Write results. */
  d_parts_recv[pid].vsig_lapu_aviscmax = res_vsig_lapu_avisci;
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
__device__ __attribute__((always_inline)) INLINE void cuda_kernel_force(
    int pid, const struct gpu_part_send_f *__restrict__ d_parts_send,
    struct gpu_part_recv_f *__restrict__ d_parts_recv, float d_a, float d_H) {

  /* First, grab handles */
  const struct gpu_part_send_f pi = d_parts_send[pid];

  const float xi = pi.x_h.x;
  const float yi = pi.x_h.y;
  const float zi = pi.x_h.z;
  const float hi = pi.x_h.w;

  const float vxi = pi.vx_m.x;
  const float vyi = pi.vx_m.y;
  const float vzi = pi.vx_m.z;
  const float mi = pi.vx_m.w;

  const float fi = pi.f_bals_rho_p.x;
  const float balsi = pi.f_bals_rho_p.y;
  const float rhoi = pi.f_bals_rho_p.z;
  const float pressurei = pi.f_bals_rho_p.w;

  const float ci = pi.c_u_avisc_adiff.x;
  const float energyi = pi.c_u_avisc_adiff.y;
  const float avisci = pi.c_u_avisc_adiff.z;
  const float adiffi = pi.c_u_avisc_adiff.w;

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
    const struct gpu_part_send_f pj = d_parts_send[j];

    const float xj = pj.x_h.x;
    const float yj = pj.x_h.y;
    const float zj = pj.x_h.z;
    const float hj = pj.x_h.w;

    const float vxj = pj.vx_m.x;
    const float vyj = pj.vx_m.y;
    const float vzj = pj.vx_m.z;
    const float mj = pj.vx_m.w;

    const float fj = pj.f_bals_rho_p.x;
    const float balsj = pj.f_bals_rho_p.y;
    const float rhoj = pj.f_bals_rho_p.z;
    const float pressurej = pj.f_bals_rho_p.w;

    const float cj = pj.c_u_avisc_adiff.x;
    const float energyj = pj.c_u_avisc_adiff.y;
    const float aviscj = pj.c_u_avisc_adiff.z;
    const float adiffj = pj.c_u_avisc_adiff.w;

    const int tbj = pi.timebin_minngbtimebin_pjs_pje.x;
    /* const int min_ngb_tbj = pi.timebin_minngbtimebin_pjs_pje.y; */

    /* Now get stuff done. */
    const float xij = xi - xj;
    const float yij = yi - yj;
    const float zij = zi - zj;
    const float r2 = xij * xij + yij * yij + zij * zij;
    const float hjg2 = hj * hj * kernel_gamma2;

    if (((r2 < hig2) || (r2 < hjg2)) && (j != pid)) {
      /* (j != pid): Exclude self contribution. This happens at a later step. */

      /* Cosmology terms for the signal velocity */
      const float fac_mu = d_pow_three_gamma_minus_five_over_two(d_a);
      const float a2_Hubble = d_a * d_a * d_H;

      const float r = sqrt(r2);
      const float r_inv = 1.f / r;

      /* Get the kernel for hi. */
      const float xi = r * hi_inv;
      float wi;
      float wi_dx;
      d_kernel_deval(xi, &wi, &wi_dx);
      const float wi_dr = hid_inv * wi_dx;

      /* Get the kernel for hj. */
      const float hj_inv = 1.0f / hj;
      const float hjd_inv = d_pow_dimension_plus_one(hj_inv); /* 1/h^(d+1) */
      const float xj = r * hj_inv;
      float wj;
      float wj_dx;
      d_kernel_deval(xj, &wj, &wj_dx);
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
      const float visc = -0.25f * alpha * v_sig * mu_ij * (balsi + balsj) * rhoij_inv;

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
      res_ahydro.x -= mj * acc * xij;
      res_ahydro.y -= mj * acc * yij;
      res_ahydro.z -= mj * acc * zij;

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
          (pressurei * adiffi + pressurej * adiffj) /
          (pressurei + pressurej);
      /* if (fabsf(pressurei + pressurej) < 1e-10) alpha_diff = 0.f; */

      const float v_diff = alpha_diff * 0.5f *
                           (sqrtf(2.f * fabsf(pressurei - pressurej) * rhoij_inv) +
                            fabsf(fac_mu * r_inv * dvdr_Hubble));

      /* wi_dx + wj_dx / 2 is F_ij */
      const float diff_du_term = v_diff * (energyi - energyj) *
          (f_ij * wi_dr * rhoi_inv + f_ji * wj_dr * rhoj_inv);

      /* Assemble the energy equation term */
      const float du_dt_i = sph_du_term_i + visc_du_term + diff_du_term;

      /* Internal energy time derivative */
      res_udt_hdt.x += du_dt_i * mj;

      /* Get the time derivative for h. */
      res_udt_hdt.y -= mj * dvdr * r_inv * rhoj_inv * wi_dr;

      if (tbj > 0) res_min_ngb_timebin = min(res_min_ngb_timebin, tbj);
    }
  } /*Loop through parts in cell j one GPU_THREAD_BLOCK_SIZE at a time*/

  d_parts_recv[pid].udt_hdt_minngbtb = {res_udt_hdt.x, res_udt_hdt.y, (float) res_min_ngb_timebin};
  d_parts_recv[pid].a_hydro = res_ahydro;
}

#ifdef __cplusplus
}
#endif

#endif /* CUDA_PARTICLE_KERNELS_CUH */
