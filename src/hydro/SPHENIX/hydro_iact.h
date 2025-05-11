/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2019 Josh Borrow (joshua.borrow@durham.ac.uk) &
 *                    Matthieu Schaller (schaller@strw.leidenuniv.nl)
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
#ifndef SWIFT_SPHENIX_HYDRO_IACT_H
#define SWIFT_SPHENIX_HYDRO_IACT_H

/**
 * @file SPHENIX/hydro_iact.h
 * @brief Density-Energy conservative implementation of SPH,
 *        with added SPHENIX physics (Borrow 2020) (interaction routines)
 */

#include "adaptive_softening_iact.h"
#include "adiabatic_index.h"
#include "fvpm_geometry.h"
#include "hydro_parameters.h"
#include "hydro_part.h"
#include "minmax.h"
#include "signal_velocity.h"

/**
 * @brief Density interaction between two particles.
 *
 * @param r2 Comoving square distance between the two particles.
 * @param dx Comoving vector separating both particles (pi - pj).
 * @param hi Comoving smoothing-length of part*icle i.
 * @param hj Comoving smoothing-length of part*icle j.
 * @param pi First part*icle.
 * @param pj Second part*icle.
 * @param a Current scale factor.
 * @param H Current Hubble parameter.
 */
__attribute__((always_inline)) INLINE static void runner_iact_density(
    const float r2, const float dx[3], const float hi, const float hj,
    struct part* restrict pi, struct part* restrict pj, const float a,
    const float H) {

  float wi, wj, wi_dx, wj_dx;
  float dv[3], curlvr[3];

  const float r = sqrtf(r2);

  /* Get the masses. */
  const float mi = part_get_mass(pi);
  const float mj = part_get_mass(pj);

  /* Compute density of pi. */
  const float hi_inv = 1.f / hi;
  const float ui = r * hi_inv;

  kernel_deval(ui, &wi, &wi_dx);

  const float rho_i = part_get_rho(pi);
  part_set_rho(pi, rho_i + mj * wi);

  const float rho_dh_i = part_get_rho_dh(pi);
  part_set_rho_dh(pi, rho_dh_i - mj * (hydro_dimension * wi + ui * wi_dx));

  const float wcount_i = part_get_wcount(pi);
  part_set_wcount(pi, wcount_i + wi);

  const float wcount_dh_i = part_get_wcount_dh(pi);
  part_set_wcount_dh(pi, wcount_dh_i - (hydro_dimension * wi + ui * wi_dx));

  adaptive_softening_add_correction_term(pi, ui, hi_inv, mj);

  /* Collect data for FVPM matrix construction */
  fvpm_accumulate_geometry_and_matrix(pi, wi, dx);
  fvpm_update_centroid_left(pi, dx, wi);

  /* Compute density of pj. */
  const float hj_inv = 1.f / hj;
  const float uj = r * hj_inv;
  kernel_deval(uj, &wj, &wj_dx);

  const float rho_j = part_get_rho(pj);
  part_set_rho(pj, rho_j + mi * wj);

  const float rho_dh_j = part_get_rho_dh(pj);
  part_set_rho_dh(pj, rho_dh_j - mi * (hydro_dimension * wj + uj * wj_dx));

  const float wcount_j = part_get_wcount(pj);
  part_set_wcount(pj, wcount_j + wj);

  const float wcount_dh_j = part_get_wcount_dh(pj);
  part_set_wcount_dh(pj, wcount_dh_j - (hydro_dimension * wj + uj * wj_dx));

  adaptive_softening_add_correction_term(pj, uj, hj_inv, mi);

  /* Collect data for FVPM matrix construction */
  fvpm_accumulate_geometry_and_matrix(pj, wj, dx);
  fvpm_update_centroid_right(pj, dx, wj);

  /* Now we need to compute the div terms */
  const float r_inv = r ? 1.0f / r : 0.0f;
  const float faci = mj * wi_dx * r_inv;
  const float facj = mi * wj_dx * r_inv;

  /* Compute dv dot r */
  dv[0] = part_get_v_ind(pi, 0) - part_get_v_ind(pj, 0);
  dv[1] = part_get_v_ind(pi, 1) - part_get_v_ind(pj, 1);
  dv[2] = part_get_v_ind(pi, 2) - part_get_v_ind(pj, 2);
  const float dvdr = dv[0] * dx[0] + dv[1] * dx[1] + dv[2] * dx[2];

  part_set_div_v(pi, part_get_div_v(pi) - faci * dvdr);
  part_set_div_v(pj, part_get_div_v(pj) - facj * dvdr);

  /* Compute dv cross r */
  curlvr[0] = dv[1] * dx[2] - dv[2] * dx[1];
  curlvr[1] = dv[2] * dx[0] - dv[0] * dx[2];
  curlvr[2] = dv[0] * dx[1] - dv[1] * dx[0];

  float* rot_v_i = part_get_rot_v(pi);
  part_set_rot_v_ind(pi, 0, rot_v_i[0] + faci * curlvr[0]);
  part_set_rot_v_ind(pi, 1, rot_v_i[0] + faci * curlvr[1]);
  part_set_rot_v_ind(pi, 2, rot_v_i[0] + faci * curlvr[2]);

  /* Negative because of the change in sign of dx & dv. */
  float* rot_v_j = part_get_rot_v(pi);
  part_set_rot_v_ind(pj, 0, rot_v_j[0] + facj * curlvr[0]);
  part_set_rot_v_ind(pj, 1, rot_v_j[0] + facj * curlvr[1]);
  part_set_rot_v_ind(pj, 2, rot_v_j[0] + facj * curlvr[2]);

#ifdef SWIFT_HYDRO_DENSITY_CHECKS
  pi->n_density += wi;
  pj->n_density += wj;
  pi->N_density++;
  pj->N_density++;
#endif
}

/**
 * @brief Density interaction between two particles (non-symmetric).
 *
 * @param r2 Comoving square distance between the two particles.
 * @param dx Comoving vector separating both particles (pi - pj).
 * @param hi Comoving smoothing-length of part*icle i.
 * @param hj Comoving smoothing-length of part*icle j.
 * @param pi First part*icle.
 * @param pj Second part*icle (not updated).
 * @param a Current scale factor.
 * @param H Current Hubble parameter.
 */
__attribute__((always_inline)) INLINE static void runner_iact_nonsym_density(
    const float r2, const float dx[3], const float hi, const float hj,
    struct part* restrict pi, const struct part* restrict pj, const float a,
    const float H) {

  float wi, wi_dx;
  float dv[3], curlvr[3];

  /* Get the masses. */
  const float mj = part_get_mass(pj);

  /* Get r and r inverse. */
  const float r = sqrtf(r2);

  const float h_inv = 1.f / hi;
  const float ui = r * h_inv;

  kernel_deval(ui, &wi, &wi_dx);

  const float rho_i = part_get_rho(pi);
  part_set_rho(pi, rho_i + mj * wi);

  const float rho_dh_i = part_get_rho_dh(pi);
  part_set_rho_dh(pi, rho_dh_i - mj * (hydro_dimension * wi + ui * wi_dx));

  const float wcount_i = part_get_wcount(pi);
  part_set_wcount(pi, wcount_i + wi);

  const float wcount_dh_i = part_get_wcount_dh(pi);
  part_set_wcount_dh(pi, wcount_dh_i - (hydro_dimension * wi + ui * wi_dx));

  adaptive_softening_add_correction_term(pi, ui, h_inv, mj);

  /* Collect data for FVPM matrix construction */
  fvpm_accumulate_geometry_and_matrix(pi, wi, dx);
  fvpm_update_centroid_left(pi, dx, wi);

  const float r_inv = r ? 1.0f / r : 0.0f;
  const float faci = mj * wi_dx * r_inv;

  /* Compute dv dot r */
  dv[0] = part_get_v_ind(pi, 0) - part_get_v_ind(pj, 0);
  dv[1] = part_get_v_ind(pi, 1) - part_get_v_ind(pj, 1);
  dv[2] = part_get_v_ind(pi, 2) - part_get_v_ind(pj, 2);
  const float dvdr = dv[0] * dx[0] + dv[1] * dx[1] + dv[2] * dx[2];

  part_set_div_v(pi, part_get_div_v(pi) - faci * dvdr);

  /* Compute dv cross r */
  curlvr[0] = dv[1] * dx[2] - dv[2] * dx[1];
  curlvr[1] = dv[2] * dx[0] - dv[0] * dx[2];
  curlvr[2] = dv[0] * dx[1] - dv[1] * dx[0];

  float* rot_v_i = part_get_rot_v(pi);
  part_set_rot_v_ind(pi, 0, rot_v_i[0] + faci * curlvr[0]);
  part_set_rot_v_ind(pi, 1, rot_v_i[0] + faci * curlvr[1]);
  part_set_rot_v_ind(pi, 2, rot_v_i[0] + faci * curlvr[2]);

#ifdef SWIFT_HYDRO_DENSITY_CHECKS
  pi->n_density += wi;
  pi->N_density++;
#endif
}

/**
 * @brief Calculate the gradient interaction between particle i and particle j
 *
 * This method wraps around hydro_gradients_collect, which can be an empty
 * method, in which case no gradients are used.
 *
 * @param r2 Comoving squared distance between particle i and particle j.
 * @param dx Comoving distance vector between the particles (dx = pi->x -
 * pj->x).
 * @param hi Comoving smoothing-length of particle i.
 * @param hj Comoving smoothing-length of particle j.
 * @param pi Particle i.
 * @param pj Particle j.
 * @param a Current scale factor.
 * @param H Current Hubble parameter.
 */
__attribute__((always_inline)) INLINE static void runner_iact_gradient(
    const float r2, const float dx[3], const float hi, const float hj,
    struct part* restrict pi, struct part* restrict pj, const float a,
    const float H) {

  /* We need to construct the maximal signal velocity between our particle
   * and all of it's neighbours */

  const float r = sqrtf(r2);
  const float r_inv = r ? 1.0f / r : 0.0f;

  /* Cosmology terms for the signal velocity */
  const float fac_mu = pow_three_gamma_minus_five_over_two(a);
  const float a2_Hubble = a * a * H;

  const float dvdr = (part_get_v_ind(pi, 0) - part_get_v_ind(pj, 0)) * dx[0] +
                     (part_get_v_ind(pi, 1) - part_get_v_ind(pj, 1)) * dx[1] +
                     (part_get_v_ind(pi, 2) - part_get_v_ind(pj, 2)) * dx[2];

  /* Add Hubble flow */

  const float dvdr_Hubble = dvdr + a2_Hubble * r2;
  /* Are the particles moving towards each others ? */
  const float omega_ij = min(dvdr_Hubble, 0.f);
  const float mu_ij = fac_mu * r_inv * omega_ij; /* This is 0 or negative */

  /* Signal velocity */
  const float new_v_sig =
      signal_velocity(dx, pi, pj, mu_ij, const_viscosity_beta);

  /* Update if we need to */
  part_set_v_sig(pi, max(part_get_v_sig(pi), new_v_sig));
  part_set_v_sig(pj, max(part_get_v_sig(pj), new_v_sig));

  /* Calculate Del^2 u for the thermal diffusion coefficient. */
  /* Need to get some kernel values F_ij = wi_dx */
  float wi, wi_dx, wj, wj_dx;

  const float ui = r / hi;
  const float uj = r / hj;

  kernel_deval(ui, &wi, &wi_dx);
  kernel_deval(uj, &wj, &wj_dx);

  const float delta_u_factor = (part_get_u(pi) - part_get_u(pj)) * r_inv;

  const float laplace_i = part_get_laplace_u(pi);
  const float m_j = part_get_mass(pj);
  const float rho_j = part_get_rho(pj);
  part_set_laplace_u(pi, laplace_i + m_j * delta_u_factor * wi_dx / rho_j);

  const float laplace_j = part_get_laplace_u(pj);
  const float m_i = part_get_mass(pi);
  const float rho_i = part_get_rho(pi);
  part_set_laplace_u(pj, laplace_j - m_i * delta_u_factor * wj_dx / rho_i);

  /* Set the maximal alpha from the previous step over the neighbours
   * (this is used to limit the diffusion in hydro_prepare_force) */
  const float alpha_i = part_get_alpha_av(pi);
  const float alpha_j = part_get_alpha_av(pj);
  part_set_alpha_visc_max_ngb(pi,
                              max(part_get_alpha_visc_max_ngb(pi), alpha_j));
  part_set_alpha_visc_max_ngb(pj,
                              max(part_get_alpha_visc_max_ngb(pi), alpha_i));

#ifdef SWIFT_HYDRO_DENSITY_CHECKS
  pi->n_gradient += wi;
  pj->n_gradient += wj;
  pi->N_gradient++;
  pj->N_gradient++;
#endif
}

/**
 * @brief Calculate the gradient interaction between particle i and particle j:
 * non-symmetric version
 *
 * This method wraps around hydro_gradients_nonsym_collect, which can be an
 * empty method, in which case no gradients are used.
 *
 * @param r2 Comoving squared distance between particle i and particle j.
 * @param dx Comoving distance vector between the particles (dx = pi->x -
 * pj->x).
 * @param hi Comoving smoothing-length of particle i.
 * @param hj Comoving smoothing-length of particle j.
 * @param pi Particle i.
 * @param pj Particle j.
 * @param a Current scale factor.
 * @param H Current Hubble parameter.
 */
__attribute__((always_inline)) INLINE static void runner_iact_nonsym_gradient(
    const float r2, const float dx[3], const float hi, const float hj,
    struct part* restrict pi, struct part* restrict pj, const float a,
    const float H) {

  /* We need to construct the maximal signal velocity between our particle
   * and all of it's neighbours */

  const float r = sqrtf(r2);
  const float r_inv = r ? 1.0f / r : 0.0f;

  /* Cosmology terms for the signal velocity */
  const float fac_mu = pow_three_gamma_minus_five_over_two(a);
  const float a2_Hubble = a * a * H;

  const float dvdr = (part_get_v_ind(pi, 0) - part_get_v_ind(pj, 0)) * dx[0] +
                     (part_get_v_ind(pi, 1) - part_get_v_ind(pj, 1)) * dx[1] +
                     (part_get_v_ind(pi, 2) - part_get_v_ind(pj, 2)) * dx[2];

  /* Add Hubble flow */

  const float dvdr_Hubble = dvdr + a2_Hubble * r2;
  /* Are the particles moving towards each others ? */
  const float omega_ij = min(dvdr_Hubble, 0.f);
  const float mu_ij = fac_mu * r_inv * omega_ij; /* This is 0 or negative */

  /* Signal velocity */
  const float new_v_sig =
      signal_velocity(dx, pi, pj, mu_ij, const_viscosity_beta);

  /* Update if we need to */
  part_set_v_sig(pi, max(part_get_v_sig(pi), new_v_sig));

  /* Calculate Del^2 u for the thermal diffusion coefficient. */
  /* Need to get some kernel values F_ij = wi_dx */
  float wi, wi_dx;

  const float ui = r / hi;

  kernel_deval(ui, &wi, &wi_dx);

  const float delta_u_factor = (part_get_u(pi) - part_get_u(pj)) * r_inv;
  const float laplace_i = part_get_laplace_u(pi);
  const float m_j = part_get_mass(pj);
  const float rho_j = part_get_rho(pj);
  part_set_laplace_u(pi, laplace_i + m_j * delta_u_factor * wi_dx / rho_j);

  /* Set the maximal alpha from the previous step over the neighbours
   * (this is used to limit the diffusion in hydro_prepare_force) */
  const float alpha_j = part_get_alpha_av(pj);
  part_set_alpha_visc_max_ngb(pi,
                              max(part_get_alpha_visc_max_ngb(pi), alpha_j));

#ifdef SWIFT_HYDRO_DENSITY_CHECKS
  pi->n_gradient += wi;
  pi->N_gradient++;
#endif
}

/**
 * @brief Force interaction between two particles.
 *
 * @param r2 Comoving square distance between the two particles.
 * @param dx Comoving vector separating both particles (pi - pj).
 * @param hi Comoving smoothing-length of part*icle i.
 * @param hj Comoving smoothing-length of part*icle j.
 * @param pi First part*icle.
 * @param pj Second part*icle.
 * @param a Current scale factor.
 * @param H Current Hubble parameter.
 */
__attribute__((always_inline)) INLINE static void runner_iact_force(
    const float r2, const float dx[3], const float hi, const float hj,
    struct part* restrict pi, struct part* restrict pj, const float a,
    const float H) {

  /* Cosmological factors entering the EoMs */
  const float fac_mu = pow_three_gamma_minus_five_over_two(a);
  const float a2_Hubble = a * a * H;

  const float r = sqrtf(r2);
  const float r_inv = r ? 1.0f / r : 0.0f;

  /* Recover some data */
  const float mi = part_get_mass(pi);
  const float mj = part_get_mass(pj);

  const float rhoi = part_get_rho(pi);
  const float rhoj = part_get_rho(pj);

  const float pressurei = part_get_pressure(pi);
  const float pressurej = part_get_pressure(pj);

  /* Get the kernel for hi. */
  const float hi_inv = 1.0f / hi;
  const float hid_inv = pow_dimension_plus_one(hi_inv); /* 1/h^(d+1) */
  const float xi = r * hi_inv;
  float wi, wi_dx;
  kernel_deval(xi, &wi, &wi_dx);
  const float wi_dr = hid_inv * wi_dx;

  /* Get the kernel for hj. */
  const float hj_inv = 1.0f / hj;
  const float hjd_inv = pow_dimension_plus_one(hj_inv); /* 1/h^(d+1) */
  const float xj = r * hj_inv;
  float wj, wj_dx;
  kernel_deval(xj, &wj, &wj_dx);
  const float wj_dr = hjd_inv * wj_dx;

  /* Compute dv dot r. */
  const float dvdr = (part_get_v_ind(pi, 0) - part_get_v_ind(pj, 0)) * dx[0] +
                     (part_get_v_ind(pi, 1) - part_get_v_ind(pj, 1)) * dx[1] +
                     (part_get_v_ind(pi, 2) - part_get_v_ind(pj, 2)) * dx[2];

  /* Includes the hubble flow term; not used for du/dt */
  const float dvdr_Hubble = dvdr + a2_Hubble * r2;

  /* Are the particles moving towards each others ? */
  const float omega_ij = min(dvdr_Hubble, 0.f);
  const float mu_ij = fac_mu * r_inv * omega_ij; /* This is 0 or negative */

  /* Compute sound speeds and signal velocity */
  const float v_sig = signal_velocity(dx, pi, pj, mu_ij, const_viscosity_beta);

  /* Variable smoothing length term */
  const float f_ij = 1.f - part_get_f_gradh(pi) / mj;
  const float f_ji = 1.f - part_get_f_gradh(pj) / mi;

  /* Balsara term */
  const float balsara_i = part_get_balsara(pi);
  const float balsara_j = part_get_balsara(pj);

  /* Construct the full viscosity term */
  const float rho_ij = rhoi + rhoj;
  const float alpha = part_get_alpha_av(pi) + part_get_alpha_av(pj);
  const float visc =
      -0.25f * alpha * v_sig * mu_ij * (balsara_i + balsara_j) / rho_ij;

  /* Convolve with the kernel */
  const float visc_acc_term =
      0.5f * visc * (wi_dr * f_ij + wj_dr * f_ji) * r_inv;

  /* Compute gradient terms */
  const float P_over_rho2_i = pressurei / (rhoi * rhoi) * f_ij;
  const float P_over_rho2_j = pressurej / (rhoj * rhoj) * f_ji;

  /* SPH acceleration term */
  const float sph_acc_term =
      (P_over_rho2_i * wi_dr + P_over_rho2_j * wj_dr) * r_inv;

  /* Adaptive softening acceleration term */
  const float adapt_soft_acc_term =
      adaptive_softening_get_acc_term(pi, pj, wi_dr, wj_dr, f_ij, f_ji, r_inv);

  /* Assemble the acceleration */
  const float acc = sph_acc_term + visc_acc_term + adapt_soft_acc_term;

  /* Use the force Luke ! */
  part_set_a_hydro_ind(pi, 0, part_get_a_hydro_ind(pi, 0) - mj * acc * dx[0]);
  part_set_a_hydro_ind(pi, 1, part_get_a_hydro_ind(pi, 1) - mj * acc * dx[1]);
  part_set_a_hydro_ind(pi, 2, part_get_a_hydro_ind(pi, 2) - mj * acc * dx[2]);

  part_set_a_hydro_ind(pj, 0, part_get_a_hydro_ind(pj, 0) + mi * acc * dx[0]);
  part_set_a_hydro_ind(pj, 1, part_get_a_hydro_ind(pj, 1) + mi * acc * dx[1]);
  part_set_a_hydro_ind(pj, 2, part_get_a_hydro_ind(pj, 2) + mi * acc * dx[2]);

  /* Get the time derivative for u. */
  const float sph_du_term_i = P_over_rho2_i * dvdr * r_inv * wi_dr;
  const float sph_du_term_j = P_over_rho2_j * dvdr * r_inv * wj_dr;

  /* Viscosity term */
  const float visc_du_term = 0.5f * visc_acc_term * dvdr_Hubble;

  /* Diffusion term */
  /* Combine the alpha_diff into a pressure-based switch -- this allows the
   * alpha from the highest pressure particle to dominate, so that the
   * diffusion limited particles always take precedence - another trick to
   * allow the scheme to work with thermal feedback. */
  const float alpha_diff = (pressurei * part_get_alpha_diff(pi) +
                            pressurej * part_get_alpha_diff(pj)) /
                           (pressurei + pressurej);
  const float v_diff = alpha_diff * 0.5f *
                       (sqrtf(2.f * fabsf(pressurei - pressurej) / rho_ij) +
                        fabsf(fac_mu * r_inv * dvdr_Hubble));
  /* wi_dx + wj_dx / 2 is F_ij */
  const float diff_du_term = v_diff * (part_get_u(pi) - part_get_u(pj)) *
                             (f_ij * wi_dr / rhoi + f_ji * wj_dr / rhoj);

  /* Assemble the energy equation term */
  const float du_dt_i = sph_du_term_i + visc_du_term + diff_du_term;
  const float du_dt_j = sph_du_term_j + visc_du_term - diff_du_term;

  /* Internal energy time derivative */
  part_set_u_dt(pi, part_get_u_dt(pi) + du_dt_i * mj);
  part_set_u_dt(pj, part_get_u_dt(pj) + du_dt_j * mi);

  /* Get the time derivative for h. */
  part_set_h_dt(pi, part_get_h_dt(pi) - mj * dvdr * r_inv / rhoj * wi_dr);
  part_set_h_dt(pj, part_get_h_dt(pj) - mi * dvdr * r_inv / rhoi * wj_dr);

#ifdef SWIFT_HYDRO_DENSITY_CHECKS
  pi->n_force += wi + wj;
  pj->n_force += wi + wj;
  pi->N_force++;
  pj->N_force++;
#endif
}

/**
 * @brief Force interaction between two particles (non-symmetric).
 *
 * @param r2 Comoving square distance between the two particles.
 * @param dx Comoving vector separating both particles (pi - pj).
 * @param hi Comoving smoothing-length of part*icle i.
 * @param hj Comoving smoothing-length of part*icle j.
 * @param pi First part*icle.
 * @param pj Second part*icle (not updated).
 * @param a Current scale factor.
 * @param H Current Hubble parameter.
 */
__attribute__((always_inline)) INLINE static void runner_iact_nonsym_force(
    const float r2, const float dx[3], const float hi, const float hj,
    struct part* restrict pi, const struct part* restrict pj, const float a,
    const float H) {

  /* Cosmological factors entering the EoMs */
  const float fac_mu = pow_three_gamma_minus_five_over_two(a);
  const float a2_Hubble = a * a * H;

  const float r = sqrtf(r2);
  const float r_inv = r ? 1.0f / r : 0.0f;

  /* Recover some data */
  const float mi = part_get_mass(pi);
  const float mj = part_get_mass(pj);

  const float rhoi = part_get_rho(pi);
  const float rhoj = part_get_rho(pj);

  const float pressurei = part_get_pressure(pi);
  const float pressurej = part_get_pressure(pj);

  /* Get the kernel for hi. */
  const float hi_inv = 1.0f / hi;
  const float hid_inv = pow_dimension_plus_one(hi_inv); /* 1/h^(d+1) */
  const float xi = r * hi_inv;
  float wi, wi_dx;
  kernel_deval(xi, &wi, &wi_dx);
  const float wi_dr = hid_inv * wi_dx;

  /* Get the kernel for hj. */
  const float hj_inv = 1.0f / hj;
  const float hjd_inv = pow_dimension_plus_one(hj_inv); /* 1/h^(d+1) */
  const float xj = r * hj_inv;
  float wj, wj_dx;
  kernel_deval(xj, &wj, &wj_dx);
  const float wj_dr = hjd_inv * wj_dx;

  /* Compute dv dot r. */
  const float dvdr = (part_get_v_ind(pi, 0) - part_get_v_ind(pj, 0)) * dx[0] +
                     (part_get_v_ind(pi, 1) - part_get_v_ind(pj, 1)) * dx[1] +
                     (part_get_v_ind(pi, 2) - part_get_v_ind(pj, 2)) * dx[2];

  /* Includes the hubble flow term; not used for du/dt */
  const float dvdr_Hubble = dvdr + a2_Hubble * r2;

  /* Are the particles moving towards each others ? */
  const float omega_ij = min(dvdr_Hubble, 0.f);
  const float mu_ij = fac_mu * r_inv * omega_ij; /* This is 0 or negative */

  /* Compute sound speeds and signal velocity */
  const float v_sig = signal_velocity(dx, pi, pj, mu_ij, const_viscosity_beta);

  /* Variable smoothing length term */
  const float f_ij = 1.f - part_get_f_gradh(pi) / mj;
  const float f_ji = 1.f - part_get_f_gradh(pj) / mi;

  /* Balsara term */
  const float balsara_i = part_get_balsara(pi);
  const float balsara_j = part_get_balsara(pj);

  /* Construct the full viscosity term */
  const float rho_ij = rhoi + rhoj;
  const float alpha = part_get_alpha_av(pi) + part_get_alpha_av(pj);
  const float visc =
      -0.25f * alpha * v_sig * mu_ij * (balsara_i + balsara_j) / rho_ij;

  /* Convolve with the kernel */
  const float visc_acc_term =
      0.5f * visc * (wi_dr * f_ij + wj_dr * f_ji) * r_inv;

  /* Compute gradient terms */
  const float P_over_rho2_i = pressurei / (rhoi * rhoi) * f_ij;
  const float P_over_rho2_j = pressurej / (rhoj * rhoj) * f_ji;

  /* SPH acceleration term */
  const float sph_acc_term =
      (P_over_rho2_i * wi_dr + P_over_rho2_j * wj_dr) * r_inv;

  /* Adaptive softening acceleration term */
  const float adapt_soft_acc_term =
      adaptive_softening_get_acc_term(pi, pj, wi_dr, wj_dr, f_ij, f_ji, r_inv);

  /* Assemble the acceleration */
  const float acc = sph_acc_term + visc_acc_term + adapt_soft_acc_term;

  /* Use the force Luke ! */
  part_set_a_hydro_ind(pi, 0, part_get_a_hydro_ind(pi, 0) - mj * acc * dx[0]);
  part_set_a_hydro_ind(pi, 1, part_get_a_hydro_ind(pi, 1) - mj * acc * dx[1]);
  part_set_a_hydro_ind(pi, 2, part_get_a_hydro_ind(pi, 2) - mj * acc * dx[2]);

  /* Get the time derivative for u. */
  const float sph_du_term_i = P_over_rho2_i * dvdr * r_inv * wi_dr;

  /* Viscosity term */
  const float visc_du_term = 0.5f * visc_acc_term * dvdr_Hubble;

  /* Diffusion term */
  /* Combine the alpha_diff into a pressure-based switch -- this allows the
   * alpha from the highest pressure particle to dominate, so that the
   * diffusion limited particles always take precedence - another trick to
   * allow the scheme to work with thermal feedback. */
  const float alpha_diff = (pressurei * part_get_alpha_diff(pi) +
                            pressurej * part_get_alpha_diff(pj)) /
                           (pressurei + pressurej);
  const float v_diff = alpha_diff * 0.5f *
                       (sqrtf(2.f * fabsf(pressurei - pressurej) / rho_ij) +
                        fabsf(fac_mu * r_inv * dvdr_Hubble));
  /* wi_dx + wj_dx / 2 is F_ij */
  const float diff_du_term = v_diff * (part_get_u(pi) - part_get_u(pj)) *
                             (f_ij * wi_dr / rhoi + f_ji * wj_dr / rhoj);

  /* Assemble the energy equation term */
  const float du_dt_i = sph_du_term_i + visc_du_term + diff_du_term;

  /* Internal energy time derivative */
  part_set_u_dt(pi, part_get_u_dt(pi) + du_dt_i * mj);

  /* Get the time derivative for h. */
  part_set_h_dt(pi, part_get_h_dt(pi) - mj * dvdr * r_inv / rhoj * wi_dr);

#ifdef SWIFT_HYDRO_DENSITY_CHECKS
  pi->n_force += wi + wj;
  pi->N_force++;
#endif
}

#endif /* SWIFT_SPHENIX_HYDRO_IACT_H */
