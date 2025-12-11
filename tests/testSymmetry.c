/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (C) 2016 Matthieu Schaller (schaller@strw.leidenuniv.nl).
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
#include <config.h>

/* Local includes. */
#include "swift.h"
#include "timestep_limiter_iact.h"

/* System includes. */
#include <fenv.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_bytes(void *p, size_t len) {
  printf("(");
  for (size_t i = 0; i < len; ++i) {
    printf("%02x", ((unsigned char *)p)[i]);
    if (i % 4 == 3) printf("|");
  }
  printf(")\n");
}

void test(void) {

  /* Start with some values for the cosmological parameters */
  const float a = (float)random_uniform(0.8, 1.);
  const float H = 1.f;
  const float mu_0 = 4. * M_PI;
  const integertime_t ti_current = 1;
  const double time_base = 1e-5;
  const int with_cosmology = floor(random_uniform(0., 2.));
  struct cosmology cosmo;
  cosmology_init_no_cosmo(&cosmo);
  struct chemistry_global_data chemistry_data;

  /* Create two random particles (don't do this at home !) */
  struct part pi, pj;
  for (size_t i = 0; i < sizeof(struct part) / sizeof(float); ++i) {
    *(((float *)&pi) + i) = (float)random_uniform(0., 2.);
    *(((float *)&pj) + i) = (float)random_uniform(0., 2.);
  }

  /* Make the particle smoothing length and position reasonable */
  for (size_t i = 0; i < 3; ++i)
    part_set_x_ind(&pi, i, random_uniform(-1., 1.));
  for (size_t i = 0; i < 3; ++i)
    part_set_x_ind(&pj, i, random_uniform(-1., 1.));
  part_set_h(&pi, 2.f);
  part_set_h(&pj, 2.f);
  part_set_id(&pi, 1ll);
  part_set_id(&pj, 2ll);
  part_set_time_bin(&pi, 1);
  part_set_time_bin(&pj, 1);

#if defined(GIZMO_MFV_SPH)
  /* Give the primitive variables sensible values, since the Riemann solver does
     not like negative densities and pressures */
  part_set_rho(&pi, random_uniform(0.1f, 1.0f));
  part_set_v_ind(&pi, 0, random_uniform(-10.0f, 10.0f));
  part_set_v_ind(&pi, 1, random_uniform(-10.0f, 10.0f));
  part_set_v_ind(&pi, 2, random_uniform(-10.0f, 10.0f));
  part_set_pressure(&pi, random_uniform(0.1f, 1.0f));
  part_set_rho(&pj, random_uniform(0.1f, 1.0f));
  part_set_v_ind(&pj, 0, random_uniform(-10.0f, 10.0f));
  part_set_v_ind(&pj, 1, random_uniform(-10.0f, 10.0f));
  part_set_v_ind(&pj, 2, random_uniform(-10.0f, 10.0f));
  part_set_pressure(&pj, random_uniform(0.1f, 1.0f));
  /* make gradients zero */
  part_set_grad_rho_ind(&pi, 0, 0.f);
  part_set_grad_rho_ind(&pi, 1, 0.f);
  part_set_grad_rho_ind(&pi, 2, 0.f);
  part_set_grad_v_ind(&pi, 0, 0, 0.f);
  part_set_grad_v_ind(&pi, 1, 0, 0.f);
  part_set_grad_v_ind(&pi, 2, 0, 0.f);
  part_set_grad_v_ind(&pi, 0, 1, 0.f);
  part_set_grad_v_ind(&pi, 1, 1, 0.f);
  part_set_grad_v_ind(&pi, 2, 1, 0.f);
  part_set_grad_v_ind(&pi, 0, 2, 0.f);
  part_set_grad_v_ind(&pi, 1, 2, 0.f);
  part_set_grad_v_ind(&pi, 2, 2, 0.f);
  part_set_grad_pressure_ind(&pi, 0, 0.f);
  part_set_grad_pressure_ind(&pi, 1, 0.f);
  part_set_grad_pressure_ind(&pi, 2, 0.f);
  part_set_grad_rho_ind(&pj, 0, 0.f);
  part_set_grad_rho_ind(&pj, 1, 0.f);
  part_set_grad_rho_ind(&pj, 2, 0.f);
  part_set_grad_v_ind(&pj, 0, 0, 0.f);
  part_set_grad_v_ind(&pj, 1, 0, 0.f);
  part_set_grad_v_ind(&pj, 2, 0, 0.f);
  part_set_grad_v_ind(&pj, 0, 1, 0.f);
  part_set_grad_v_ind(&pj, 1, 1, 0.f);
  part_set_grad_v_ind(&pj, 2, 1, 0.f);
  part_set_grad_v_ind(&pj, 0, 2, 0.f);
  part_set_grad_v_ind(&pj, 1, 2, 0.f);
  part_set_grad_v_ind(&pj, 2, 2, 0.f);
  part_set_grad_pressure_ind(&pj, 0, 0.f);
  part_set_grad_pressure_ind(&pj, 1, 0.f);
  part_set_grad_pressure_ind(&pj, 2, 0.f);
#endif

  /* Make an xpart companion */
  struct xpart xpi, xpj;
  bzero(&xpi, sizeof(struct xpart));
  bzero(&xpj, sizeof(struct xpart));

  /* Make some copies */
  struct part pi2, pj2;
  memcpy(&pi2, &pi, sizeof(struct part));
  memcpy(&pj2, &pj, sizeof(struct part));

  int i_not_ok = memcmp(&pi, &pi2, sizeof(struct part));
  int j_not_ok = memcmp(&pj, &pj2, sizeof(struct part));

  if (i_not_ok) error("Particles 'pi' do not match after copy");
  if (j_not_ok) error("Particles 'pj' do not match after copy");

  /* Compute distance vector */
  float dx[3];
  dx[0] = part_get_x_ind(&pi, 0) - part_get_x_ind(&pj, 0);
  dx[1] = part_get_x_ind(&pi, 1) - part_get_x_ind(&pj, 1);
  dx[2] = part_get_x_ind(&pi, 2) - part_get_x_ind(&pj, 2);
  float r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
  const float hi = part_get_h(&pi);
  const float hj = part_get_h(&pj);
  const float hi2 = part_get_h(&pi2);
  const float hj2 = part_get_h(&pj2);

  /* --- Test the density loop --- */

  /* Call the symmetric version */
  runner_iact_density(r2, dx, hi, hj, &pi, &pj, a, H);
  runner_iact_mhd_density(r2, dx, hi, hj, &pi, &pj, mu_0, a, H);
  runner_iact_chemistry(r2, dx, hi, hj, &pi, &pj, a, H);
  runner_iact_pressure_floor(r2, dx, hi, hj, &pi, &pj, a, H);
  runner_iact_star_formation(r2, dx, hi, hj, &pi, &pj, a, H);
  runner_iact_sink(r2, dx, hi, hj, &pi, &pj, a, H);

  /* Call the non-symmetric version */
  runner_iact_nonsym_density(r2, dx, hi2, hj2, &pi2, &pj2, a, H);
  runner_iact_nonsym_mhd_density(r2, dx, hi2, hj2, &pi2, &pj2, mu_0, a, H);
  runner_iact_nonsym_chemistry(r2, dx, hi2, hj2, &pi2, &pj2, a, H);
  runner_iact_nonsym_pressure_floor(r2, dx, hi2, hj2, &pi2, &pj2, a, H);
  runner_iact_nonsym_star_formation(r2, dx, hi2, hj2, &pi2, &pj2, a, H);
  runner_iact_nonsym_sink(r2, dx, hi2, hj2, &pi2, &pj2, a, H);
  dx[0] = -dx[0];
  dx[1] = -dx[1];
  dx[2] = -dx[2];
  runner_iact_nonsym_density(r2, dx, hj2, hi2, &pj2, &pi2, a, H);
  runner_iact_nonsym_mhd_density(r2, dx, hj2, hi2, &pj2, &pi2, mu_0, a, H);
  runner_iact_nonsym_chemistry(r2, dx, hj2, hi2, &pj2, &pi2, a, H);
  runner_iact_nonsym_pressure_floor(r2, dx, hj2, hi2, &pj2, &pi2, a, H);
  runner_iact_nonsym_star_formation(r2, dx, hj2, hi2, &pj2, &pi2, a, H);
  runner_iact_nonsym_sink(r2, dx, hj2, hi2, &pj2, &pi2, a, H);

  /* Check that the particles are the same */
  i_not_ok = memcmp(&pi, &pi2, sizeof(struct part));
  j_not_ok = memcmp(&pj, &pj2, sizeof(struct part));

  if (i_not_ok) {
    printParticle_single(&pi, &xpi);
    printParticle_single(&pi2, &xpi);
    print_bytes(&pi, sizeof(struct part));
    print_bytes(&pi2, sizeof(struct part));
    error("Particles 'pi' do not match after density (byte = %d)", i_not_ok);
  }
  if (j_not_ok) {
    printParticle_single(&pj, &xpj);
    printParticle_single(&pj2, &xpj);
    print_bytes(&pj, sizeof(struct part));
    print_bytes(&pj2, sizeof(struct part));
    error("Particles 'pj' do not match after density (byte = %d)", j_not_ok);
  }

  /* --- Test the gradient loop --- */
#ifdef EXTRA_HYDRO_LOOP

  /* Call the symmetric version */
  runner_iact_gradient(r2, dx, hi, hj, &pi, &pj, a, H);
  runner_iact_mhd_gradient(r2, dx, hi, hj, &pi, &pj, mu_0, a, H);
  runner_iact_gradient_diffusion(r2, dx, hi, hj, &pi, &pj, a, H);

  /* Call the non-symmetric version */
  runner_iact_nonsym_gradient(r2, dx, hi2, hj2, &pi2, &pj2, a, H);
  runner_iact_nonsym_mhd_gradient(r2, dx, hi2, hj2, &pi2, &pj2, mu_0, a, H);
  runner_iact_nonsym_gradient_diffusion(r2, dx, hi, hj, &pi2, &pj2, a, H);
  dx[0] = -dx[0];
  dx[1] = -dx[1];
  dx[2] = -dx[2];
  runner_iact_nonsym_gradient(r2, dx, hj2, hi2, &pj2, &pi2, a, H);
  runner_iact_nonsym_mhd_gradient(r2, dx, hj2, hi2, &pj2, &pi2, mu_0, a, H);
  runner_iact_nonsym_gradient_diffusion(r2, dx, hj, hi, &pj2, &pi2, a, H);

  i_not_ok = memcmp((char *)&pi, (char *)&pi2, sizeof(struct part));
  j_not_ok = memcmp((char *)&pj, (char *)&pj2, sizeof(struct part));

  if (i_not_ok) {
    printParticle_single(&pi, &xpi);
    printParticle_single(&pi2, &xpi);
    print_bytes(&pi, sizeof(struct part));
    print_bytes(&pi2, sizeof(struct part));
    error("Particles 'pi' do not match after gradient (byte = %d)", i_not_ok);
  }
  if (j_not_ok) {
    printParticle_single(&pj, &xpj);
    printParticle_single(&pj2, &xpj);
    print_bytes(&pj, sizeof(struct part));
    print_bytes(&pj2, sizeof(struct part));
    error("Particles 'pj' do not match after gradient (byte = %d)", j_not_ok);
  }
#endif

  /* --- Test the force loop --- */

  /* Call the symmetric version */
  runner_iact_force(r2, dx, hi, hj, &pi, &pj, a, H);
  runner_iact_mhd_force(r2, dx, hi, hj, &pi, &pj, mu_0, a, H);
  runner_iact_diffusion(r2, dx, hi, hj, &pi, &pj, a, H, time_base,
                        ti_current, &cosmo, with_cosmology, &chemistry_data);
  runner_iact_timebin(r2, dx, hi, hj, &pi, &pj, a, H);
  runner_iact_rt_timebin(r2, dx, hi, hj, &pi, &pj, a, H);

  /* Call the non-symmetric version */
  runner_iact_nonsym_force(r2, dx, hi2, hj2, &pi2, &pj2, a, H);
  runner_iact_nonsym_mhd_force(r2, dx, hi2, hj2, &pi2, &pj2, mu_0, a, H);
  runner_iact_nonsym_diffusion(r2, dx, hi2, hj2, &pi2, &pj2, a, H,
                               time_base, ti_current, &cosmo, with_cosmology,
                               &chemistry_data);
  runner_iact_nonsym_timebin(r2, dx, hi2, hj2, &pi2, &pj2, a, H);
  runner_iact_nonsym_rt_timebin(r2, dx, hi2, hj2, &pi2, &pj2, a, H);
  dx[0] = -dx[0];
  dx[1] = -dx[1];
  dx[2] = -dx[2];
  runner_iact_nonsym_force(r2, dx, hj2, hi2, &pj2, &pi2, a, H);
  runner_iact_nonsym_mhd_force(r2, dx, hj2, hi2, &pj2, &pi2, mu_0, a, H);
  runner_iact_nonsym_diffusion(r2, dx, hj2, hi2, &pj2, &pi2, a, H,
                               time_base, ti_current, &cosmo, with_cosmology,
                               &chemistry_data);
  runner_iact_nonsym_timebin(r2, dx, hj2, hi2, &pj2, &pi2, a, H);
  runner_iact_nonsym_rt_timebin(r2, dx, hj2, hi2, &pj2, &pi2, a, H);

/* Check that the particles are the same */
#if defined(GIZMO_MFV_SPH)
  i_not_ok = 0;
  j_not_ok = 0;
  for (size_t i = 0; i < sizeof(struct part) / sizeof(float); ++i) {

    /* try this first to avoid dealing with NaNs and infinities */
    int check_i = memcmp((float *)&pi + i, (float *)&pi2 + i, sizeof(float));
    int check_j = memcmp((float *)&pj + i, (float *)&pj2 + i, sizeof(float));

    if (!check_i && !check_j) continue;

    if (check_i) {
      /* allow some wiggle room for roundoff errors */
      float aa = *(((float *)&pi) + i);
      float bb = *(((float *)&pi2) + i);

      int a_is_not_b;
      if ((aa + bb)) {
        a_is_not_b = (fabs((aa - bb) / (aa + bb)) > 1.e-4);
      } else {
        a_is_not_b = !(aa == 0.0f);
      }

      if (a_is_not_b) {
        message("%.8e, %.8e, %lu", aa, bb, i);
      }

      i_not_ok |= a_is_not_b;
    }

    if (check_j) {
      /* allow some wiggle room for roundoff errors */
      float cc = *(((float *)&pj) + i);
      float dd = *(((float *)&pj2) + i);
      int c_is_not_d;
      if ((cc + dd)) {
        c_is_not_d = (fabs((cc - dd) / (cc + dd)) > 1.e-4);
      } else {
        c_is_not_d = !(cc == 0.0f);
      }

      if (c_is_not_d) {
        message("%.8e, %.8e, %lu", cc, dd, i);
      }

      j_not_ok |= c_is_not_d;
    }
  }
#else
  i_not_ok = memcmp((char *)&pi, (char *)&pi2, sizeof(struct part));
  j_not_ok = memcmp((char *)&pj, (char *)&pj2, sizeof(struct part));
#endif

  if (i_not_ok) {
    printParticle_single(&pi, &xpi);
    printParticle_single(&pi2, &xpi);
    print_bytes(&pi, sizeof(struct part));
    print_bytes(&pi2, sizeof(struct part));
    error("Particles 'pi' do not match after force (byte = %d)", i_not_ok);
  }
  if (j_not_ok) {
    printParticle_single(&pj, &xpj);
    printParticle_single(&pj2, &xpj);
    print_bytes(&pj, sizeof(struct part));
    print_bytes(&pj2, sizeof(struct part));
    error("Particles 'pj' do not match after force (byte = %d)", j_not_ok);
  }
}

int main(int argc, char *argv[]) {

  /* Initialize CPU frequency, this also starts time. */
  unsigned long long cpufreq = 0;
  clocks_set_cpufreq(cpufreq);

/* Choke on FPEs */
#ifdef HAVE_FE_ENABLE_EXCEPT
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif

  /* Get some randomness going */
  const int seed = time(NULL);
  message("Seed = %d", seed);
  srand(seed);

  for (int i = 0; i < 100; ++i) {
    message("Random test %d/100", i);
    test();
  }
  message("All good");

  return 0;
}
