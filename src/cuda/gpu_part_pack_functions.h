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
#ifndef GPU_PART_PACK_FUNCTIONS_H
#define GPU_PART_PACK_FUNCTIONS_H

/**
 * @file cuda/gpu_part_pack_functions.h
 * @brief Functions related to packing and unpacking particles to/from a cell.
 * This needs to be specific to any hydro scheme/SPH flavour.
 */

#include "active.h"
#include "cell.h"
#include "gpu_offload_data.h"

/**
 * @brief Unpacks the density data from GPU buffers into cell's particle arrays
 *
 * @param c the #cell
 * @param parts_buffer the "receive" buffer to copy from
 * @param unpack_ind the index in parts_buffer to start copying from
 * @param count number of particles to unpack
 * @param e the #engine
 */
__attribute__((always_inline)) INLINE static void gpu_unpack_part_density(
    struct cell *restrict c,
    const struct gpu_part_recv_d *restrict parts_buffer, const int unpack_ind,
    const int count, const struct engine *e) {

  const struct gpu_part_recv_d *parts_recv = &parts_buffer[unpack_ind];

  for (int i = 0; i < count; i++) {

    struct part *p = &c->hydro.parts[i];
    if (!part_is_active(p, e)) continue;

    struct gpu_part_recv_d pr = parts_recv[i];

    float rho = part_get_rho(p) + pr.rho_rhodh_wcount_wcount_dh.x;
    part_set_rho(p, rho);

    float rho_dh = part_get_rho_dh(p) + pr.rho_rhodh_wcount_wcount_dh.y;
    part_set_rho_dh(p, rho_dh);

    float wcount = part_get_wcount(p) + pr.rho_rhodh_wcount_wcount_dh.z;
    part_set_wcount(p, wcount);

    float wcount_dh = part_get_wcount_dh(p) + pr.rho_rhodh_wcount_wcount_dh.w;
    part_set_wcount_dh(p, wcount_dh);

    float *rot_v = part_get_rot_v(p);
    rot_v[0] += pr.rot_vx_div_v.x;
    rot_v[1] += pr.rot_vx_div_v.y;
    rot_v[2] += pr.rot_vx_div_v.z;

    float div_v = part_get_div_v(p) + pr.rot_vx_div_v.w;
    part_set_div_v(p, div_v);
  }
}

/**
 * @brief Unpacks the density data from GPU buffers into cell's particle arrays
 *
 * @param c the #cell
 * @param parts_buffer the "receive" buffer to copy from
 * @param unpack_ind the index in parts_buffer to start copying from
 * @param count number of particles to unpack
 * @param e the #engine
 */
__attribute__((always_inline)) INLINE static void gpu_unpack_part_gradient(
    struct cell *restrict c,
    const struct gpu_part_recv_g *restrict parts_buffer, const int unpack_ind,
    const int count, const struct engine *e) {

  const struct gpu_part_recv_g *parts_recv = &parts_buffer[unpack_ind];

  for (int i = 0; i < count; i++) {

    struct part *p = &c->hydro.parts[i];
    if (!part_is_active(p, e)) continue;

    struct gpu_part_recv_g pr = parts_recv[i];

    float vsig = fmaxf(pr.vsig_lapu_aviscmax.x, part_get_v_sig(p));
    part_set_v_sig(p, vsig);

    float lu = pr.vsig_lapu_aviscmax.y + part_get_laplace_u(p);
    part_set_laplace_u(p, lu);

    float avisc_old = part_get_alpha_visc_max_ngb(p);
    float avisc = fmaxf(avisc_old, pr.vsig_lapu_aviscmax.z);
    part_set_alpha_visc_max_ngb(p, avisc);
  }
}

/**
 * @brief Unpacks the density data from GPU buffers into cell's particle arrays
 *
 * @param c the #cell
 * @param parts_buffer the "receive" buffer to copy from
 * @param unpack_ind the index in parts_buffer to start copying from
 * @param count number of particles to unpack
 * @param e the #engine
 */
__attribute__((always_inline)) INLINE static void gpu_unpack_part_force(
    struct cell *restrict c,
    const struct gpu_part_recv_f *restrict parts_buffer, const int unpack_ind,
    const int count, const struct engine *e) {

  const struct gpu_part_recv_f *parts_recv = &parts_buffer[unpack_ind];

  for (int i = 0; i < count; i++) {

    struct part *restrict p = &c->hydro.parts[i];
    if (!part_is_active(p, e)) continue;

    struct gpu_part_recv_f pr = parts_recv[i];

    float *a = part_get_a_hydro(p);
    a[0] += pr.a_hydro.x;
    a[1] += pr.a_hydro.y;
    a[2] += pr.a_hydro.z;

    float u_dt = pr.udt_hdt_minngbtb.x + part_get_u_dt(p);
    part_set_u_dt(p, u_dt);

    float h_dt = pr.udt_hdt_minngbtb.y + part_get_h_dt(p);
    part_set_h_dt(p, h_dt);

    timebin_t mintbin = (timebin_t)(pr.udt_hdt_minngbtb.z + 0.5f);
    part_set_timestep_limiter_min_ngb_time_bin(p, mintbin);
  }
}

/**
 * @brief Packs the cell particle data for pair density interactions into the
 * CPU-side buffers.
 *
 * @param c the #cell
 * @param parts_buffer the buffer to pack into
 * @param pack_ind the first free index in the buffer arrays to copy data into
 * @param shift periodic boundary shift
 * @param cjstart start index of cell cj's particles (which cell ci is to be
 * interacted with) in buffer
 * @param cjend end index of cell cj's particles (which cell ci is to be
 * interacted with) in buffer
 */
__attribute__((always_inline)) INLINE static void gpu_pack_part_density(
    const struct cell *restrict c,
    struct gpu_part_send_d *restrict parts_buffer, const int pack_ind,
    const double shift[3], const int cjstart, const int cjend) {

  /* Grab handles */
  const int count = c->hydro.count;
  const struct part *parts = c->hydro.parts;
  struct gpu_part_send_d *ps = &parts_buffer[pack_ind];

  for (int i = 0; i < count; i++) {

    const struct part *p = &parts[i];

    const double *x = part_get_const_x(p);
    ps[i].x_h.x = x[0] - shift[0];
    ps[i].x_h.y = x[1] - shift[1];
    ps[i].x_h.z = x[2] - shift[2];
    ps[i].x_h.w = part_get_h(p);

    const float *v = part_get_const_v(p);
    ps[i].vx_m.x = v[0];
    ps[i].vx_m.y = v[1];
    ps[i].vx_m.z = v[2];
    ps[i].vx_m.w = part_get_mass(p);

    ps[i].pjs_pje.x = cjstart;
    ps[i].pjs_pje.y = cjend;
  }
}

/**
 * @brief Packs the cell particle data for pair gradient interactions into the
 * CPU-side buffers.
 *
 * @param ci the #cell
 * @param parts_buffer the buffer to pack into
 * @param pack_ind the first free index in the buffer arrays to copy data into
 * @param shift periodic boundary shift
 * @param cjstart start index of cell cj's particles (which cell ci is to be
 * interacted with) in buffer
 * @param cjend end index of cell cj's particles (which cell ci is to be
 * interacted with) in buffer
 */
__attribute__((always_inline)) INLINE static void gpu_pack_part_gradient(
    const struct cell *restrict ci,
    struct gpu_part_send_g *restrict parts_buffer, const int pack_ind,
    const double shift[3], const int cjstart, const int cjend) {

  /* Grab handles */
  const int count = ci->hydro.count;
  const struct part *parts = ci->hydro.parts;
  struct gpu_part_send_g *ps = &parts_buffer[pack_ind];

  for (int i = 0; i < count; i++) {

    const struct part *p = &parts[i];

    const double *x = part_get_const_x(p);
    ps[i].x_h.x = x[0] - shift[0];
    ps[i].x_h.y = x[1] - shift[1];
    ps[i].x_h.z = x[2] - shift[2];
    ps[i].x_h.w = part_get_h(p);

    const float *v = part_get_const_v(p);
    ps[i].vx_m.x = v[0];
    ps[i].vx_m.y = v[1];
    ps[i].vx_m.z = v[2];
    ps[i].vx_m.w = part_get_mass(p);

    ps[i].rho_avisc_u_c.x = part_get_rho(p);
    ps[i].rho_avisc_u_c.y = part_get_alpha_av(p);
    ps[i].rho_avisc_u_c.z = part_get_u(p);
    ps[i].rho_avisc_u_c.w = part_get_soundspeed(p);

    ps[i].pjs_pje.x = cjstart;
    ps[i].pjs_pje.y = cjend;
  }
}

/**
 * @brief Packs the cell particle data for pair gradient interactions into the
 * CPU-side buffers.
 *
 * @param ci the #cell
 * @param parts_buffer the buffer to pack into
 * @param pack_ind the first free index in the buffer arrays to copy data into
 * @param shift periodic boundary shift
 * @param cjstart start index of cell cj's particles (which cell ci is to be
 * interacted with) in buffer
 * @param cjend end index of cell cj's particles (which cell ci is to be
 * interacted with) in buffer
 */
__attribute__((always_inline)) INLINE static void gpu_pack_part_force(
    const struct cell *restrict ci,
    struct gpu_part_send_f *restrict parts_buffer, const int pack_ind,
    const double shift[3], const int cjstart, const int cjend) {

  const int count = ci->hydro.count;
  const struct part *parts = ci->hydro.parts;
  struct gpu_part_send_f *ps = &parts_buffer[pack_ind];

  for (int i = 0; i < count; i++) {

    const struct part *p = &parts[i];

    const double *x = part_get_const_x(p);
    ps[i].x_h.x = x[0] - shift[0];
    ps[i].x_h.y = x[1] - shift[1];
    ps[i].x_h.z = x[2] - shift[2];
    ps[i].x_h.w = part_get_h(p);

    const float *v = part_get_const_v(p);
    ps[i].vx_m.x = v[0];
    ps[i].vx_m.y = v[1];
    ps[i].vx_m.z = v[2];
    ps[i].vx_m.w = part_get_mass(p);

    ps[i].f_bals_rho_p.x = part_get_f_gradh(p);
    ps[i].f_bals_rho_p.y = part_get_balsara(p);
    ps[i].f_bals_rho_p.z = part_get_rho(p);
    ps[i].f_bals_rho_p.w = part_get_pressure(p);

    ps[i].c_u_avisc_adiff.x = part_get_soundspeed(p);
    ps[i].c_u_avisc_adiff.y = part_get_u(p);
    ps[i].c_u_avisc_adiff.z = part_get_alpha_av(p);
    ps[i].c_u_avisc_adiff.w = part_get_alpha_diff(p);

    ps[i].timebin_minngbtimebin_pjs_pje.x = (int)part_get_time_bin(p);
    int mintbin = (int)part_get_timestep_limiter_min_ngb_time_bin(p);
    ps[i].timebin_minngbtimebin_pjs_pje.y = mintbin;
    ps[i].timebin_minngbtimebin_pjs_pje.z = cjstart;
    ps[i].timebin_minngbtimebin_pjs_pje.w = cjend;
  }
}

#endif /* GPU_PART_PACK_FUNCTIONS_H */
