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
 * @brief packs particle data for density tasks into CPU-side buffers for self
 * tasks
 */
__attribute__((always_inline)) INLINE static void gpu_pack_part_self_density(
    const struct cell* restrict c, struct gpu_offload_data* restrict buf) {

  const int count = c->hydro.count;
  const size_t local_pack_position = buf->pv.count_parts;

  /* TODO: WHY ARE WE MEMCPYING HERE???? */
  struct part ptmps[count];
  memcpy(ptmps, (c->hydro.parts), count * sizeof(struct part));

  const float cellx = c->loc[0];
  const float celly = c->loc[1];
  const float cellz = c->loc[2];

  for (int i = 0; i < count; i++) {

    const int id_in_pack = i + local_pack_position;

    /* Data to be copied to GPU */
    const double *x = part_get_const_x(&ptmps[i]);
    buf->parts_send_d[id_in_pack].x_p_h.x = x[0] - cellx;
    buf->parts_send_d[id_in_pack].x_p_h.y = x[1] - celly;
    buf->parts_send_d[id_in_pack].x_p_h.z = x[2] - cellz;
    buf->parts_send_d[id_in_pack].x_p_h.w = part_get_h(&ptmps[i]);
    const float *v = part_get_const_v(&ptmps[i]);
    buf->parts_send_d[id_in_pack].ux_m.x = v[0];
    buf->parts_send_d[id_in_pack].ux_m.y = v[1];
    buf->parts_send_d[id_in_pack].ux_m.z = v[2];
    buf->parts_send_d[id_in_pack].ux_m.w = part_get_mass(&ptmps[i]);
  }
}

/**
 * @brief packs particle data for gradient tasks into CPU-side buffers for self
 * tasks
 */
__attribute__((always_inline)) INLINE static void gpu_pack_part_self_gradient(
    const struct cell* restrict c, struct gpu_offload_data* restrict buf) {

  const int count = c->hydro.count;
  const size_t local_pack_position = buf->pv.count_parts;

  const struct part *ptmps = c->hydro.parts;
  const float cellx = c->loc[0];
  const float celly = c->loc[1];
  const float cellz = c->loc[2];
  for (int i = 0; i < count; i++) {

    int id_in_pack = i + local_pack_position;
    const struct part *p = &ptmps[i];

    /* Data to be copied to GPU */
    const double *x = part_get_const_x(p);
    buf->parts_send_g[id_in_pack].x_h.x = x[0] - cellx;
    buf->parts_send_g[id_in_pack].x_h.y = x[1] - celly;
    buf->parts_send_g[id_in_pack].x_h.z = x[2] - cellz;
    buf->parts_send_g[id_in_pack].x_h.w = part_get_h(p);
    const float *v = part_get_const_v(&ptmps[i]);
    buf->parts_send_g[id_in_pack].ux_m.x = v[0];
    buf->parts_send_g[id_in_pack].ux_m.y = v[1];
    buf->parts_send_g[id_in_pack].ux_m.z = v[2];
    buf->parts_send_g[id_in_pack].ux_m.w = part_get_mass(p);
    buf->parts_send_g[id_in_pack].rho_avisc_u_c.x = part_get_rho(p);
    buf->parts_send_g[id_in_pack].rho_avisc_u_c.y = part_get_alpha_av(p);
    buf->parts_send_g[id_in_pack].rho_avisc_u_c.z = part_get_u(p);  // p.density.rot_v[0];
    buf->parts_send_g[id_in_pack].rho_avisc_u_c.w = part_get_soundspeed(p);
    //        p.force.soundspeed;  // p.density.rot_v[0];
  }
}

/**
 * @brief packs particle data for force tasks into CPU-side buffers for self
 * tasks
 */
__attribute__((always_inline)) INLINE static void gpu_pack_part_self_force(
    const struct cell* restrict c,
    struct gpu_offload_data *restrict buf) {

  const int count = c->hydro.count;
  const size_t local_pack_position = buf->pv.count_parts;

  const int pp = local_pack_position;
  const float cellx = c->loc[0];
  const float celly = c->loc[1];
  const float cellz = c->loc[2];

  /* Data to be copied to GPU local memory */
  const struct part *ptmps = c->hydro.parts;

  for (int i = 0; i < count; i++) {
    const struct part *p = &ptmps[i];
    const double *x = part_get_const_x(p);
    const int id_in_pack = i + pp;
    buf->parts_send_f[id_in_pack].x_h.x = x[0] - cellx;
    buf->parts_send_f[id_in_pack].x_h.y = x[1] - celly;
    buf->parts_send_f[id_in_pack].x_h.z = x[2] - cellz;
    buf->parts_send_f[id_in_pack].x_h.w = part_get_h(p);

    const float *v = part_get_const_v(p);
    buf->parts_send_f[id_in_pack].ux_m.x = v[0];
    buf->parts_send_f[id_in_pack].ux_m.y = v[1];
    buf->parts_send_f[id_in_pack].ux_m.z = v[2];
    buf->parts_send_f[id_in_pack].ux_m.w = part_get_mass(p);
    buf->parts_send_f[id_in_pack].f_bals_timebin_mintimebin_ngb.x = part_get_f_gradh(p);
    buf->parts_send_f[id_in_pack].f_bals_timebin_mintimebin_ngb.y = part_get_balsara(p);
    buf->parts_send_f[id_in_pack].f_bals_timebin_mintimebin_ngb.z = part_get_time_bin(p);

    buf->parts_send_f[id_in_pack].f_bals_timebin_mintimebin_ngb.w =
        part_get_timestep_limiter_min_ngb_time_bin(p);
    buf->parts_send_f[id_in_pack].rho_p_c_vsigi.x = part_get_rho(p);
    buf->parts_send_f[id_in_pack].rho_p_c_vsigi.y = part_get_pressure(p);
    buf->parts_send_f[id_in_pack].rho_p_c_vsigi.z = part_get_soundspeed(p);
    buf->parts_send_f[id_in_pack].rho_p_c_vsigi.w = part_get_v_sig(p);
    buf->parts_send_f[id_in_pack].u_alphavisc_alphadiff.x = part_get_u(p);
    buf->parts_send_f[id_in_pack].u_alphavisc_alphadiff.y = part_get_alpha_av(p);
    buf->parts_send_f[id_in_pack].u_alphavisc_alphadiff.z = part_get_alpha_diff(p);
  }
}

/**
 * @brief Unpacks the density data from GPU buffers of self tasks into particles
 */
__attribute__((always_inline)) INLINE static void gpu_unpack_part_self_density(
    struct cell* restrict c,
    const struct gpu_part_recv_d* restrict parts_buffer,
    const size_t pack_position,
    const size_t count, const struct engine *e){

  const struct gpu_part_recv_d *parts_tmp = &parts_buffer[pack_position];

  for (size_t i = 0; i < count; i++) {

    struct part *p = &c->hydro.parts[i];
    if (!part_is_active(p, e)) continue;

    struct gpu_part_recv_d p_tmp = parts_tmp[i];
    float4 rho_dh_wcount = p_tmp.rho_dh_wcount;
    float4 rot_ux_div_v = p_tmp.rot_ux_div_v;

    part_set_rho(p, part_get_rho(p) + rho_dh_wcount.x);
    part_set_rho_dh(p, part_get_rho_dh(p) + rho_dh_wcount.y);
    part_set_wcount(p, part_get_wcount(p) + rho_dh_wcount.z);
    part_set_wcount_dh(p, part_get_wcount_dh(p) + rho_dh_wcount.w);
    //    p->rho += rho_dh_wcount.x;
    //    p->density.rho_dh += rho_dh_wcount.y;
    //    p->density.wcount += rho_dh_wcount.z;
    //    p->density.wcount_dh += rho_dh_wcount.w;

    float *rot_v = part_get_rot_v(p);
    rot_v[0] += rot_ux_div_v.x;
    rot_v[1] += rot_ux_div_v.y;
    rot_v[2] += rot_ux_div_v.z;
    part_set_div_v(p, part_get_div_v(p) + rot_ux_div_v.w);
  }
}

/**
 * @brief Unpacks the gradient data from GPU buffers of self tasks into particles
 */
__attribute__((always_inline)) INLINE static void gpu_unpack_part_self_gradient(
    struct cell* restrict c,
    const struct gpu_part_recv_g* restrict parts_buffer,
    const size_t pack_position,
    const size_t count, const struct engine *e){

  const struct gpu_part_recv_g *parts_tmp = &parts_buffer[pack_position];

  for (size_t i = 0; i < count; i++) {

    struct part *p = &c->hydro.parts[i];
    if (!part_is_active(p, e)) continue;

    struct gpu_part_recv_g p_tmp = parts_tmp[i];

    part_set_v_sig(p, fmaxf(p_tmp.vsig_lapu_aviscmax.x, part_get_v_sig(p)));
    part_set_laplace_u(p, part_get_laplace_u(p) + p_tmp.vsig_lapu_aviscmax.y);
    part_set_alpha_visc_max_ngb(
        p, fmaxf(part_get_alpha_visc_max_ngb(p), p_tmp.vsig_lapu_aviscmax.z));
  }
}

/**
 * @brief Unpacks the force data from GPU buffers of self tasks into particles
 */
__attribute__((always_inline)) INLINE static void gpu_unpack_part_self_force(
    struct cell* restrict c,
    const struct gpu_part_recv_f* restrict parts_buffer,
    const size_t pack_position,
    const size_t count, const struct engine *e){

  const struct gpu_part_recv_f *parts_tmp = &parts_buffer[pack_position];

  for (size_t i = 0; i < count; i++) {

    struct part *p = &c->hydro.parts[i];
    if (!part_is_active(p, e)) continue;

    struct gpu_part_recv_f p_tmp = parts_tmp[i];

    float *a = part_get_a_hydro(p);
    part_set_a_hydro_ind(p, 0, a[0] + p_tmp.a_hydro.x);
    part_set_a_hydro_ind(p, 1, a[1] + p_tmp.a_hydro.y);
    part_set_a_hydro_ind(p, 2, a[2] + p_tmp.a_hydro.z);

    part_set_u_dt(p, p_tmp.udt_hdt_vsig_mintimebin_ngb.x + part_get_u_dt(p));

    part_set_h_dt(p, p_tmp.udt_hdt_vsig_mintimebin_ngb.y + part_get_h_dt(p));

    part_set_v_sig(
        p, fmaxf(p_tmp.udt_hdt_vsig_mintimebin_ngb.z, part_get_v_sig(p)));

    timebin_t min_ngb_time_bin = (int)(p_tmp.udt_hdt_vsig_mintimebin_ngb.w + 0.5f);
    part_set_timestep_limiter_min_ngb_time_bin(p, min_ngb_time_bin);
  }
}

/* TODO: IDEALLY, THIS SHOULD BE IDENTICAL FOR THE SELF TASKS.
 * PASS A CELL, BUFFER, INDEX TO COPY BACK. THIS REPLICATION IS
 * UNNECESSARY.*/
__attribute__((always_inline)) INLINE static void gpu_unpack_part_pair_density(
    struct cell* restrict c,
    const struct gpu_part_recv_d* restrict parts_buffer,
    const size_t pack_ind,
    const size_t count) {

  const struct gpu_part_recv_d *parts_tmp = &parts_buffer[pack_ind];

  for (size_t i = 0; i < count; i++) {
    /* TODO: WHY ARE WE NOT CHECKING WHETHER PARTICLE IS ACTIVE HERE???? */
    struct gpu_part_recv_d p_tmp = parts_tmp[i];
    struct part *p = &c->hydro.parts[i];
    part_set_rho(p, part_get_rho(p) + p_tmp.rho_dh_wcount.x);
    part_set_rho_dh(p, part_get_rho_dh(p) + p_tmp.rho_dh_wcount.y);
    part_set_wcount(p, part_get_wcount(p) + p_tmp.rho_dh_wcount.z);
    part_set_wcount_dh(p, part_get_wcount_dh(p) + p_tmp.rho_dh_wcount.w);
    const float *rot_v = part_get_rot_v(p);
    part_set_rot_v_ind(p, 0, rot_v[0] + p_tmp.rot_ux_div_v.x);
    part_set_rot_v_ind(p, 1, rot_v[1] + p_tmp.rot_ux_div_v.y);
    part_set_rot_v_ind(p, 2, rot_v[2] + p_tmp.rot_ux_div_v.z);
    part_set_div_v(p, part_get_div_v(p) + p_tmp.rot_ux_div_v.w);
  }
}

/* TODO: IDEALLY, THIS SHOULD BE IDENTICAL FOR THE SELF TASKS.
 * PASS A CELL, BUFFER, INDEX TO COPY BACK. THIS REPLICATION IS
 * UNNECESSARY.*/
__attribute__((always_inline)) INLINE static void gpu_unpack_part_pair_gradient(
    struct cell* restrict c,
    const struct gpu_part_recv_g* restrict parts_buffer,
    const size_t pack_ind,
    const size_t count) {

  const struct gpu_part_recv_g *parts_tmp = &parts_buffer[pack_ind];

  for (size_t i = 0; i < count; i++) {
    struct gpu_part_recv_g p_tmp = parts_tmp[i];
    struct part *p = &c->hydro.parts[i];

    part_set_v_sig(p, fmaxf(p_tmp.vsig_lapu_aviscmax.x, part_get_v_sig(p)));
    part_set_laplace_u(p, part_get_laplace_u(p) + p_tmp.vsig_lapu_aviscmax.y);
    part_set_alpha_visc_max_ngb(
        p, fmaxf(part_get_alpha_visc_max_ngb(p), p_tmp.vsig_lapu_aviscmax.z));
  }
}

/* TODO: IDEALLY, THIS SHOULD BE IDENTICAL FOR THE SELF TASKS.
 * PASS A CELL, BUFFER, INDEX TO COPY BACK. THIS REPLICATION IS
 * UNNECESSARY.*/
__attribute__((always_inline)) INLINE static void gpu_unpack_part_pair_force(
    struct cell * restrict c,
    const struct gpu_part_recv_f *restrict parts_buffer,
    const size_t pack_ind,
    const size_t count) {

  const struct gpu_part_recv_f *parts_tmp = &parts_buffer[pack_ind];

  for (size_t i = 0; i < count; i++) {
    struct gpu_part_recv_f p_tmp = parts_tmp[i];
    struct part *restrict p = &c->hydro.parts[i];

    float *a = part_get_a_hydro(p);
    part_set_a_hydro_ind(p, 0, a[0] + p_tmp.a_hydro.x);
    part_set_a_hydro_ind(p, 1, a[1] + p_tmp.a_hydro.y);
    part_set_a_hydro_ind(p, 2, a[2] + p_tmp.a_hydro.z);

    part_set_u_dt(p, p_tmp.udt_hdt_vsig_mintimebin_ngb.x + part_get_u_dt(p));
    part_set_h_dt(p, p_tmp.udt_hdt_vsig_mintimebin_ngb.y + part_get_h_dt(p));
    part_set_v_sig(
        p, fmaxf(p_tmp.udt_hdt_vsig_mintimebin_ngb.z, part_get_v_sig(p)));
    timebin_t min_ngb_time_bin =
        (int)(p_tmp.udt_hdt_vsig_mintimebin_ngb.w + 0.5f);
    part_set_timestep_limiter_min_ngb_time_bin(p, min_ngb_time_bin);
  }
}


__attribute__((always_inline)) INLINE static void gpu_pack_part_pair_density(
    const struct cell *restrict c, struct gpu_part_send_d *restrict parts_buffer,
    const int local_pack_position,
    const int count, const double3 shift, const int2 cstarts) {

  /* Data to be copied to GPU */
  for (int i = 0; i < count; i++) {
    const int id_in_pack = i + local_pack_position;
    const struct part *p = &c->hydro.parts[i];
    const double *x = part_get_const_x(p);
    parts_buffer[id_in_pack].x_p_h.x = x[0] - shift.x;
    parts_buffer[id_in_pack].x_p_h.y = x[1] - shift.y;
    parts_buffer[id_in_pack].x_p_h.z = x[2] - shift.z;
    parts_buffer[id_in_pack].x_p_h.w = part_get_h(p);
    const float *v = part_get_const_v(p);
    parts_buffer[id_in_pack].ux_m.x = v[0];
    parts_buffer[id_in_pack].ux_m.y = v[1];
    parts_buffer[id_in_pack].ux_m.z = v[2];
    parts_buffer[id_in_pack].ux_m.w = part_get_mass(p);
    parts_buffer[id_in_pack].cjs_cje.x = cstarts.x;
    parts_buffer[id_in_pack].cjs_cje.y = cstarts.y;
  }
}


__attribute__((always_inline)) INLINE static void gpu_pack_part_pair_gradient(
    const struct cell* restrict c,
    struct gpu_part_send_g* restrict parts_buffer,
    const int local_pack_position, const int count, const double3 shift,
    const int2 cstarts) {

  /* Data to be copied to GPU */
  const struct part *ptmps = c->hydro.parts;

  for (int i = 0; i < count; i++) {
    const int id_in_pack = i + local_pack_position;
    const struct part *p = &ptmps[i];
    const double *x = part_get_const_x(p);
    parts_buffer[id_in_pack].x_h.x = x[0] - shift.x;
    parts_buffer[id_in_pack].x_h.y = x[1] - shift.y;
    parts_buffer[id_in_pack].x_h.z = x[2] - shift.z;
    parts_buffer[id_in_pack].x_h.w = part_get_h(p);
    const float *v = part_get_const_v(&ptmps[i]);
    parts_buffer[id_in_pack].ux_m.x = v[0];
    parts_buffer[id_in_pack].ux_m.y = v[1];
    parts_buffer[id_in_pack].ux_m.z = v[2];
    parts_buffer[id_in_pack].ux_m.w = part_get_mass(p);
    parts_buffer[id_in_pack].rho_avisc_u_c.x = part_get_rho(p);
    parts_buffer[id_in_pack].rho_avisc_u_c.y = part_get_alpha_av(p);
    parts_buffer[id_in_pack].rho_avisc_u_c.z = part_get_u(p);
    parts_buffer[id_in_pack].rho_avisc_u_c.w = part_get_soundspeed(p);

    parts_buffer[id_in_pack].cjs_cje.x = cstarts.x;
    parts_buffer[id_in_pack].cjs_cje.y = cstarts.y;
  }
}


__attribute__((always_inline)) INLINE static void gpu_pack_part_pair_force(
    const struct cell* restrict c,
    struct gpu_part_send_f* restrict parts_buffer,
    const int local_pack_position, const int count, const double3 shift,
    const int2 cstarts) {

  const int pp = local_pack_position;
  const struct part *ptmps = c->hydro.parts;

  /*Data to be copied to GPU local memory*/
  for (int i = 0; i < count; i++) {
    const struct part *p = &ptmps[i];
    const double *x = part_get_const_x(p);
    const int id_in_pack = i + pp;
    parts_buffer[id_in_pack].x_h.x = x[0] - shift.x;
    parts_buffer[id_in_pack].x_h.y = x[1] - shift.y;
    parts_buffer[id_in_pack].x_h.z = x[2] - shift.z;

    parts_buffer[id_in_pack].x_h.w = part_get_h(p);

    const float *v = part_get_const_v(p);
    parts_buffer[id_in_pack].ux_m.x = v[0];
    parts_buffer[id_in_pack].ux_m.y = v[1];
    parts_buffer[id_in_pack].ux_m.z = v[2];
    parts_buffer[id_in_pack].ux_m.w = part_get_mass(p);
    parts_buffer[id_in_pack].f_bals_timebin_mintimebin_ngb.x = part_get_f_gradh(p);
    parts_buffer[id_in_pack].f_bals_timebin_mintimebin_ngb.y = part_get_balsara(p);
    parts_buffer[id_in_pack].f_bals_timebin_mintimebin_ngb.z =
        part_get_time_bin(p);
    parts_buffer[id_in_pack].f_bals_timebin_mintimebin_ngb.w =
        part_get_timestep_limiter_min_ngb_time_bin(p);
    parts_buffer[id_in_pack].rho_p_c_vsigi.x = part_get_rho(p);
    parts_buffer[id_in_pack].rho_p_c_vsigi.y = part_get_pressure(p);
    parts_buffer[id_in_pack].rho_p_c_vsigi.z = part_get_soundspeed(p);
    parts_buffer[id_in_pack].rho_p_c_vsigi.w = part_get_v_sig(p);
    parts_buffer[id_in_pack].u_alphavisc_alphadiff.x = part_get_u(p);
    parts_buffer[id_in_pack].u_alphavisc_alphadiff.y = part_get_alpha_av(p);
    parts_buffer[id_in_pack].u_alphavisc_alphadiff.z = part_get_alpha_diff(p);
    parts_buffer[id_in_pack].cjs_cje.x = cstarts.x;
    parts_buffer[id_in_pack].cjs_cje.y = cstarts.y;
  }
}


#endif /* GPU_PART_PACK_FUNCTIONS_H */
