/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c)
 *               2025 Abouzied M. A. Nasar (abouzied.nasar@manchester.ac.uk)
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

/* This object's header. */
#include "runner_gpu_pack_functions.h"

/* Local headers. */
#include "active.h"
#include "engine.h"
#include "runner_doiact_hydro.h"
#include "scheduler.h"
#include "space_getsid.h"
#include "timers.h"


/**
 * @brief packs particle data for density tasks into CPU-side buffers for self
 * tasks
 */
void gpu_pack_part_self_density(const struct cell* restrict c,
    struct gpu_offload_data *buf) {

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

    //    /*Initialise sums to zero before CPU/GPU copy*/
    //    const float4 zeroes = {0.0, 0.0, 0.0, 0.0};
    //    parts_aos_buffer[id_in_pack].rho_dh_wcount = zeroes;
    //    parts_aos_buffer[id_in_pack].rot_ux_div_v = zeroes;
  }
}

/**
 * @brief packs particle data for gradient tasks into CPU-side buffers for self
 * tasks
 */
void gpu_pack_part_self_gradient(const struct cell* restrict c,
                struct gpu_offload_data *buf) {

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
void gpu_pack_part_self_force(const struct cell* restrict c, struct gpu_offload_data *buf) {

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
void gpu_unpack_part_self_density(struct cell* restrict c,
    const struct gpu_part_recv_d* restrict parts_aos_buffer,
    const size_t pack_position,
    const size_t count, const struct engine *e){

  const struct gpu_part_recv_d *parts_tmp = &parts_aos_buffer[pack_position];

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
void gpu_unpack_part_self_gradient(struct cell* restrict c,
    const struct gpu_part_recv_g* restrict parts_aos_buffer,
    const size_t pack_position,
    const size_t count, const struct engine *e){

  const struct gpu_part_recv_g *parts_tmp = &parts_aos_buffer[pack_position];

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
void gpu_unpack_part_self_force(struct cell* restrict c,
    const struct gpu_part_recv_f* restrict parts_aos_buffer,
    const size_t pack_position,
    const size_t count, const struct engine *e){

  const struct gpu_part_recv_f *parts_tmp = &parts_aos_buffer[pack_position];

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
void gpu_unpack_part_pair_density(
    struct cell *c,
    const struct gpu_part_recv_d *parts_aos_buffer,
    const size_t pack_ind,
    const size_t count) {

  const struct gpu_part_recv_d *parts_tmp = &parts_aos_buffer[pack_ind];

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
void gpu_unpack_part_pair_gradient(
    struct cell *c,
    const struct gpu_part_recv_g *parts_aos_buffer,
    const size_t pack_ind,
    const size_t count) {

  const struct gpu_part_recv_g *parts_tmp = &parts_aos_buffer[pack_ind];

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
void gpu_unpack_part_pair_force(
    struct cell *c,
    const struct gpu_part_recv_f *parts_aos_buffer,
    const size_t pack_ind,
    const size_t count) {

  const struct gpu_part_recv_f *restrict parts_tmp = &parts_aos_buffer[pack_ind];

  /* int pp = local_pack_position; */
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


void gpu_pack_part_pair_density(
    const struct cell *c, struct gpu_part_send_d *parts_aos_buffer,
    const int local_pack_position,
    const int count, const double3 shift, const int2 cstarts) {

  /*Data to be copied to GPU*/
  for (int i = 0; i < count; i++) {
    const int id_in_pack = i + local_pack_position;
    const struct part *p = &c->hydro.parts[i];
    const double *x = part_get_const_x(p);
    parts_aos_buffer[id_in_pack].x_p_h.x = x[0] - shift.x;
    parts_aos_buffer[id_in_pack].x_p_h.y = x[1] - shift.y;
    parts_aos_buffer[id_in_pack].x_p_h.z = x[2] - shift.z;
    parts_aos_buffer[id_in_pack].x_p_h.w = part_get_h(p);
    const float *v = part_get_const_v(p);
    parts_aos_buffer[id_in_pack].ux_m.x = v[0];
    parts_aos_buffer[id_in_pack].ux_m.y = v[1];
    parts_aos_buffer[id_in_pack].ux_m.z = v[2];
    parts_aos_buffer[id_in_pack].ux_m.w = part_get_mass(p);
    parts_aos_buffer[id_in_pack].cjs_cje.x = cstarts.x;
    parts_aos_buffer[id_in_pack].cjs_cje.y = cstarts.y;
  }
}


void gpu_pack_part_pair_gradient(
    const struct cell* restrict c,
    struct gpu_part_send_g* parts_aos_buffer,
    const int local_pack_position, const int count, const double3 shift,
    const int2 cstarts) {

  /* Data to be copied to GPU */
  const struct part *ptmps = c->hydro.parts;

  for (int i = 0; i < count; i++) {
    const int id_in_pack = i + local_pack_position;
    const struct part *p = &ptmps[i];
    const double *x = part_get_const_x(p);
    parts_aos_buffer[id_in_pack].x_h.x = x[0] - shift.x;
    parts_aos_buffer[id_in_pack].x_h.y = x[1] - shift.y;
    parts_aos_buffer[id_in_pack].x_h.z = x[2] - shift.z;
    parts_aos_buffer[id_in_pack].x_h.w = part_get_h(p);
    const float *v = part_get_const_v(&ptmps[i]);
    parts_aos_buffer[id_in_pack].ux_m.x = v[0];
    parts_aos_buffer[id_in_pack].ux_m.y = v[1];
    parts_aos_buffer[id_in_pack].ux_m.z = v[2];
    parts_aos_buffer[id_in_pack].ux_m.w = part_get_mass(p);
    parts_aos_buffer[id_in_pack].rho_avisc_u_c.x = part_get_rho(p);
    parts_aos_buffer[id_in_pack].rho_avisc_u_c.y = part_get_alpha_av(p);
    parts_aos_buffer[id_in_pack].rho_avisc_u_c.z = part_get_u(p);
    parts_aos_buffer[id_in_pack].rho_avisc_u_c.w = part_get_soundspeed(p);

    parts_aos_buffer[id_in_pack].cjs_cje.x = cstarts.x;
    parts_aos_buffer[id_in_pack].cjs_cje.y = cstarts.y;
  }
}

void gpu_pack_part_pair_force(
    const struct cell* restrict c,
    struct gpu_part_send_f* parts_aos_buffer,
    const int local_pack_position, const int count, const double3 shift,
    const int2 cstarts) {

  const int pp = local_pack_position;
  const struct part *ptmps = c->hydro.parts;

  /*Data to be copied to GPU local memory*/
  for (int i = 0; i < count; i++) {
    const struct part *p = &ptmps[i];
    const double *x = part_get_const_x(p);
    const int id_in_pack = i + pp;
    parts_aos_buffer[id_in_pack].x_h.x = x[0] - shift.x;
    parts_aos_buffer[id_in_pack].x_h.y = x[1] - shift.y;
    parts_aos_buffer[id_in_pack].x_h.z = x[2] - shift.z;

    parts_aos_buffer[id_in_pack].x_h.w = part_get_h(p);

    const float *v = part_get_const_v(p);
    parts_aos_buffer[id_in_pack].ux_m.x = v[0];
    parts_aos_buffer[id_in_pack].ux_m.y = v[1];
    parts_aos_buffer[id_in_pack].ux_m.z = v[2];
    parts_aos_buffer[id_in_pack].ux_m.w = part_get_mass(p);
    parts_aos_buffer[id_in_pack].f_bals_timebin_mintimebin_ngb.x = part_get_f_gradh(p);
    parts_aos_buffer[id_in_pack].f_bals_timebin_mintimebin_ngb.y = part_get_balsara(p);
    parts_aos_buffer[id_in_pack].f_bals_timebin_mintimebin_ngb.z =
        part_get_time_bin(p);
    parts_aos_buffer[id_in_pack].f_bals_timebin_mintimebin_ngb.w =
        part_get_timestep_limiter_min_ngb_time_bin(p);
    parts_aos_buffer[id_in_pack].rho_p_c_vsigi.x = part_get_rho(p);
    parts_aos_buffer[id_in_pack].rho_p_c_vsigi.y = part_get_pressure(p);
    parts_aos_buffer[id_in_pack].rho_p_c_vsigi.z = part_get_soundspeed(p);
    parts_aos_buffer[id_in_pack].rho_p_c_vsigi.w = part_get_v_sig(p);
    parts_aos_buffer[id_in_pack].u_alphavisc_alphadiff.x = part_get_u(p);
    parts_aos_buffer[id_in_pack].u_alphavisc_alphadiff.y = part_get_alpha_av(p);
    parts_aos_buffer[id_in_pack].u_alphavisc_alphadiff.z = part_get_alpha_diff(p);
    parts_aos_buffer[id_in_pack].cjs_cje.x = cstarts.x;
    parts_aos_buffer[id_in_pack].cjs_cje.y = cstarts.y;
  }
}


void gpu_unpack_pair_density(
    const struct runner *r,
    struct cell *ci,
    struct cell *cj,
    const struct gpu_part_recv_d *parts_aos_buffer,
    size_t *pack_ind,
    size_t count_max_parts
    ) {

  const struct engine* e = r->e;

  if (!cell_is_active_hydro(ci, e) && !cell_is_active_hydro(cj, e)) {
    message("Inactive cell");
    return;
  }

  size_t count_ci = ci->hydro.count;
  size_t count_cj = cj->hydro.count;

#ifdef SWIFT_DEBUG_CHECKS
  if (*pack_ind + count_ci + count_cj >= count_max_parts) {
    error("Exceeded count_max_parts_tmp. Make arrays bigger! pack_ind is "
          "%lu, counts are %lu %lu, max is %lu",
          *pack_ind, count_ci, count_cj, count_max_parts);
  }
#endif

  if (cell_is_active_hydro(ci, e)){
    /* Pack the particle data into CPU-side buffers*/
    gpu_unpack_part_pair_density(ci, parts_aos_buffer, *pack_ind, count_ci);

    /* Increment packed index accordingly */
    *pack_ind += count_ci;
  }

  if (cell_is_active_hydro(cj, e)){
    /* Pack the particle data into CPU-side buffers*/
    gpu_unpack_part_pair_density(cj, parts_aos_buffer, *pack_ind, count_cj);

    /* Increment packed index accordingly */
    *pack_ind += count_cj;
  }
}


void gpu_unpack_pair_gradient(
    const struct runner *r,
    struct cell *ci,
    struct cell *cj,
    const struct gpu_part_recv_g *parts_aos_buffer,
    size_t *pack_ind,
    size_t count_max_parts){

  const struct engine* e = r->e;

  /* Anything to do here? */
  if (!cell_is_active_hydro(ci, e) && !cell_is_active_hydro(cj, e)) {
    return;
  }

  size_t count_ci = ci->hydro.count;
  size_t count_cj = cj->hydro.count;

#ifdef SWIFT_DEBUG_CHECKS
  if (*pack_ind + count_ci + count_cj >= count_max_parts) {
    error("Exceeded count_max_parts_tmp. Make arrays bigger! pack_ind is "
          "%lu, counts are %lu %lu, max is %lu",
          *pack_ind, count_ci, count_cj, count_max_parts);
  }
#endif

  if (cell_is_active_hydro(ci, e)){
    /* Pack the particle data into CPU-side buffers*/
    gpu_unpack_part_pair_gradient(ci, parts_aos_buffer, *pack_ind, count_ci);

    /* Increment packed index accordingly */
    *pack_ind += count_ci;
  }

  if (cell_is_active_hydro(cj, e)){
    /* Pack the particle data into CPU-side buffers*/
    gpu_unpack_part_pair_gradient(cj, parts_aos_buffer, *pack_ind, count_cj);

    /* Increment packed index accordingly */
    (*pack_ind) += count_cj;
  }
}


void gpu_unpack_pair_force(
    const struct runner *r,
    struct cell *ci,
    struct cell *cj,
    const struct gpu_part_recv_f *parts_aos_buffer,
    size_t *pack_ind,
    size_t count_max_parts){

  const struct engine* e = r->e;

  if (!cell_is_active_hydro(ci, e) && !cell_is_active_hydro(cj, e)) {
    return;
  }

  size_t count_ci = ci->hydro.count;
  size_t count_cj = cj->hydro.count;

#ifdef SWIFT_DEBUG_CHECKS
  if (*pack_ind + count_ci + count_cj >= count_max_parts) {
    error("Exceeded count_max_parts_tmp. Make arrays bigger! pack_ind is "
          "%lu, counts are %lu %lu, max is %lu",
          *pack_ind, count_ci, count_cj, count_max_parts);
  }
#endif

  if (cell_is_active_hydro(ci, e)){
    /* Pack the particle data into CPU-side buffers*/
    gpu_unpack_part_pair_force(ci, parts_aos_buffer, *pack_ind, count_ci);

    /* Increment packed index accordingly */
    *pack_ind += count_ci;
  }

  if (cell_is_active_hydro(cj, e)){
    /* Pack the particle data into CPU-side buffers*/
    gpu_unpack_part_pair_force(cj, parts_aos_buffer, *pack_ind, count_cj);

    /* Increment pack length accordingly */
    *pack_ind += count_cj;
  }
}

/**
 * @brief packs up two cells for the pair density GPU task
 */
void gpu_pack_pair_density(struct gpu_offload_data* buf,
    const struct runner *r, const struct cell *ci, const struct cell *cj,
    const double3 shift_tmp) {

  TIMER_TIC;

  /* Anything to do here? */
  const int count_ci = ci->hydro.count;
  const int count_cj = cj->hydro.count;
  if (count_ci == 0 || count_cj == 0) return;

  struct gpu_pack_vars* pack_vars = &buf->pv;

  /* Get how many particles we've packed until now */
  /* DOUBLE-CHECK THIS */
  size_t pack_ind = pack_vars->count_parts;

#ifdef SWIFT_DEBUG_CHECKS
  if (pack_ind + count_ci + count_cj >= 2 * pack_vars->count_max_parts) {
    error("Exceeded count_max_parts_tmp. Make arrays bigger! pack_ind %lu"
          "ci %i cj %i count_max %lu",
          pack_ind, count_ci, count_cj, pack_vars->count_max_parts);
  }
#endif

  /* Pack the particle data into CPU-side buffers. Start by assigning the shifts
   * (if positions shifts are required)*/
  const double3 shift_i = {shift_tmp.x + cj->loc[0], shift_tmp.y + cj->loc[1], shift_tmp.z + cj->loc[2]};

  /* Get first and last particles of cell i */
  const int2 cis_cie = {pack_ind, pack_ind + count_ci};

  /* Get first and last particles of cell j */
  const int2 cjs_cje = {pack_ind + count_ci, pack_ind + count_ci + count_cj};

  /* Pack cell i */
  gpu_pack_part_pair_density(ci, buf->parts_send_d, pack_ind, count_ci, shift_i, cjs_cje);

  /* Update the particles packed counter */
  pack_ind += count_ci;

  /* Do the same for cj */
  const double3 shift_j = {cj->loc[0], cj->loc[1], cj->loc[2]};

  gpu_pack_part_pair_density(cj, buf->parts_send_d, pack_ind, count_cj, shift_j, cis_cie);

  pack_ind += count_cj;

  /* Update incremented pack length accordingly */
  pack_vars->count_parts = pack_ind;

  TIMER_TOC(timer_dopair_gpu_pack_d);
}



/**
 * @brief packs up two cells for the pair density GPU task
 */
void gpu_pack_pair_gradient(struct gpu_offload_data* buf,
    const struct runner *r, const struct cell *ci, const struct cell *cj,
    const double3 shift_tmp) {

  TIMER_TIC;

  /* Anything to do here? */
  const int count_ci = ci->hydro.count;
  const int count_cj = cj->hydro.count;
  if (count_ci == 0 || count_cj == 0) return;

  struct gpu_pack_vars* pack_vars = &buf->pv;

  /* Get how many particles we've packed until now */
  size_t pack_ind = pack_vars->count_parts;

#ifdef SWIFT_DEBUG_CHECKS
  if (pack_ind + count_ci + count_cj >= 2 * pack_vars->count_max_parts) {
    error("Exceeded count_max_parts_tmp. Make arrays bigger! pack_ind %lu"
          "ci %i cj %i count_max %lu",
          pack_ind, count_ci, count_cj, pack_vars->count_max_parts);
  }
#endif

  /* Pack the particle data into CPU-side buffers. Start by assigning the shifts
   * (if positions shifts are required)*/
  const double3 shift_i = {shift_tmp.x + cj->loc[0], shift_tmp.y + cj->loc[1], shift_tmp.z + cj->loc[2]};

  /* Get first and last particles of cell i */
  const int2 cis_cie = {pack_ind, pack_ind + count_ci};

  /* Get first and last particles of cell j */
  const int2 cjs_cje = {pack_ind + count_ci, pack_ind + count_ci + count_cj};

  /* Pack cell i */
  gpu_pack_part_pair_gradient(ci, buf->parts_send_g, pack_ind, count_ci, shift_i, cjs_cje);

  /* Update the particles packed counter */
  pack_ind += count_ci;

  /* Do the same for cj */
  const double3 shift_j = {cj->loc[0], cj->loc[1], cj->loc[2]};

  gpu_pack_part_pair_gradient(cj, buf->parts_send_g, pack_ind, count_cj, shift_j, cis_cie);

  /* Update the particles packed counter */
  pack_ind += count_cj;

  /* Store incremented pack length accordingly */
  pack_vars->count_parts = pack_ind;

  TIMER_TOC(timer_dopair_gpu_pack_g);
}


void gpu_pack_pair_force(struct gpu_offload_data* buf,
    const struct runner *r, const struct cell *ci, const struct cell *cj,
    const double3 shift_tmp) {

  TIMER_TIC;

  /* Anything to do here? */

  /* Anything to do here? */
  const int count_ci = ci->hydro.count;
  const int count_cj = cj->hydro.count;
  if (count_ci == 0 || count_cj == 0) return;

  struct gpu_pack_vars* pack_vars = &buf->pv;

  /* Get how many particles we've packed until now */
  size_t pack_ind = pack_vars->count_parts;

#ifdef SWIFT_DEBUG_CHECKS
  if (pack_ind + count_ci + count_cj >= 2 * pack_vars->count_max_parts) {
    error("Exceeded count_max_parts_tmp. Make arrays bigger! pack_ind %lu"
          "ci %i cj %i count_max %lu",
          pack_ind, count_ci, count_cj, pack_vars->count_max_parts);
  }
#endif

  /* Pack the particle data into CPU-side buffers*/
  const double3 shift_i = {shift_tmp.x + cj->loc[0], shift_tmp.y + cj->loc[1],
                          shift_tmp.z + cj->loc[2]};

  /* Get first and last particles of cell i */
  const int2 cis_cie = {pack_ind, pack_ind + count_ci};

  /* Get first and last particles of cell j */
  const int2 cjs_cje = {pack_ind + count_ci, pack_ind + count_ci + count_cj};

  gpu_pack_part_pair_force(ci, buf->parts_send_f, pack_ind, count_ci, shift_i, cjs_cje);

  /* Update the particles packed counter */
  pack_ind += count_ci;

  /* Pack the particle data into CPU-side buffers*/
  const double3 shift_j = {cj->loc[0], cj->loc[1], cj->loc[2]};

  gpu_pack_part_pair_force(cj, buf->parts_send_f, pack_ind, count_cj, shift_j, cis_cie);

  /* Update the particles packed counter */
  pack_ind += count_cj;

  /* Store incremented pack length accordingly */
  pack_vars->count_parts = pack_ind;

  TIMER_TOC(timer_dopair_gpu_pack_f);
}
