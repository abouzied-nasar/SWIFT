/* This object's header. */
#include "runner_gpu_pack_functions.h"

/* Local headers. */
#include "active.h"
#include "engine.h"
#include "runner_doiact_hydro.h"
#include "scheduler.h"
#include "space_getsid.h"
#include "timers.h"

/* #include <stdatomic.h> */


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
    const struct part_aos_f4_recv_d* restrict parts_aos_buffer,
    const size_t pack_position,
    const size_t count, const struct engine *e){

  const struct part_aos_f4_recv_d *parts_tmp = &parts_aos_buffer[pack_position];

  for (size_t i = 0; i < count; i++) {

    struct part *p = &c->hydro.parts[i];
    if (!part_is_active(p, e)) continue;

    struct part_aos_f4_recv_d p_tmp = parts_tmp[i];
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
    const struct part_aos_f4_recv_g* restrict parts_aos_buffer,
    const size_t pack_position,
    const size_t count, const struct engine *e){

  const struct part_aos_f4_recv_g *parts_tmp = &parts_aos_buffer[pack_position];

  for (size_t i = 0; i < count; i++) {

    struct part *p = &c->hydro.parts[i];
    if (!part_is_active(p, e)) continue;

    struct part_aos_f4_recv_g p_tmp = parts_tmp[i];

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
    const struct part_aos_f4_recv_f* restrict parts_aos_buffer,
    const size_t pack_position,
    const size_t count, const struct engine *e){

  const struct part_aos_f4_recv_f *parts_tmp = &parts_aos_buffer[pack_position];

  for (size_t i = 0; i < count; i++) {

    struct part *p = &c->hydro.parts[i];
    if (!part_is_active(p, e)) continue;

    struct part_aos_f4_recv_f p_tmp = parts_tmp[i];

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
    const struct part_aos_f4_recv_d *parts_aos_buffer,
    const size_t pack_ind,
    const size_t count) {

  const struct part_aos_f4_recv_d *parts_tmp = &parts_aos_buffer[pack_ind];

  for (size_t i = 0; i < count; i++) {
    /* TODO: WHY ARE WE NOT CHECKING WHETHER PARTICLE IS ACTIVE HERE???? */
    struct part_aos_f4_recv_d p_tmp = parts_tmp[i];
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
    const struct part_aos_f4_recv_g *parts_aos_buffer,
    const size_t pack_ind,
    const size_t count) {

  const struct part_aos_f4_recv_g *parts_tmp = &parts_aos_buffer[pack_ind];

  for (size_t i = 0; i < count; i++) {
    struct part_aos_f4_recv_g p_tmp = parts_tmp[i];
    struct part *p = &c->hydro.parts[i];
    //      const float v_sig = p->viscosity.v_sig;
    //      p->viscosity.v_sig = fmaxf(p_tmp.vsig_lapu_aviscmax.x, v_sig);
    //      p->diffusion.laplace_u += p_tmp.vsig_lapu_aviscmax.y;
    //      const float max_ngb = p->force.alpha_visc_max_ngb;
    //      p->force.alpha_visc_max_ngb = fmaxf(p_tmp.vsig_lapu_aviscmax.z,
    //      max_ngb);

    part_set_v_sig(p, fmaxf(p_tmp.vsig_lapu_aviscmax.x, part_get_v_sig(p)));
    part_set_laplace_u(p, part_get_laplace_u(p) + p_tmp.vsig_lapu_aviscmax.y);
    part_set_alpha_visc_max_ngb(
        p, fmaxf(part_get_alpha_visc_max_ngb(p), p_tmp.vsig_lapu_aviscmax.z));
  }
}

void unpack_neat_pair_aos_f4_f(
    struct runner *r, struct cell *restrict c,
    struct part_aos_f4_recv_f *restrict parts_aos_buffer, int tid,
    int local_pack_position, int count, const struct engine *e) {

  struct part_aos_f4_recv_f *restrict parts_tmp =
      &parts_aos_buffer[local_pack_position];
  if (cell_is_active_hydro(c, e)) {
    /* int pp = local_pack_position; */
    for (int i = 0; i < count; i++) {
      struct part_aos_f4_recv_f p_tmp = parts_tmp[i];
      struct part *restrict p = &c->hydro.parts[i];

      //      c->hydro.parts[i].a_hydro[0] += parts_aos_buffer[j].a_hydro.x;
      //      c->hydro.parts[i].a_hydro[1] += parts_aos_buffer[j].a_hydro.y;
      //      c->hydro.parts[i].a_hydro[2] += parts_aos_buffer[j].a_hydro.z;

      float *a = part_get_a_hydro(p);
      part_set_a_hydro_ind(p, 0, a[0] + p_tmp.a_hydro.x);
      part_set_a_hydro_ind(p, 1, a[1] + p_tmp.a_hydro.y);
      part_set_a_hydro_ind(p, 2, a[2] + p_tmp.a_hydro.z);

      //      c->hydro.parts[i].u_dt +=
      //          parts_aos_buffer[j].udt_hdt_vsig_mintimebin_ngb.x;

      part_set_u_dt(p, p_tmp.udt_hdt_vsig_mintimebin_ngb.x + part_get_u_dt(p));

      //      c->hydro.parts[i].force.h_dt +=
      //          parts_aos_buffer[j].udt_hdt_vsig_mintimebin_ngb.y;

      part_set_h_dt(p, p_tmp.udt_hdt_vsig_mintimebin_ngb.y + part_get_h_dt(p));

      //      c->hydro.parts[i].viscosity.v_sig =
      //          fmaxf(parts_aos_buffer[j].udt_hdt_vsig_mintimebin_ngb.z,
      //                c->hydro.parts[i].viscosity.v_sig);

      part_set_v_sig(
          p, fmaxf(p_tmp.udt_hdt_vsig_mintimebin_ngb.z, part_get_v_sig(p)));

      //      c->hydro.parts[i].limiter_data.min_ngb_time_bin =
      //          (int)(parts_aos_buffer[j].udt_hdt_vsig_mintimebin_ngb.w +
      //          0.5f);
      timebin_t min_ngb_time_bin =
          (int)(p_tmp.udt_hdt_vsig_mintimebin_ngb.w + 0.5f);
      part_set_timestep_limiter_min_ngb_time_bin(p, min_ngb_time_bin);
    }
  }
}


void gpu_pack_part_pair_density(
    const struct cell *c, struct part_aos_f4_send_d *parts_aos_buffer,
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
    struct part_aos_f4_send_g* parts_aos_buffer,
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

void pack_neat_pair_aos_f4_f(
    struct cell *__restrict c, struct part_aos_f4_send_f *__restrict parts_aos,
    int tid, const int local_pack_position, const int count, const float3 shift,
    const int2 cstarts) {
  //  const struct part *restrict ptmps;
  //  ptmps = c->hydro.parts;
  const int pp = local_pack_position;
  const struct part *ptmps = c->hydro.parts;
  /*Data to be copied to GPU local memory*/
  for (int i = 0; i < count; i++) {
    const struct part *p = &ptmps[i];
    const double *x = part_get_const_x(p);
    const int id_in_pack = i + pp;
    parts_aos[id_in_pack].x_h.x = x[0] - shift.x;
    parts_aos[id_in_pack].x_h.y = x[1] - shift.y;
    parts_aos[id_in_pack].x_h.z = x[2] - shift.z;
    //    parts_aos[i + pp].x_h.x = c->hydro.parts[i].x[0] - cellx;
    //    parts_aos[i + pp].x_h.y = c->hydro.parts[i].x[1] - celly;
    //    parts_aos[i + pp].x_h.z = c->hydro.parts[i].x[2] - cellz;

    parts_aos[id_in_pack].x_h.w = part_get_h(p);

    const float *v = part_get_const_v(p);
    parts_aos[id_in_pack].ux_m.x = v[0];
    parts_aos[id_in_pack].ux_m.y = v[1];
    parts_aos[id_in_pack].ux_m.z = v[2];
    parts_aos[id_in_pack].ux_m.w = part_get_mass(p);
    parts_aos[id_in_pack].f_bals_timebin_mintimebin_ngb.x = part_get_f_gradh(p);
    parts_aos[id_in_pack].f_bals_timebin_mintimebin_ngb.y = part_get_balsara(p);
    parts_aos[id_in_pack].f_bals_timebin_mintimebin_ngb.z =
        part_get_time_bin(p);
    parts_aos[id_in_pack].f_bals_timebin_mintimebin_ngb.w =
        part_get_timestep_limiter_min_ngb_time_bin(p);
    parts_aos[id_in_pack].rho_p_c_vsigi.x = part_get_rho(p);
    parts_aos[id_in_pack].rho_p_c_vsigi.y = part_get_pressure(p);
    parts_aos[id_in_pack].rho_p_c_vsigi.z = part_get_soundspeed(p);
    parts_aos[id_in_pack].rho_p_c_vsigi.w = part_get_v_sig(p);
    parts_aos[id_in_pack].u_alphavisc_alphadiff.x = part_get_u(p);
    parts_aos[id_in_pack].u_alphavisc_alphadiff.y = part_get_alpha_av(p);
    parts_aos[id_in_pack].u_alphavisc_alphadiff.z = part_get_alpha_diff(p);
    parts_aos[id_in_pack].cjs_cje.x = cstarts.x;
    parts_aos[id_in_pack].cjs_cje.y = cstarts.y;
  }
}


void gpu_unpack_pair_density(
    const struct runner *r,
    struct cell *ci,
    struct cell *cj,
    const struct part_aos_f4_recv_d *parts_aos_buffer,
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
    const struct part_aos_f4_recv_g *parts_aos_buffer,
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

void runner_do_ci_cj_gpu_unpack_neat_aos_f4_f(
    struct runner *r, struct cell *ci, struct cell *cj,
    struct part_aos_f4_recv_f *parts_aos_buffer, int timer, size_t *pack_length,
    int tid, int count_max_parts_tmp, const struct engine *e) {

  if (!cell_is_active_hydro(ci, e) && !cell_is_active_hydro(cj, e)) {
    message("Inactive cell");
    return;
  }
  int count_ci = ci->hydro.count;
  int count_cj = cj->hydro.count;
  int local_pack_position = (*pack_length);

#ifdef SWIFT_DEBUG_CHECKS
  if (local_pack_position + count_ci + count_cj >= count_max_parts_tmp) {
    error("Exceeded count_max_parts_tmp. Make arrays bigger! pack_length is "
          "%lu pointer to pack_length is %p, local_pack_position is % i, "
          "count is %i",
          (*pack_length), pack_length, local_pack_position, count_ci);
  }
#endif

  /* Pack the particle data into CPU-side buffers*/
  unpack_neat_pair_aos_f4_f(r, ci, parts_aos_buffer, tid, local_pack_position,
                            count_ci, e);
  local_pack_position += count_ci;
  /* Pack the particle data into CPU-side buffers*/
  unpack_neat_pair_aos_f4_f(r, cj, parts_aos_buffer, tid, local_pack_position,
                            count_cj, e);
  /* Increment pack length accordingly */
  (*pack_length) += count_ci + count_cj;
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
    fprintf(stderr,
            "Exceeded count_max_parts_tmp. Make arrays bigger! pack_ind %lu"
            "ci %i cj %i count_max %lu",
            pack_ind, count_ci, count_cj, pack_vars->count_max_parts);
    error();
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
    fprintf(stderr,
            "Exceeded count_max_parts_tmp. Make arrays bigger! pack_ind %lu"
            "ci %i cj %i count_max %lu",
            pack_ind, count_ci, count_cj, pack_vars->count_max_parts);
    error();
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


void runner_do_ci_cj_gpu_pack_neat_aos_f4_f(
    struct runner *r, struct cell *restrict ci, struct cell *restrict cj,
    struct part_aos_f4_send_f *restrict parts_aos_buffer, int timer,
    size_t *pack_length, int tid, int count_max_parts_tmp, const int count_ci,
    const int count_cj, double3 shift_tmp) {

  TIMER_TIC;

  /* Anything to do here? */
  if (ci->hydro.count == 0) return;

  int local_pack_position = (*pack_length);

#ifdef SWIFT_DEBUG_CHECKS
  if (local_pack_position + count_ci + count_cj >= 2 * count_max_parts_tmp) {
    error("Exceeded count_max_parts_tmp. Make arrays bigger! Pack pos %i"
          "ci %i cj %i count_max %i",
          local_pack_position, count_ci, count_cj, count_max_parts_tmp);
  }
#endif

  /* Pack the particle data into CPU-side buffers*/
  const float3 shift_i = {shift_tmp.x + cj->loc[0], shift_tmp.y + cj->loc[1],
                          shift_tmp.z + cj->loc[2]};
  const int lpp1 = local_pack_position;

  const int2 cis_cie = {local_pack_position, local_pack_position + count_ci};

  const int2 cjs_cje = {local_pack_position + count_ci,
                        local_pack_position + count_ci + count_cj};

  pack_neat_pair_aos_f4_f(ci, parts_aos_buffer, tid, lpp1, count_ci, shift_i,
                          cjs_cje);

  local_pack_position += count_ci;
  /* Pack the particle data into CPU-side buffers*/
  const float3 shift_j = {cj->loc[0], cj->loc[1], cj->loc[2]};
  const int lpp2 = local_pack_position;

  pack_neat_pair_aos_f4_f(cj, parts_aos_buffer, tid, lpp2, count_cj, shift_j,
                          cis_cie);
  /* Increment pack length accordingly */
  (*pack_length) += count_ci + count_cj;

  if (timer) TIMER_TOC(timer_doself_gpu_pack);
}
