#ifndef CUDA_GPU_PART_STRUCTS_H
#define CUDA_GPU_PART_STRUCTS_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef WITH_CUDA
#include <vector_types.h>
#endif

/* Config parameters. */
#include "../../config.h"
#include "../align.h"
#include "../timeline.h"

/*Container for particle data requierd for density calcs*/
struct part_aos_f4_send_d {
  /*! Particle position and h -> x, y, z, h */
  float4 x_p_h;

  /*! Particle predicted velocity and mass -> ux, uy, uz, m */
  float4 ux_m;

  //  /*Temporary trial to see if doing shifts on GPU works*/
  //  float3 shift;

  /*Markers for where neighbour cell j starts and stops in array indices for
   * pair tasks*/
  int2 cjs_cje;

} __attribute__((aligned(SWIFT_STRUCT_ALIGNMENT)));

struct part_aos_f4_recv_d {
  /* Density information; rho */
  /*! Derivative of density with respect to h; rho_dh,
   * Neighbour number count; w_count
   * * Derivative of the neighbour number with respect to h; w_count_dh */
  float4 rho_dh_wcount;
  /*! Particle velocity curl; rot_ux and
   * velocity divergence; div_v */
  float4 rot_ux_div_v;
} ;

/*Container for particle data required for density calcs*/
struct part_aos_f4_d {
  /*! Particle position and h -> x, y, z, h */
  float4 x_p_h;

  /*! Particle predicted velocity and mass -> ux, uy, uz, m */
  float4 ux_m;
  /* Density information; rho */
  /*! Derivative of density with respect to h; rho_dh,
   * Neighbour number count; w_count
   * * Derivative of the neighbour number with respect to h; w_count_dh */
  float4 rho_dh_wcount;

  /*! Particle velocity curl; rot_ux and
   * velocity divergence; div_v */
  float4 rot_ux_div_v;

} ;

/*Container for particle data required for force calcs*/
struct part_aos_f {

  /*! Particle position. */
  double x_p;
  double y_p;
  double z_p;

  /*! Particle predicted velocity. */
  float ux;
  float uy;
  float uz;
  /*! Particle mass. */
  float mass;
  /*! Particle smoothing length. */
  float h;
  /*! Particle density. */
  float rho;
  /*! Particle pressure. */
  float pressure;

  /* Density information */
  /*! Speed of sound. */
  float soundspeed;
  /*! Variable smoothing length term */
  float f;
  /*! Derivative of density with respect to h */
  float balsara;
  /*! Particle velocity curl. */
  float alpha_visc;
  float a_hydrox;
  float a_hydroy;
  float a_hydroz;
  float alpha_diff;

  /* viscosity information */
  /*! Internal energy */
  float u;
  float u_dt;
  /*! h time derivative */
  float h_dt;
  float v_sig;

  /* timestep stuff */
  /*! Time-step length */
  int time_bin;
  int min_ngb_time_bin;
} part_aos_f;

/*Container for particle data requierd for force calcs*/
struct part_aos_f4_f {

  /*Data required for the calculation:
  Values read to local GPU memory*/
  /*! Particle position smoothing length */
  float4 x_h;
  /*! Particle predicted velocity and mass */
  float4 ux_m;
  /*! Variable smoothing length term f, balsara, timebin
   * and initial value of min neighbour timebin */
  float4 f_bals_timebin_mintimebin_ngb;
  /*! Particle density, pressure, speed of sound & v_sig to read*/
  float4 rho_p_c_vsigi;
  /*! Particle Internal energy u, alpha constants for visc and diff */
  float3 u_alphavisc_alphadiff;

  /*Result: Values output to global GPU memory*/
  /* change of u and h with dt, v_sig and returned value of
   * minimum neighbour timebin */
  float4 udt_hdt_vsig_mintimebin_ngb;
  /*Particle acceleration vector*/
  float3 a_hydro;

};

/*Container for particle data requierd for force calcs*/
struct part_aos_f4_f_send {

  /*Data required for the calculation:
  Values read to local GPU memory*/
  /*! Particle position smoothing length */
  float4 x_h;
  /*! Particle predicted velocity and mass */
  float4 ux_m;
  /*! Variable smoothing length term f, balsara, timebin
   * and initial value of min neighbour timebin */
  float4 f_bals_timebin_mintimebin_ngb;
  /*! Particle density, pressure, speed of sound & v_sig to read*/
  float4 rho_p_c_vsigi;
  /*! Particle Internal energy u, alpha constants for visc and diff */
  float3 u_alphavisc_alphadiff;

  int2 cjs_cje;

};

/*Container for particle data requierd for force calcs*/
struct part_aos_f4_f_recv {

  /*Result: Values output to global GPU memory*/
  /* change of u and h with dt, v_sig and returned value of
   * minimum neighbour timebin */
  float4 udt_hdt_vsig_mintimebin_ngb;
  /*Particle acceleration vector*/
  float3 a_hydro;

};

/*Container for particle data requierd for gradient calcs*/
struct part_aos_g {

  /*! Particle position. */
  double x_p;
  double y_p;
  double z_p;

  /*! Particle velocity. */
  float ux;
  float uy;
  float uz;
  /*! Particle mass. */
  float mass;
  /*! Particle smoothing length. */
  float h;
  /*! Particle density. */
  float rho;

  /* viscosity information */
  float visc_alpha;
  float laplace_u;
  float alpha_visc_max_ngb;
  float v_sig;

  float u;

  float soundspeed;

  /* timestep stuff */
  /*! Time-step length */
  int time_bin;
};

/*Container for particle data requierd for gradient calcs*/
struct part_aos_f4_g {

  /*! Particle position & smoothing length */
  float4 x_h;

  /*! Particle velocity and mass */
  float4 ux_m;

  /*! Particle density alpha visc internal energy u and speed of sound c */
  float4 rho_avisc_u_c;

  /* viscosity information results */
  float3 vsig_lapu_aviscmax_empty;

};

/*Container for particle data requierd for gradient calcs*/
struct part_aos_f4_g_send {

  /*! Particle position & smoothing length */
  float4 x_h;

  /*! Particle velocity and mass */
  float4 ux_m;

  /*! Particle density alpha visc internal energy u and speed of sound c */
  float4 rho_avisc_u_c;

  /* viscosity information results */
  float3 vsig_lapu_aviscmax;

  /*Data for cell start and end*/
  int2 cjs_cje;

};

/*Container for particle data requierd for gradient calcs*/
struct part_aos_f4_g_recv {

  /* viscosity information results */
  float3 vsig_lapu_aviscmax;

};

#ifdef __cplusplus
}
#endif

/* #else [> ifdef WITH_CUDA <] */
/* MAKE EMPTY STRUCTS HERE SO IT COMPILES. */
/* #endif [> ifdef WITH_CUDA <] */

#endif  // CUDA_GPU_PART_STRUCTS_H
