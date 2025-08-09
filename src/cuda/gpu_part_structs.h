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
struct gpu_part_send_d {
#ifdef WITH_CUDA
  /*! Particle position and h -> x, y, z, h */
  float4 x_p_h;

  /*! Particle predicted velocity and mass -> ux, uy, uz, m */
  float4 ux_m;

  /*! Markers for where neighbour cell j starts and stops in array indices for
   * pair tasks*/
  int2 cjs_cje;

#endif
} __attribute__((aligned(SWIFT_STRUCT_ALIGNMENT)));

struct gpu_part_recv_d {
#ifdef WITH_CUDA

  /* Derivative of the neighbour number with respect to h; w_count_dh */
  float4 rho_dh_wcount;

  /*! Particle velocity curl; rot_ux and velocity divergence; div_v */
  float4 rot_ux_div_v;

#endif
};

/*Container for particle data requierd for force calcs*/
struct gpu_part_send_f {
#ifdef WITH_CUDA

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

#endif
};

/*Container for particle data requierd for force calcs*/
struct gpu_part_recv_f {
#ifdef WITH_CUDA

  /*Result: Values output to global GPU memory*/
  /* change of u and h with dt, v_sig and returned value of
   * minimum neighbour timebin */
  float4 udt_hdt_vsig_mintimebin_ngb;
  /*Particle acceleration vector*/
  float3 a_hydro;

#endif
};

/*Container for particle data requierd for gradient calcs*/
struct gpu_part_send_g {
#ifdef WITH_CUDA

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

#endif
};

/*Container for particle data requierd for gradient calcs*/
struct gpu_part_recv_g {
#ifdef WITH_CUDA

  /* viscosity information results */
  float3 vsig_lapu_aviscmax;

#endif
};

#ifdef __cplusplus
}
#endif

#endif  // CUDA_GPU_PART_STRUCTS_H
