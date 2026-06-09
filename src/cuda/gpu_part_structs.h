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

/*! Container for particle data required for density calcs */
struct gpu_part_data_d {
#ifdef WITH_CUDA
//TODO: This needs changing to doubles too. Darn it...
/*! Particle position and h -> x, y, z, h */
  float4 __align__(16) x_h;

  /*! Particle predicted velocity and mass -> ux, uy, uz, m */
  float4 __align__(16) vx_m;

#endif
};

/*! Container for cell positions */
struct gpu_cell_pos {
#ifdef WITH_CUDA

  /*! Cell position. This is set as the last entry in the
   * range of particles contained within a cell (i.e
   * N+1 contains info for cell position)*/
  double4 x;

#endif
};

/*! Over-arching union used to switch
 * between particle data and cell position. Saves us copying
 * cell positions to GPU individually*/
struct gpu_part_send_d {
#ifdef WITH_CUDA
  union {
    /*! Container for particle data required for density calcs */
    struct gpu_part_data_d p_data;
    /*! Container for cell positions for density calcs */
    struct gpu_cell_pos c_loc;
  };
#endif
};

/*! Container for particle data sent back to CPU for density calcs */
struct gpu_part_recv_d {
#ifdef WITH_CUDA

  /*! rho, rho_dh, wcount, wcount_dh */
  float4 rho_rhodh_wcount_wcount_dh;

  /*! Particle velocity curl; rot_ux and velocity divergence; div_v */
  float4 rot_vx_div_v;

#endif
};

/*! Container for particle data required for gradient calcs */
struct gpu_part_data_g {
#ifdef WITH_CUDA

  /*! Particle position & smoothing length */
  float4 __align__(16) x_h;

  /*! Particle velocity and mass */
  float4 __align__(16) vx_m;

  /*TODO: aviscmax is no longer needed as we do not use it on GPU. Remove!*/
  /*! Particle internal energy u, density speed of sound and alpha visc
   * Better to use float4 than float3 even if we only need space for 3 vars*/
  float4 __align__(16) u_rho_c_aviscmax;

  /*! viscosity information required for particle i when comparing to neighbours*/
  /*Leave as float4 for now. Gives best perf for pipeline async prefetching in kernel
   * due to __align__(16) filling bus*/
  float4 __align__(16) avisc_vsig;

#endif
};

/*Particle and cell position data required for GPU density computations*/
struct gpu_part_send_g{
  union {
    /*! Container for particle data required for density calcs */
    struct gpu_part_data_g p_data;
    /*! Container for cell positions for density calcs */
    struct gpu_cell_pos c_loc;
  };
} ;

/*! Container for particle data sent back to CPU for gradient calcs */
struct gpu_part_recv_g {
#ifdef WITH_CUDA

  /*! viscosity information results. Better to use float4 than float3
   * even if we have 3 vars */
  float4 aviscmax_vsig_lapu;

#endif
};

/*! Container for particle data required for force calcs */
struct gpu_part_data_f {
#ifdef WITH_CUDA

  /* Data required for the calculation: Values read to local GPU memory */

  /*! Particle positions, smoothing length */
  float4 __align__(16) x_h;

  /*! Particle predicted velocity and mass */
  float4 __align__(16) vx_m;

  /*! internal energy, density, variable smoothing length term f, pressure */
  float4 __align__(16) u_rho_f_p;

  /*! balsara, Particle speed of sound, alpha constants for
   * viscosity and diffusion */
  float4 __align__(16) bals_c_avisc_adiff;

  /*! Particle timebin, initial value of min neighbour timebin, start
   * and end index of particles to be interacted with in particle buffer
   * arrays */
  /*TODO: Change this to remove pjs and pje as will no longer be required*/
  int4 __align__(16) timebin_minngbtimebin;

#endif
};

/*! Container for particle data required for force calcs */
struct gpu_part_send_f {
#ifdef WITH_CUDA
  union {
    /*! Container for particle data required for density calcs */
    struct gpu_part_data_f p_data;
    /*! Container for cell positions for density calcs */
    struct gpu_cell_pos c_loc;
  };
#endif
} ;

/*! Container for particle data sent back to CPU for force calcs */
struct gpu_part_recv_f {
#ifdef WITH_CUDA

  /*! Particle acceleration vector */
  float3 a_hydro;

  /*Note: Needed to revert back to float2 and int
   * due to CUDA atomicMin requiring int*/
  /*! change of u and h with dt, v_sig */
  float2 udt_hdt;

  /*! change of u and h with dt, v_sig */
  int minngbtb;

#endif
};

#ifdef __cplusplus
}
#endif

#endif  // CUDA_GPU_PART_STRUCTS_H
