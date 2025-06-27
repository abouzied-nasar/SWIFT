/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2020 Mladen Ivkovic (mladen.ivkovic@hotmail.com)
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
#ifndef SWIFT_RT_STRUCT_GEAR_H
#define SWIFT_RT_STRUCT_GEAR_H

/**
 * @file src/rt/GEAR/rt_struct.h
 * @brief Main header file for the GEAR M1 Closure radiative transfer struct.
 */


#include <stddef.h>

#include "inline.h"


/* Additional RT data in hydro particle struct */
struct rt_part_data {

  /* Radiation state vector. */
  struct {
    float energy_density;
    float flux[3];
  } radiation[RT_NGROUPS];

  /* Fluxes in the conservation law sense */
  struct {
    float energy;
    float flux[3];
  } flux[RT_NGROUPS];

  /* Particle RT time step. */
  float flux_dt;

  /* gradients of the radiation state. */
  /* for the flux[3][3] quantity:
   *    first index: x, y, z coordinate of the flux.
   *    Second index: gradient along x, y, z direction. */
  struct {
    float energy_density[3];
    float flux[3][3];
  } gradient[RT_NGROUPS];

  /* cell slope limiter quantities */
  /* array of length two: store min among all neighbours
   * at first index, store max among all neighbours at
   * second index */
  /* the Gizmo-style slope limiting doesn't help for RT as is,
   * so we're skipping it for now. */
  /* struct { */
  /*   float energy_density[2]; */
  /*   float flux[3][2]; */
  /*   [> float maxr; [> just use the hydro one <] <] */
  /* } limiter[RT_NGROUPS]; */

  /* Data for thermochemistry */
  struct {
    float mass_fraction_HI;         /* mass fraction taken by HI */
    float mass_fraction_HII;        /* mass fraction taken by HII */
    float mass_fraction_HeI;        /* mass fraction taken by HeI */
    float mass_fraction_HeII;       /* mass fraction taken by HeII */
    float mass_fraction_HeIII;      /* mass fraction taken by HeIII */
    float number_density_electrons; /* number density of electrons */
  } tchem;

#ifdef GIZMO_MFV_SPH
  /* Keep track of the actual mass fluxes of the gas species */
  struct {
    float HI;    /* mass fraction taken by HI */
    float HII;   /* mass fraction taken by HII */
    float HeI;   /* mass fraction taken by HeI */
    float HeII;  /* mass fraction taken by HeII */
    float HeIII; /* mass fraction taken by HeIII */
  } mass_flux;
#endif

#ifdef SWIFT_RT_DEBUG_CHECKS
  /* debugging data to store during entire run */

  /*! how much radiation this part received from stars during total lifetime */
  unsigned long long debug_radiation_absorbed_tot;

  /* data to store during one time step */

  /*! how many stars this part interacted with during injection*/
  /* Note: It's useless to write this in outputs, as it gets reset
   * at the end of every step. */
  int debug_iact_stars_inject;

  /*! calls from gradient interaction loop in actual function */
  int debug_calls_iact_gradient_interaction;

  /*! calls from transport interaction loop in actual function */
  int debug_calls_iact_transport_interaction;

  /* Task completion flags */

  /*! part got kicked? */
  int debug_kicked;

  /*! calls from ghost1 tasks */
  int debug_injection_done;

  /*! finalised computing gradients? */
  int debug_gradients_done;

  /*! transport step done? */
  int debug_transport_done;

  /*! thermochemistry done? */
  int debug_thermochem_done;

  /* Subcycling flags */

  /*! Current subcycle wrt (last) hydro step */
  int debug_nsubcycles;

#endif
};

/* Additional RT data in star particle struct */
struct rt_spart_data {

  /* Stellar energy emission that will be injected in to gas.
   * Total energy, not density, not rate! */
  float emission_this_step[RT_NGROUPS];

  /*! Neighbour weigths in each octant surrounding the star */
  float octant_weights[8];

#ifdef SWIFT_RT_DEBUG_CHECKS
  /* data to store during entire run */

  /*! how much radiation this star emitted during total lifetime */
  unsigned long long debug_radiation_emitted_tot;

  /* data to store during one time step */

  /*! how many hydro particles this particle interacted with
   * during injection */
  int debug_iact_hydro_inject;

  /*! how many hydro particles this particle interacted with
   * during injection prep*/
  int debug_iact_hydro_inject_prep;

  /*! stellar photon emisison rate computed? */
  int debug_emission_rate_set;

  /*! how much energy this star particle actually has injected into the gas */
  float debug_injected_energy[RT_NGROUPS];

  /*! how much energy this star particle actually has injected into the gas over
   * the entire run*/
  float debug_injected_energy_tot[RT_NGROUPS];

  /*! sum up total weights used during injection to compare consistency */
  float debug_psi_sum;
#endif
};


/* ===================================
 * RT Struct Getters and Setters
 * =================================== */

/**
 * Radiation energy density getters and setters
 */
static __attribute__((always_inline)) INLINE float
rt_part_data_get_radiation_energy_density(const struct rt_part_data* restrict rtd, const size_t g){
  return rtd->radiation[g].energy_density;
}

static __attribute__((always_inline)) INLINE float*
rt_part_data_get_radiation_energy_density_p(struct rt_part_data* restrict rtd, const size_t g){
  return &rtd->radiation[g].energy_density;
}

static __attribute__((always_inline)) INLINE const float*
rt_part_data_get_const_radiation_energy_density_p(const struct rt_part_data* restrict rtd, const size_t g){
  return &rtd->radiation[g].energy_density;
}

static __attribute__((always_inline)) INLINE void
rt_part_data_set_radiation_energy_density(struct rt_part_data* restrict rtd, const size_t g, const float energy_density){
  rtd->radiation[g].energy_density = energy_density;
}


/**
 * Radiation flux getters and setters
 */
static __attribute__((always_inline)) INLINE float*
rt_part_data_get_radiation_flux(struct rt_part_data* restrict rtd, const size_t group){
  return rtd->radiation[group].flux;
}

static __attribute__((always_inline)) INLINE const float*
rt_part_data_get_const_radiation_flux(const struct rt_part_data* restrict rtd, const size_t group){
  return rtd->radiation[group].flux;
}

static __attribute__((always_inline)) INLINE float
rt_part_data_get_radiation_flux_ind(const struct rt_part_data* restrict rtd, const size_t group, const size_t index){
  return rtd->radiation[group].flux[index];
}

static __attribute__((always_inline)) INLINE void
rt_part_data_set_radiation_flux(struct rt_part_data* restrict rtd, const size_t group, const float flux[3]){
  rtd->radiation[group].flux[0] = flux[0];
  rtd->radiation[group].flux[1] = flux[1];
  rtd->radiation[group].flux[2] = flux[2];
}

static __attribute__((always_inline)) INLINE void
rt_part_data_set_radiation_flux_ind(struct rt_part_data* restrict rtd, const size_t group, const size_t index, const float flux){
  rtd->radiation[group].flux[index] = flux;
}


/**
 * Energy flux getters and setters
 */
static __attribute__((always_inline)) INLINE float
rt_part_data_get_energy_flux(const struct rt_part_data* restrict rtd, const size_t g){
  return rtd->flux[g].energy;
}

static __attribute__((always_inline)) INLINE float*
rt_part_data_get_energy_flux_p(struct rt_part_data* restrict rtd, const size_t g){
  return &rtd->flux[g].energy;
}

static __attribute__((always_inline)) INLINE const float*
rt_part_data_get_const_energy_flux_p(const struct rt_part_data* restrict rtd, const size_t g){
  return &rtd->flux[g].energy;
}

static __attribute__((always_inline)) INLINE void
rt_part_data_set_energy_flux(struct rt_part_data* restrict rtd, const size_t g, const float energy_flux){
  rtd->flux[g].energy = energy_flux;
}



/**
 * Flux flux getters and setters
 */
static __attribute__((always_inline)) INLINE float*
rt_part_data_get_flux_flux(struct rt_part_data* restrict rtd, const size_t group){
  return rtd->flux[group].flux;
}

static __attribute__((always_inline)) INLINE float
rt_part_data_get_flux_flux_ind(const struct rt_part_data* restrict rtd, const size_t group, const size_t index){
  return rtd->flux[group].flux[index];
}

static __attribute__((always_inline)) INLINE const float*
rt_part_data_get_const_flux_flux(const struct rt_part_data* restrict rtd, const size_t group){
  return rtd->flux[group].flux;
}

static __attribute__((always_inline)) INLINE void
rt_part_data_set_flux_flux_ind(struct rt_part_data* restrict rtd, const size_t group, const size_t index, const float flux_flux){
  rtd->flux[group].flux[index] = flux_flux;
}

static __attribute__((always_inline)) INLINE void
rt_part_data_set_flux_flux(struct rt_part_data* restrict rtd, const size_t group, const float flux[3]){
  rtd->flux[group].flux[0] = flux[0];
  rtd->flux[group].flux[1] = flux[1];
  rtd->flux[group].flux[2] = flux[2];
}



/**
 * flux_dt
 */
static __attribute__((always_inline)) INLINE float
rt_part_data_get_flux_dt(const struct rt_part_data* restrict rtd){
  return rtd->flux_dt;
}

static __attribute__((always_inline)) INLINE float*
rt_part_data_get_flux_dt_p(struct rt_part_data* restrict rtd){
  return &rtd->flux_dt;
}

static __attribute__((always_inline)) INLINE const float*
rt_part_data_get_const_flux_dt_p(const struct rt_part_data* restrict rtd){
  return &rtd->flux_dt;
}

static __attribute__((always_inline)) INLINE void
rt_part_data_set_flux_dt(struct rt_part_data* restrict rtd, const float flux_dt){
  rtd->flux_dt = flux_dt;
}


// TODO(mivkov): Gradients getters/setters


/**
 * HI mass fraction
 */
static __attribute__((always_inline)) INLINE float
rt_part_data_get_mass_fraction_HI(const struct rt_part_data* restrict rtd){
  return rtd->tchem.mass_fraction_HI;
}

static __attribute__((always_inline)) INLINE float*
rt_part_data_get_mass_fraction_HI_p(struct rt_part_data* restrict rtd){
  return &rtd->tchem.mass_fraction_HI;
}

static __attribute__((always_inline)) INLINE const float*
rt_part_data_get_const_mass_fraction_HI_p(const struct rt_part_data* restrict rtd){
  return &rtd->tchem.mass_fraction_HI;
}

static __attribute__((always_inline)) INLINE void
rt_part_data_set_mass_fraction_HI(struct rt_part_data* restrict rtd, const float mass_fraction_HI){
  rtd->tchem.mass_fraction_HI = mass_fraction_HI;
}


/**
 * HII mass fraction
 */
static __attribute__((always_inline)) INLINE float
rt_part_data_get_mass_fraction_HII(const struct rt_part_data* restrict rtd){
  return rtd->tchem.mass_fraction_HII;
}

static __attribute__((always_inline)) INLINE float*
rt_part_data_get_mass_fraction_HII_p(struct rt_part_data* restrict rtd){
  return &rtd->tchem.mass_fraction_HII;
}

static __attribute__((always_inline)) INLINE const float*
rt_part_data_get_const_mass_fraction_HII_p(const struct rt_part_data* restrict rtd){
  return &rtd->tchem.mass_fraction_HII;
}

static __attribute__((always_inline)) INLINE void
rt_part_data_set_mass_fraction_HII(struct rt_part_data* restrict rtd, const float mass_fraction_HII){
  rtd->tchem.mass_fraction_HII = mass_fraction_HII;
}


/**
 * HeI mass fraction
 */
static __attribute__((always_inline)) INLINE float
rt_part_data_get_mass_fraction_HeI(const struct rt_part_data* restrict rtd){
  return rtd->tchem.mass_fraction_HeI;
}

static __attribute__((always_inline)) INLINE float*
rt_part_data_get_mass_fraction_HeI_p(struct rt_part_data* restrict rtd){
  return &rtd->tchem.mass_fraction_HeI;
}

static __attribute__((always_inline)) INLINE const float*
rt_part_data_get_const_mass_fraction_HeI_p(const struct rt_part_data* restrict rtd){
  return &rtd->tchem.mass_fraction_HeI;
}

static __attribute__((always_inline)) INLINE void
rt_part_data_set_mass_fraction_HeI(struct rt_part_data* restrict rtd, const float mass_fraction_HeI){
  rtd->tchem.mass_fraction_HeI = mass_fraction_HeI;
}


/**
 * HeII mass fraction
 */
static __attribute__((always_inline)) INLINE float
rt_part_data_get_mass_fraction_HeII(const struct rt_part_data* restrict rtd){
  return rtd->tchem.mass_fraction_HeII;
}

static __attribute__((always_inline)) INLINE float*
rt_part_data_get_mass_fraction_HeII_p(struct rt_part_data* restrict rtd){
  return &rtd->tchem.mass_fraction_HeII;
}

static __attribute__((always_inline)) INLINE const float*
rt_part_data_get_const_mass_fraction_HeII_p(const struct rt_part_data* restrict rtd){
  return &rtd->tchem.mass_fraction_HeII;
}

static __attribute__((always_inline)) INLINE void
rt_part_data_set_mass_fraction_HeII(struct rt_part_data* restrict rtd, const float mass_fraction_HeII){
  rtd->tchem.mass_fraction_HeII = mass_fraction_HeII;
}


/**
 * HeIII mass fraction
 */
static __attribute__((always_inline)) INLINE float
rt_part_data_get_mass_fraction_HeIII(const struct rt_part_data* restrict rtd){
  return rtd->tchem.mass_fraction_HeIII;
}

static __attribute__((always_inline)) INLINE float*
rt_part_data_get_mass_fraction_HeIII_p(struct rt_part_data* restrict rtd){
  return &rtd->tchem.mass_fraction_HeIII;
}

static __attribute__((always_inline)) INLINE const float*
rt_part_data_get_const_mass_fraction_HeIII_p(const struct rt_part_data* restrict rtd){
  return &rtd->tchem.mass_fraction_HeIII;
}

static __attribute__((always_inline)) INLINE void
rt_part_data_set_mass_fraction_HeIII(struct rt_part_data* restrict rtd, const float mass_fraction_HeIII){
  rtd->tchem.mass_fraction_HeIII = mass_fraction_HeIII;
}


/**
 * electron number density
 */
static __attribute__((always_inline)) INLINE float
rt_part_data_get_number_density_electrons(const struct rt_part_data* restrict rtd){
  return rtd->tchem.number_density_electrons;
}

static __attribute__((always_inline)) INLINE float*
rt_part_data_get_number_density_electrons_p(struct rt_part_data* restrict rtd){
  return &rtd->tchem.number_density_electrons;
}

static __attribute__((always_inline)) INLINE const float*
rt_part_data_get_const_number_density_electrons_p(const struct rt_part_data* restrict rtd){
  return &rtd->tchem.number_density_electrons;
}

static __attribute__((always_inline)) INLINE void
rt_part_data_set_number_density_electrons(struct rt_part_data* restrict rtd, const float number_density_electrons){
  rtd->tchem.number_density_electrons = number_density_electrons;
}


/**
 * HI mass flux
 */
static __attribute__((always_inline)) INLINE float
rt_part_data_get_mass_flux_HI(const struct rt_part_data* restrict rtd){
#ifdef GIZMO_MFV_SPH
  return rtd->mass_flux.HI;
#else
  return 0.f;
#endif
}

static __attribute__((always_inline)) INLINE float*
rt_part_data_get_mass_flux_HI_p(struct rt_part_data* restrict rtd){
#ifdef GIZMO_MFV_SPH
  return &rtd->mass_flux.HI;
#else
  return NULL;
#endif
}

static __attribute__((always_inline)) INLINE const float*
rt_part_data_get_const_mass_flux_HI_p(const struct rt_part_data* restrict rtd){
#ifdef GIZMO_MFV_SPH
  return &rtd->mass_flux.HI;
#else
  return NULL;
#endif
}

static __attribute__((always_inline)) INLINE void
rt_part_data_set_mass_flux_HI(struct rt_part_data* restrict rtd, const float mass_flux_HI){
#ifdef GIZMO_MFV_SPH
  rtd->mass_flux.HI = mass_flux_HI;
#endif
}


/**
 * HII mass flux
 */
static __attribute__((always_inline)) INLINE float
rt_part_data_get_mass_flux_HII(const struct rt_part_data* restrict rtd){
#ifdef GIZMO_MFV_SPH
  return rtd->mass_flux.HII;
#else
  return 0.f;
#endif
}

static __attribute__((always_inline)) INLINE float*
rt_part_data_get_mass_flux_HII_p(struct rt_part_data* restrict rtd){
#ifdef GIZMO_MFV_SPH
  return &rtd->mass_flux.HII;
#else
  return NULL;
#endif
}

static __attribute__((always_inline)) INLINE const float*
rt_part_data_get_const_mass_flux_HII_p(const struct rt_part_data* restrict rtd){
#ifdef GIZMO_MFV_SPH
  return &rtd->mass_flux.HII;
#else
  return NULL;
#endif
}

static __attribute__((always_inline)) INLINE void
rt_part_data_set_mass_flux_HII(struct rt_part_data* restrict rtd, const float mass_flux_HII){
#ifdef GIZMO_MFV_SPH
  rtd->mass_flux.HII = mass_flux_HII;
#endif
}


/**
 * HeI mass flux
 */
static __attribute__((always_inline)) INLINE float
rt_part_data_get_mass_flux_HeI(const struct rt_part_data* restrict rtd){
#ifdef GIZMO_MFV_SPH
  return rtd->mass_flux.HeI;
#else
  return 0.f;
#endif
}

static __attribute__((always_inline)) INLINE float*
rt_part_data_get_mass_flux_HeI_p(struct rt_part_data* restrict rtd){
#ifdef GIZMO_MFV_SPH
  return &rtd->mass_flux.HeI;
#else
  return NULL;
#endif
}

static __attribute__((always_inline)) INLINE const float*
rt_part_data_get_const_mass_flux_HeI_p(const struct rt_part_data* restrict rtd){
#ifdef GIZMO_MFV_SPH
  return &rtd->mass_flux.HeI;
#else
  return NULL;
#endif
}

static __attribute__((always_inline)) INLINE void
rt_part_data_set_mass_flux_HeI(struct rt_part_data* restrict rtd, const float mass_flux_HeI){
#ifdef GIZMO_MFV_SPH
  rtd->mass_flux.HeI = mass_flux_HeI;
#endif
}


/**
 * HeII mass flux
 */
static __attribute__((always_inline)) INLINE float
rt_part_data_get_mass_flux_HeII(const struct rt_part_data* restrict rtd){
#ifdef GIZMO_MFV_SPH
  return rtd->mass_flux.HeII;
#else
  return 0.f;
#endif
}

static __attribute__((always_inline)) INLINE float*
rt_part_data_get_mass_flux_HeII_p(struct rt_part_data* restrict rtd){
#ifdef GIZMO_MFV_SPH
  return &rtd->mass_flux.HeII;
#else
  return NULL;
#endif
}

static __attribute__((always_inline)) INLINE const float*
rt_part_data_get_const_mass_flux_HeII_p(const struct rt_part_data* restrict rtd){
#ifdef GIZMO_MFV_SPH
  return &rtd->mass_flux.HeII;
#else
  return NULL;
#endif
}

static __attribute__((always_inline)) INLINE void
rt_part_data_set_mass_flux_HeII(struct rt_part_data* restrict rtd, const float mass_flux_HeII){
#ifdef GIZMO_MFV_SPH
  rtd->mass_flux.HeII = mass_flux_HeII;
#endif
}


/**
 * HeIII mass flux
 */
static __attribute__((always_inline)) INLINE float
rt_part_data_get_mass_flux_HeIII(const struct rt_part_data* restrict rtd){
#ifdef GIZMO_MFV_SPH
  return rtd->mass_flux.HeIII;
#else
  return 0.f;
#endif
}

static __attribute__((always_inline)) INLINE float*
rt_part_data_get_mass_flux_HeIII_p(struct rt_part_data* restrict rtd){
#ifdef GIZMO_MFV_SPH
  return &rtd->mass_flux.HeIII;
#else
  return NULL;
#endif
}

static __attribute__((always_inline)) INLINE const float*
rt_part_data_get_const_mass_flux_HeIII_p(const struct rt_part_data* restrict rtd){
#ifdef GIZMO_MFV_SPH
  return &rtd->mass_flux.HeIII;
#else
  return NULL;
#endif
}

static __attribute__((always_inline)) INLINE void
rt_part_data_set_mass_flux_HeIII(struct rt_part_data* restrict rtd, const float mass_flux_HeIII){
#ifdef GIZMO_MFV_SPH
  rtd->mass_flux.HeIII = mass_flux_HeIII;
#endif
}



#endif /* SWIFT_RT_STRUCT_GEAR_H */
