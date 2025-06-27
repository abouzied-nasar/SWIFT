#ifndef RT_GEAR_PART_API_H
#define RT_GEAR_PART_API_H

#include "inline.h"
#include "part.h"
#include "rt_struct.h"

/* ===================================
 * Particle Struct Getters and Setters
 * =================================== */

/* forwards declarations */
/* struct part; */
/* struct rt_part_data* part_get_rt_data_p(struct part* restrict p); */
/* const struct rt_part_data* part_get_const_rt_data_p(const struct part*
 * restrict p); */
/* struct spart; */
/* struct rt_spart_data* spart_get_rt_data_p(struct part* restrict p); */
/* const struct rt_spart_data* spart_get_const_rt_data_p(const struct part*
 * restrict p); */

/**
 * Radiation energy density
 */
static __attribute__((always_inline)) INLINE float
part_get_rt_radiation_energy_density(const struct part* restrict p,
                                     const size_t g) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_radiation_energy_density(rtd, g);
}

static __attribute__((always_inline)) INLINE float*
part_get_rt_radiation_energy_density_p(struct part* restrict p,
                                       const size_t g) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  return rt_part_data_get_radiation_energy_density_p(rtd, g);
}

static __attribute__((always_inline)) INLINE const float*
part_get_const_rt_radiation_energy_density_p(const struct part* restrict p,
                                             const size_t g) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_const_radiation_energy_density_p(rtd, g);
}

static __attribute__((always_inline)) INLINE void
part_set_rt_radiation_energy_density(struct part* restrict p, const size_t g,
                                     const float energy_density) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  rt_part_data_set_radiation_energy_density(rtd, g, energy_density);
}

/**
 * Radiation flux
 */
static __attribute__((always_inline)) INLINE float* part_get_rt_radiation_flux(
    struct part* restrict p, const size_t group) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  return rt_part_data_get_radiation_flux(rtd, group);
}

static __attribute__((always_inline)) INLINE const float*
part_get_const_rt_radiation_flux(const struct part* restrict p,
                                 const size_t group) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_const_radiation_flux(rtd, group);
}

static __attribute__((always_inline)) INLINE float
part_get_rt_radiation_flux_ind(const struct part* restrict p,
                               const size_t group, const size_t index) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_radiation_flux_ind(rtd, group, index);
}

static __attribute__((always_inline)) INLINE void part_set_rt_radiation_flux(
    struct part* restrict p, const size_t group, const float flux[3]) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  rt_part_data_set_radiation_flux(rtd, group, flux);
}

static __attribute__((always_inline)) INLINE void
part_set_rt_radiation_flux_ind(struct part* restrict p, const size_t group,
                               const size_t index, const float flux) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  rt_part_data_set_radiation_flux_ind(rtd, group, index, flux);
}

/**
 * Energy flux
 */
static __attribute__((always_inline)) INLINE float part_get_rt_energy_flux(
    const struct part* restrict p, const size_t g) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_energy_flux(rtd, g);
}

static __attribute__((always_inline)) INLINE float* part_get_rt_energy_flux_p(
    struct part* restrict p, const size_t g) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  return rt_part_data_get_energy_flux_p(rtd, g);
}

static __attribute__((always_inline)) INLINE const float*
part_get_const_rt_energy_flux_p(const struct part* restrict p, const size_t g) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_const_energy_flux_p(rtd, g);
}

static __attribute__((always_inline)) INLINE void part_set_rt_energy_flux(
    struct part* restrict p, const size_t g, const float energy_flux) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  rt_part_data_set_energy_flux(rtd, g, energy_flux);
}

/**
 * Flux flux
 */
static __attribute__((always_inline)) INLINE float* part_get_rt_flux_flux(
    struct part* restrict p, const size_t group) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  return rt_part_data_get_flux_flux(rtd, group);
}

static __attribute__((always_inline)) INLINE float part_get_rt_flux_flux_ind(
    const struct part* restrict p, const size_t group, const size_t index) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_flux_flux_ind(rtd, group, index);
}

static __attribute__((always_inline)) INLINE const float*
part_get_const_rt_flux_flux(const struct part* restrict p, const size_t group) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_const_flux_flux(rtd, group);
}

static __attribute__((always_inline)) INLINE void part_set_rt_flux_flux(
    struct part* restrict p, const size_t group, const float flux_flux[3]) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  rt_part_data_set_flux_flux(rtd, group, flux_flux);
}

static __attribute__((always_inline)) INLINE void part_set_rt_flux_flux_ind(
    struct part* restrict p, const size_t group, const size_t index,
    const float flux_flux) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  rt_part_data_set_flux_flux_ind(rtd, group, index, flux_flux);
}

/**
 * Flux_dt
 */
static __attribute__((always_inline)) INLINE float part_get_rt_flux_dt(
    const struct part* restrict p) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_flux_dt(rtd);
}

static __attribute__((always_inline)) INLINE float* part_get_rt_flux_dt_p(
    struct part* restrict p) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  return rt_part_data_get_flux_dt_p(rtd);
}

static __attribute__((always_inline)) INLINE const float*
part_get_const_rt_flux_dt_p(const struct part* restrict p) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_const_flux_dt_p(rtd);
}

static __attribute__((always_inline)) INLINE void part_set_rt_flux_dt(
    struct part* restrict p, const float flux_dt) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  rt_part_data_set_flux_dt(rtd, flux_dt);
}

// TODO(mivkov): Gradients getters/setters

/**
 * HI mass fraction
 */
static __attribute__((always_inline)) INLINE float part_get_rt_mass_fraction_HI(
    const struct part* restrict p) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_mass_fraction_HI(rtd);
}

static __attribute__((always_inline)) INLINE float*
part_get_rt_mass_fraction_HI_p(struct part* restrict p) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  return rt_part_data_get_mass_fraction_HI_p(rtd);
}

static __attribute__((always_inline)) INLINE const float*
part_get_const_rt_mass_fraction_HI_p(const struct part* restrict p) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_const_mass_fraction_HI_p(rtd);
}

static __attribute__((always_inline)) INLINE void part_set_rt_mass_fraction_HI(
    struct part* restrict p, const float mass_fraction_HI) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  rt_part_data_set_mass_fraction_HI(rtd, mass_fraction_HI);
}

/**
 * HII mass fraction
 */
static __attribute__((always_inline)) INLINE float
part_get_rt_mass_fraction_HII(const struct part* restrict p) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_mass_fraction_HII(rtd);
}

static __attribute__((always_inline)) INLINE float*
part_get_rt_mass_fraction_HII_p(struct part* restrict p) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  return rt_part_data_get_mass_fraction_HII_p(rtd);
}

static __attribute__((always_inline)) INLINE const float*
part_get_const_rt_mass_fraction_HII_p(const struct part* restrict p) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_const_mass_fraction_HII_p(rtd);
}

static __attribute__((always_inline)) INLINE void part_set_rt_mass_fraction_HII(
    struct part* restrict p, const float mass_fraction_HII) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  rt_part_data_set_mass_fraction_HII(rtd, mass_fraction_HII);
}

/**
 * HeI mass fraction
 */
static __attribute__((always_inline)) INLINE float
part_get_rt_mass_fraction_HeI(const struct part* restrict p) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_mass_fraction_HeI(rtd);
}

static __attribute__((always_inline)) INLINE float*
part_get_rt_mass_fraction_HeI_p(struct part* restrict p) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  return rt_part_data_get_mass_fraction_HeI_p(rtd);
}

static __attribute__((always_inline)) INLINE const float*
part_get_const_rt_mass_fraction_HeI_p(const struct part* restrict p) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_const_mass_fraction_HeI_p(rtd);
}

static __attribute__((always_inline)) INLINE void part_set_rt_mass_fraction_HeI(
    struct part* restrict p, const float mass_fraction_HeI) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  rt_part_data_set_mass_fraction_HeI(rtd, mass_fraction_HeI);
}

/**
 * HeII mass fraction
 */
static __attribute__((always_inline)) INLINE float
part_get_rt_mass_fraction_HeII(const struct part* restrict p) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_mass_fraction_HeII(rtd);
}

static __attribute__((always_inline)) INLINE float*
part_get_rt_mass_fraction_HeII_p(struct part* restrict p) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  return rt_part_data_get_mass_fraction_HeII_p(rtd);
}

static __attribute__((always_inline)) INLINE const float*
part_get_const_rt_mass_fraction_HeII_p(const struct part* restrict p) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_const_mass_fraction_HeII_p(rtd);
}

static __attribute__((always_inline)) INLINE void
part_set_rt_mass_fraction_HeII(struct part* restrict p,
                               const float mass_fraction_HeII) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  rt_part_data_set_mass_fraction_HeII(rtd, mass_fraction_HeII);
}

/**
 * HeIII mass fraction
 */
static __attribute__((always_inline)) INLINE float
part_get_rt_mass_fraction_HeIII(const struct part* restrict p) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_mass_fraction_HeIII(rtd);
}

static __attribute__((always_inline)) INLINE float*
part_get_rt_mass_fraction_HeIII_p(struct part* restrict p) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  return rt_part_data_get_mass_fraction_HeIII_p(rtd);
}

static __attribute__((always_inline)) INLINE const float*
part_get_const_rt_mass_fraction_HeIII_p(const struct part* restrict p) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_const_mass_fraction_HeIII_p(rtd);
}

static __attribute__((always_inline)) INLINE void
part_set_rt_mass_fraction_HeIII(struct part* restrict p,
                                const float mass_fraction_HeIII) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  rt_part_data_set_mass_fraction_HeIII(rtd, mass_fraction_HeIII);
}

/**
 * electron number density
 */
static __attribute__((always_inline)) INLINE float
part_get_rt_number_density_electrons(const struct part* restrict p) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_number_density_electrons(rtd);
}

static __attribute__((always_inline)) INLINE float*
part_get_rt_number_density_electrons_p(struct part* restrict p) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  return rt_part_data_get_number_density_electrons_p(rtd);
}

static __attribute__((always_inline)) INLINE const float*
part_get_const_rt_number_density_electrons_p(const struct part* restrict p) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_const_number_density_electrons_p(rtd);
}

static __attribute__((always_inline)) INLINE void
part_set_rt_number_density_electrons(struct part* restrict p,
                                     const float number_density_electrons) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  rt_part_data_set_number_density_electrons(rtd, number_density_electrons);
}

/**
 * HI mass flux
 */
static __attribute__((always_inline)) INLINE float part_get_rt_mass_flux_HI(
    const struct part* restrict p) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_mass_flux_HI(rtd);
}

static __attribute__((always_inline)) INLINE float* part_get_rt_mass_flux_HI_p(
    struct part* restrict p) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  return rt_part_data_get_mass_flux_HI_p(rtd);
}

static __attribute__((always_inline)) INLINE const float*
part_get_const_rt_mass_flux_HI_p(const struct part* restrict p) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_const_mass_flux_HI_p(rtd);
}

static __attribute__((always_inline)) INLINE void part_set_rt_mass_flux_HI(
    struct part* restrict p, const float mass_flux_HI) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  rt_part_data_set_mass_flux_HI(rtd, mass_flux_HI);
}

/**
 * HII mass flux
 */
static __attribute__((always_inline)) INLINE float part_get_rt_mass_flux_HII(
    const struct part* restrict p) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_mass_flux_HII(rtd);
}

static __attribute__((always_inline)) INLINE float* part_get_rt_mass_flux_HII_p(
    struct part* restrict p) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  return rt_part_data_get_mass_flux_HII_p(rtd);
}

static __attribute__((always_inline)) INLINE const float*
part_get_const_rt_mass_flux_HII_p(const struct part* restrict p) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_const_mass_flux_HII_p(rtd);
}

static __attribute__((always_inline)) INLINE void part_set_rt_mass_flux_HII(
    struct part* restrict p, const float mass_flux_HII) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  rt_part_data_set_mass_flux_HII(rtd, mass_flux_HII);
}

/**
 * HeI mass flux
 */
static __attribute__((always_inline)) INLINE float part_get_rt_mass_flux_HeI(
    const struct part* restrict p) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_mass_flux_HeI(rtd);
}

static __attribute__((always_inline)) INLINE float* part_get_rt_mass_flux_HeI_p(
    struct part* restrict p) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  return rt_part_data_get_mass_flux_HeI_p(rtd);
}

static __attribute__((always_inline)) INLINE const float*
part_get_const_rt_mass_flux_HeI_p(const struct part* restrict p) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_const_mass_flux_HeI_p(rtd);
}

static __attribute__((always_inline)) INLINE void part_set_rt_mass_flux_HeI(
    struct part* restrict p, const float mass_flux_HeI) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  rt_part_data_set_mass_flux_HeI(rtd, mass_flux_HeI);
}

/**
 * HeII mass flux
 */
static __attribute__((always_inline)) INLINE float part_get_rt_mass_flux_HeII(
    const struct part* restrict p) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_mass_flux_HeII(rtd);
}

static __attribute__((always_inline)) INLINE float*
part_get_rt_mass_flux_HeII_p(struct part* restrict p) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  return rt_part_data_get_mass_flux_HeII_p(rtd);
}

static __attribute__((always_inline)) INLINE const float*
part_get_const_rt_mass_flux_HeII_p(const struct part* restrict p) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_const_mass_flux_HeII_p(rtd);
}

static __attribute__((always_inline)) INLINE void part_set_rt_mass_flux_HeII(
    struct part* restrict p, const float mass_flux_HeII) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  rt_part_data_set_mass_flux_HeII(rtd, mass_flux_HeII);
}

/**
 * HeIII mass flux
 */
static __attribute__((always_inline)) INLINE float part_get_rt_mass_flux_HeIII(
    const struct part* restrict p) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_mass_flux_HeIII(rtd);
}

static __attribute__((always_inline)) INLINE float*
part_get_rt_mass_flux_HeIII_p(struct part* restrict p) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  return rt_part_data_get_mass_flux_HeIII_p(rtd);
}

static __attribute__((always_inline)) INLINE const float*
part_get_const_rt_mass_flux_HeIII_p(const struct part* restrict p) {
  const struct rt_part_data* rtd = part_get_const_rt_data_p(p);
  return rt_part_data_get_const_mass_flux_HeIII_p(rtd);
}

static __attribute__((always_inline)) INLINE void part_set_rt_mass_flux_HeIII(
    struct part* restrict p, const float mass_flux_HeIII) {
  struct rt_part_data* rtd = part_get_rt_data_p(p);
  rt_part_data_set_mass_flux_HeIII(rtd, mass_flux_HeIII);
}

#endif
