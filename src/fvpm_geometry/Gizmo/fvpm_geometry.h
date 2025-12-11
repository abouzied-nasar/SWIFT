#ifndef SWIFT_FVPM_GEOMETRY_GIZMO_H
#define SWIFT_FVPM_GEOMETRY_GIZMO_H

#include "const.h"
#include "part.h"

#include <config.h>

/**
 * @file Gizmo/fvpm_geometry.h
 * @brief Functions related to the Gizmo FVPM geometry struct collection,
 * in particular the collection of the data required for the matrix needed
 * for gradients.
 * This was moved here so we can cleanly couple GEAR-RT on top of SPH
 * hydrodynamics while avoiding code replication.
 */

#if defined(RT_GEAR) && defined(GIZMO_MFM_SPH)
/* Some functions clash here. MFM resets and does some geometry centroid
 * stuff, while GEAR-RT, which uses MFV, doesn't. So we'd need to split the
 * functions for RT and for hydro use.
 * However, it is very unlikely we'll ever actually use that combination,
 * so leaving it as-is for now. */
#error "Combining GIZMO MFM and GEAR-RT not implemented yet."
#endif

#if defined(GIZMO_MFV_SPH) || defined(RT_GEAR)
#include "./MFV/fvpm_geometry.h"
#elif defined(GIZMO_MFM_SPH)
#include "./MFM/fvpm_geometry.h"
#endif

/**
 * @brief Check if the gradient matrix for this particle is well behaved.
 *
 * @param p Particle.
 * @return 1 if the gradient matrix is well behaved, 0 otherwise.
 */
__attribute__((always_inline)) INLINE static int
fvpm_part_geometry_well_behaved(const struct part *restrict p) {

  const struct fvpm_geometry_struct *geometry =
      part_get_const_fvpm_geometry_p(p);
  return geometry->wcorr > const_gizmo_min_wcorr;
}

/**
 * @brief Collect the data needed for the matrix construction.
 */
__attribute__((always_inline)) INLINE static void
fvpm_accumulate_geometry_and_matrix(struct part *restrict pi, const float wi,
                                    const float dx[3]) {

  struct fvpm_geometry_struct *geometry = part_get_fvpm_geometry_p(pi);

  /* these are eqns. (1) and (2) in the Gizmo theory summary */
  geometry->volume += wi;
  for (int k = 0; k < 3; k++)
    for (int l = 0; l < 3; l++) geometry->matrix_E[k][l] += dx[k] * dx[l] * wi;
}

__attribute__((always_inline)) INLINE static void fvpm_geometry_init(
    struct part *restrict p) {

  struct fvpm_geometry_struct *geometry = part_get_fvpm_geometry_p(p);

  geometry->volume = 0.0f;
  geometry->matrix_E[0][0] = 0.0f;
  geometry->matrix_E[0][1] = 0.0f;
  geometry->matrix_E[0][2] = 0.0f;
  geometry->matrix_E[1][0] = 0.0f;
  geometry->matrix_E[1][1] = 0.0f;
  geometry->matrix_E[1][2] = 0.0f;
  geometry->matrix_E[2][0] = 0.0f;
  geometry->matrix_E[2][1] = 0.0f;
  geometry->matrix_E[2][2] = 0.0f;

  /* reset the centroid variables used for the velocity correction in MFV */
  fvpm_reset_centroids(p);
}

/**
 * @brief Finish the computation of the matrix.
 *
 * @param p the particle to work on
 * @param ihdim 1/h^{dim}
 */
__attribute__((always_inline)) INLINE static void
fvpm_compute_volume_and_matrix(struct part *restrict p, const float ihdim) {

  struct fvpm_geometry_struct *geometry = part_get_fvpm_geometry_p(p);

  /* Final operation on the geometry. */
  /* we multiply with the smoothing kernel normalization ih3 and calculate the
   * volume */
  const float volume_inv = ihdim * (geometry->volume + kernel_root);
  const float volume = 1.0f / volume_inv;
  geometry->volume = volume;

  /* we multiply with the smoothing kernel normalization */
  geometry->matrix_E[0][0] *= ihdim;
  geometry->matrix_E[0][1] *= ihdim;
  geometry->matrix_E[0][2] *= ihdim;
  geometry->matrix_E[1][0] *= ihdim;
  geometry->matrix_E[1][1] *= ihdim;
  geometry->matrix_E[1][2] *= ihdim;
  geometry->matrix_E[2][0] *= ihdim;
  geometry->matrix_E[2][1] *= ihdim;
  geometry->matrix_E[2][2] *= ihdim;

  /* normalise the centroids for MFV */
  fvpm_normalise_centroid(p, part_get_wcount(p));

  /* Check the condition number to see if we have a stable geometry. */
  const float condition_number_E =
      geometry->matrix_E[0][0] * geometry->matrix_E[0][0] +
      geometry->matrix_E[0][1] * geometry->matrix_E[0][1] +
      geometry->matrix_E[0][2] * geometry->matrix_E[0][2] +
      geometry->matrix_E[1][0] * geometry->matrix_E[1][0] +
      geometry->matrix_E[1][1] * geometry->matrix_E[1][1] +
      geometry->matrix_E[1][2] * geometry->matrix_E[1][2] +
      geometry->matrix_E[2][0] * geometry->matrix_E[2][0] +
      geometry->matrix_E[2][1] * geometry->matrix_E[2][1] +
      geometry->matrix_E[2][2] * geometry->matrix_E[2][2];

  float condition_number = 0.0f;
  if (invert_dimension_by_dimension_matrix(geometry->matrix_E) != 0) {
    /* something went wrong in the inversion; force bad condition number */
    condition_number = const_gizmo_max_condition_number + 1.0f;
  } else {
    const float condition_number_Einv =
        geometry->matrix_E[0][0] * geometry->matrix_E[0][0] +
        geometry->matrix_E[0][1] * geometry->matrix_E[0][1] +
        geometry->matrix_E[0][2] * geometry->matrix_E[0][2] +
        geometry->matrix_E[1][0] * geometry->matrix_E[1][0] +
        geometry->matrix_E[1][1] * geometry->matrix_E[1][1] +
        geometry->matrix_E[1][2] * geometry->matrix_E[1][2] +
        geometry->matrix_E[2][0] * geometry->matrix_E[2][0] +
        geometry->matrix_E[2][1] * geometry->matrix_E[2][1] +
        geometry->matrix_E[2][2] * geometry->matrix_E[2][2];

    condition_number =
        hydro_dimension_inv * sqrtf(condition_number_E * condition_number_Einv);
  }

  if (condition_number > const_gizmo_max_condition_number &&
      geometry->wcorr > const_gizmo_min_wcorr) {
#ifdef GIZMO_PATHOLOGICAL_ERROR
    error("Condition number larger than %g (%g)!",
          const_gizmo_max_condition_number, condition_number);
#endif
#ifdef GIZMO_PATHOLOGICAL_WARNING
    message("Condition number too large: %g (> %g, p->id: %llu)!",
            condition_number, const_gizmo_max_condition_number, part_get_id(p));
#endif
    /* add a correction to the number of neighbours for this particle */
    geometry->wcorr = const_gizmo_w_correction_factor * geometry->wcorr;
  }
}

#endif /* SWIFT_FVPM_GEOMETRY_GIZMO_H */
