/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (C) 2015 Matthieu Schaller (schaller@strw.leidenuniv.nl).
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

/* Some standard headers. */
#include <config.h>

/* Some standard headers. */
#include <stdlib.h>

/* Includes. */
#include "swift.h"

int main(int argc, char *argv[]) {

  size_t Ngas = 0;
  size_t Ngpart = 0;
  size_t Ngpart_background = 0;
  size_t Nspart = 0;
  size_t Nbpart = 0;
  size_t Nsink = 0;
  size_t Nnupart = 0;
  int flag_entropy_ICs = -1;
  int i, j, k;
  double dim[3];
  struct part *parts = NULL;
  struct gpart *gparts = NULL;
  struct spart *sparts = NULL;
  struct bpart *bparts = NULL;
  struct sink *sinks = NULL;
  struct ic_info ics_metadata;
  strcpy(ics_metadata.group_name, "NoSUCH");

  /* Default unit system */
  struct unit_system us;
  units_init_cgs(&us);

  /* Properties of the ICs */
  const double boxSize = 1.;
  const size_t L = 4;
  const double rho = 2.;
#ifdef CHEMISTRY_GRACKLE
  const float he_density = rho * 0.24;
#endif

  /* Read data */
  read_ic_single("input.hdf5", &us, dim, &parts, &gparts, &sinks, &sparts,
                 &bparts, &Ngas, &Ngpart, &Ngpart_background, &Nnupart, &Nsink,
                 &Nspart, &Nbpart, &flag_entropy_ICs,
                 /*with_hydro=*/1,
                 /*with_gravity=*/1,
                 /*with_sink=*/0,
                 /*with_stars=*/0,
                 /*with_black_holes=*/0,
                 /*with_cosmology=*/0,
                 /*cleanup_h=*/0,
                 /*cleanup_sqrt_a=*/0,
                 /*h=*/1., /*a=*/1., /*n_threads=*/1, /*dry_run=*/0,
                 /*remap_ids=*/0, &ics_metadata);

  /* Check global properties read are correct */
  assert(dim[0] == boxSize);
  assert(dim[1] == boxSize);
  assert(dim[2] == boxSize);
  assert(Ngas == L * L * L);
  assert(Ngpart == L * L * L);

  /* Check particles */
  for (size_t n = 0; n < Ngas; ++n) {

    /* Check that indices are in a reasonable range */
    unsigned long long index = part_get_id(&parts[n]);
    assert(index < Ngas);

    /* Check masses */
    float mass = hydro_get_mass(&parts[n]);
    float correct_mass = boxSize * boxSize * boxSize * rho / Ngas;
    assert(mass == correct_mass);

    /* Check smoothing length */
    float h = part_get_h(&parts[n]);
    float correct_h = 2.251 * boxSize / L;
    assert(h == correct_h);

    /* Check velocity */
    assert(part_get_v_ind(&parts[n], 0) == 0.);
    assert(part_get_v_ind(&parts[n], 1) == 0.);
    assert(part_get_v_ind(&parts[n], 2) == 0.);

    /* Check positions */
    k = index % 4;
    j = ((index - k) / 4) % 4;
    i = (index - k - 4 * j) / 16;
    double correct_x = i * boxSize / L + boxSize / (2 * L);
    double correct_y = j * boxSize / L + boxSize / (2 * L);
    double correct_z = k * boxSize / L + boxSize / (2 * L);
    assert(part_get_x_ind(&parts[n], 0) == correct_x);
    assert(part_get_x_ind(&parts[n], 1) == correct_y);
    assert(part_get_x_ind(&parts[n], 2) == correct_z);

    /* Check accelerations */
    assert(part_get_a_hydro_ind(&parts[n], 0) == 0.);
    assert(part_get_a_hydro_ind(&parts[n], 1) == 0.);
    assert(part_get_a_hydro_ind(&parts[n], 2) == 0.);

#ifdef CHEMISTRY_GRACKLE
    assert(parts[n].chemistry_data.he_density == he_density);
#endif
  }

  /* Clean-up */
  free(parts);
  free(gparts);

  return 0;
}
