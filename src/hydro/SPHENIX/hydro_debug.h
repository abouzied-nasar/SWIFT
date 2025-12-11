/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2019 Josh Borrow (joshua.borrow@durham.ac.uk) &
 *                    Matthieu Schaller (schaller@strw.leidenuniv.nl)
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

#ifndef SWIFT_SPHENIX_HYDRO_DEBUG_H
#define SWIFT_SPHENIX_HYDRO_DEBUG_H

/**
 * @file SPHENIX/hydro_debug.h
 * @brief Density-Energy conservative implementation of SPH,
 *        with added SPHENIX physics (Borrow 2020) (Debugging routines)
 */

#include "hydro.h"

__attribute__((always_inline)) INLINE static void hydro_debug_particle(
    const struct part *p, const struct xpart *xp) {
  warning("[PID%lld] part:", part_get_id(p));
  warning(
      "[PID%lld] x=[%.3e,%.3e,%.3e], "
      "v=[%.3e,%.3e,%.3e], a=[%.3e,%.3e,%.3e], "
      "u=%.3e, du/dt=%.3e v_sig=%.3e, P=%.3e, "
      "h=%.3e, dh/dt=%.3e wcount=%d, m=%.3e, dh_drho=%.3e, rho=%.3e, "
      "alpha=%.3e, "
      "time_bin=%d",
      part_get_id(p), part_get_x_ind(p, 0), part_get_x_ind(p, 1),
      part_get_x_ind(p, 2), part_get_v_ind(p, 0), part_get_v_ind(p, 1),
      part_get_v_ind(p, 2), part_get_a_hydro_ind(p, 0),
      part_get_a_hydro_ind(p, 1), part_get_a_hydro_ind(p, 2), part_get_u(p),
      part_get_u_dt(p), part_get_v_sig(p), hydro_get_comoving_pressure(p),
      part_get_h(p), part_get_h_dt(p), (int)part_get_wcount(p),
      part_get_mass(p), part_get_rho_dh(p), part_get_rho(p),
      part_get_alpha_av(p), part_get_time_bin(p));
  if (xp != NULL) {
    warning("[PID%lld] xpart:", part_get_id(p));
    warning("[PID%lld] v_full=[%.3e,%.3e,%.3e]", part_get_id(p), xp->v_full[0],
            xp->v_full[1], xp->v_full[2]);
  }
}

#endif /* SWIFT_SPHENIX_HYDRO_DEBUG_H */
