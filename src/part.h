/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2012 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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
#ifndef SWIFT_PART_H
#define SWIFT_PART_H

/* Config parameters. */
#include <config.h>

/* Standard headers. */
#include <stddef.h>

/* MPI headers. */
#ifdef WITH_MPI
#include <mpi.h>
#endif

/* Local headers. */
#include "align.h"
#include "part_type.h"

/* Pre-declarations */
struct threadpool;

/* Some constants. */
#define part_align 128
#define xpart_align 128
#define spart_align 128
#define gpart_align 128
#define bpart_align 128
#define sink_align 128


/* Import the right hydro particle definition */
#include "hydro_part.h"
/* Import the right gravity particle definition */
#include "gravity_part.h"
/* Import the right start particle definition */
#include "stars_part.h"
/* Import the right black hole particle definition */
#include "black_holes_part.h"
/* Import the right sink particle definition */
#include "sink_part.h"

void part_relink_gparts_to_parts(struct part *parts, const size_t N,
                                 const ptrdiff_t offset);
void part_relink_gparts_to_sparts(struct spart *sparts, const size_t N,
                                  const ptrdiff_t offset);
void part_relink_gparts_to_bparts(struct bpart *bparts, const size_t N,
                                  const ptrdiff_t offset);
void part_relink_gparts_to_sinks(struct sink *sinks, const size_t N,
                                 const ptrdiff_t offset);
void part_relink_parts_to_gparts(struct gpart *gparts, const size_t N,
                                 struct part *parts);
void part_relink_sparts_to_gparts(struct gpart *gparts, const size_t N,
                                  struct spart *sparts);
void part_relink_bparts_to_gparts(struct gpart *gparts, const size_t N,
                                  struct bpart *bparts);
void part_relink_sinks_to_gparts(struct gpart *gparts, const size_t N,
                                 struct sink *sinks);
void part_relink_all_parts_to_gparts(struct gpart *gparts, const size_t N,
                                     struct part *parts, struct sink *sinks,
                                     struct spart *sparts, struct bpart *bparts,
                                     struct threadpool *tp);
void part_verify_links(struct part *parts, struct gpart *gparts,
                       struct sink *sinks, struct spart *sparts,
                       struct bpart *bparts, size_t nr_parts, size_t nr_gparts,
                       size_t nr_sinks, size_t nr_sparts, size_t nr_bparts,
                       int verbose);

#ifdef WITH_MPI
/* MPI data type for the particle transfers */
extern MPI_Datatype part_mpi_type;
extern MPI_Datatype xpart_mpi_type;
extern MPI_Datatype gpart_mpi_type;
extern MPI_Datatype gpart_foreign_mpi_type;
extern MPI_Datatype gpart_fof_foreign_mpi_type;
extern MPI_Datatype spart_mpi_type;
extern MPI_Datatype bpart_mpi_type;
extern MPI_Datatype sink_mpi_type;

void part_create_mpi_types(void);
void part_free_mpi_types(void);
#endif

#endif /* SWIFT_PART_H */
