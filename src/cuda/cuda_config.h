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

#ifndef CUDA_CONFIG_H
#define CUDA_CONFIG_H

/**
 * @file src/cuda/cuda_config.h
 * @brief Temporary file to store all macro definitions relating to
 * configuring the cuda setup/run until we sort everything out cleanly.
 */

#define GPU_THREAD_BLOCK_SIZE 64

#define _N_TASKS_PER_PACK_SELF 8
#define _N_TASKS_BUNDLE_SELF 2

#define _N_TASKS_PER_PACK_PAIR 8
#define _N_TASKS_BUNDLE_PAIR 2

/* Config parameters. */
/* TODO: DO WE STILL NEED THESE??? */
#define GPUOFFLOAD_DENSITY 1  /* off-load hydro density to GPU */
#define GPUOFFLOAD_GRADIENT 1 /* off-load hydro gradient to GPU */
#define GPUOFFLOAD_FORCE 1    /* off-load hydro force to GPU */

#endif
