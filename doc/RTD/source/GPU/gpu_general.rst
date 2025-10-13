.. GPU execution
   Mladen Ivkovic, June 2025

.. _gpu_general_label:

General info on SWIFT with GPU acceleration
=================================================


General Notes
-------------------

- Currently, we're focussing our efforts on the development with CUDA. Despite
  some HIP files existing, they do not work and will likely not even compile.
  They are still a work in progress.

- Currently, only hydrodynamics tasks for the SPHENIX SPH scheme are
  accelerated.



How It Works
-------------------

To make sense of what the following sections on how to compile and run SWIFT
with GPU acceleration will require you to do, it can help to get a general
picture of how things work internally first.

- Internally, SWIFT uses the a task-based parallelisation strategy: Particles
  are sorted (recursively) into (a tree of) cells. 
- The equations we solve require us to interact particles with other,
  neighbouring particles.
- Typically, the interactions are executed between leaf cells of the particle
  spacetree, i.e. on the lowest leve.
- At some level of that tree, we assign a "workload", i.e. all the physics
  equations we need solved, to that cell. This combination of "workload" and
  "data" (= cell's data) is referred to as a "task".
- We differentiate between two types of tasks which interact particles with
  other particles:
  - A ``self`` type task interacts particles of a cell with other particles
    within the same cell.
  - A ``pair`` type task interacts particles of a cell with other particles of a
    different cell.
- When running the CPU-only SWIFT, tasks are executed in parallel on the CPU.
  The correct order of operations and data races are handled internally.
- When running GPU-SWIFT, the tasks are too small for efficient offloading.
  Instead, we pack them together into a bigger package (which we call a
  ``pack``) and ship that bigger package to the GPU.
- For efficiency, while transferring and solving the ``pack`` on the GPU, we
  do that by splitting it into a few smaller packages (which are still bigger
  than a single leaf-cell leaf-cell interaction). We call these smaller packages
  ``bundle`` s. Hence, a ``pack`` consists of several ``bundle`` s.


