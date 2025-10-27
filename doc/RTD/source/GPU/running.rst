.. GPU execution
   Mladen Ivkovic, June 2025

.. _gpu_running_label:

Running SWIFT with GPU acceleration
======================================

To run the executable, run it as you would the non-GPU version. It doesn't need
any extra flags. It does, however, require some additional parameters at
runtime, which are specified in your experiment's ``parameters.yml`` file.
Details below.




Runtime Parameters
------------------------

Mandatory Parameters
~~~~~~~~~~~~~~~~~~~~~~~


To run SWIFT with GPU acceleration, the following mandatory parameters must be
provided in your ``parameters.yml`` file:

.. code-block::yaml

   # Parameters for the task scheduling
   Scheduler:
     gpu_pack_size:               1024
     gpu_bundle_size:             256


* ``gpu_pack_size``: Sets how many leaf cells and pairs of leaf cells of to pack
  for a single offloading cycle. Offloading cycles are repeated until there are
  no more tasks left.

* ``gpu_bundle_size``: Sets how many leaf cells and pairs of leaf cells of
  to pack into a bundle while offloading. ``gpu_pack_size`` total cells will be
  offloaded per cycle, in bundles of size ``gpu_bundle_size``. 



See ``examples/HydroTests/GreshoVortex_3D/greshoGPU256.yml`` for an example.


Optional Parameters
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block::yaml

   # Parameters for the task scheduling
   Scheduler:
     gpu_recursion_max_depth:          0
     gpu_part_buffer_size:            -1


* ``gpu_recursion_max_depth``: Sets the maximal depth we expect to recurse down
  to from super-level tasks to reach leaf cells. We need this to estimate the
  size of buffers to allocate.

* ``gpu_part_buffer_size``: Sets the size of GPU particle buffers. i.e. How many
  particles we expect we will offload per GPU offload .

