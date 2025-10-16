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
     gpu_self_pack_size:               0
     gpu_self_bundle_size:             0
     gpu_pair_pack_size:               0
     gpu_pair_bundle_size:             0


* ``gpu_self_pack_size``: Sets how many leaf cells and pairs of leaf cells of
  ``self`` tasks to pack for offloading. 

* ``gpu_self_bundle_size``: Sets how many leaf cells and pairs of leaf cells of
  ``self`` tasks to pack into a bundle while offloading. ``gpu_self_pack_size``
  total cells will be offloaded per cycle, in bundles of size
  ``gpu_self_bundle_size``. 

* ``gpu_pair_pack_size``:  Sets how many pairs of leaf cells of ``pair`` tasks
  to pack for offloading. 

* ``gpu_pair_bundle_size``: Sets how many leaf cell pairs of ``pair`` tasks to
  pack into a bundle while offloading. ``gpu_pair_pack_size`` total cells will
  be offloaded per cycle, in bundles of size ``gpu_pair_bundle_size``.


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

* ``gpu_part_buffer_size``: Sets the size of GPU particle array buffers.

