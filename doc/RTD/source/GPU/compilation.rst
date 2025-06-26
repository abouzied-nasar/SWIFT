.. GPU compilation
   Mladen Ivkovic, June 2025

.. _gpu_compilation_label:

Compiling SWIFT with GPU acceleration
=========================================


.. warning:: 
   This documentation holds only for the hydrodynamics GPU offloading.
   This is still in heavy development and subject to rapid changes.



CUDA
~~~~~~~

Configuration
----------------

To enable GPU acceleration via the ``cuda`` implementation, configure with the
``--with-cuda`` flag:

.. code-block:: bash

   ./configure --with-cuda

or, if your ``cuda`` installation isn't found, provide the full path:

.. code-block:: bash

   ./configure --with-cuda=/path/to/your/cuda/installation



If you want to pass on flags to the compilers, you can do so by setting variables:

- ``$CUDA_CFLAGS``: Cuda-related flags for host (C) code, passed on to the C
  compiler. Use to e.g. manually provide include path
  (``CUDA_CFLAGS=-I/path/to/cuda/include``)
- ``$NVCC_FLAGS``: Flags to be passed on to the ``nvcc`` compiler.

Example usage (note the line break escape characters '\'):

.. code-block:: bash

   CUDA_CFLAGS="-I/path/to/cuda/include" \ 
   NVCC_FLAGS=" -allow-unsupported-compiler -diag-suppress 177 -diag-suppress 550" \
   ./configure --with-cuda=/path/to/your/cuda/installation



.. warning::
   We haven't implemented automatic architecture detection yet. Currently, it's
   hardcoded in ``src/cuda/Makefile.am`` as ``CUDA_MYFLAGS += -arch=sm_XX``.
   Make sure to modify that depending on the hardware you're running. See e.g.
   `CUDA Documentation <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list>`_ for options.




On Cosma
---------------

The following set of modules work:

.. code-block:: 

  Currently Loaded Modulefiles:
   1) gnu_comp/13.1.0   
   2) hdf5/1.12.2       
   3) nvhpc-byo-compiler/24.5   
   4) autoconf/2.71             
   5) openmpi/4.1.5(default)   
   6) fftw/3.3.10(default)     
   7) parmetis/4.0.3(default)   
   8) ucx/1.17.0(default)      
   9) metis/5.1.0-64bit  
   10) jemalloc/5.1.0 


Then configure with

.. code-block:: bash

  CC=mpicc CXX=mpic++ ./configure --enable-compiler-warnings --with-cuda

To use sanitizer configure with

.. code-block:: bash

   ./configure --enable-compiler-warnings --with-cuda LDFLAGS="-fsanitize=address" --enable-sanitizer 

N.B CUDA and ASAN don't like each other so WHEN RUNNING CODE COMPILED WITH SANITIZER YOU MUST USE

.. code-block:: bash

   ASAN_OPTIONS=protect_shadow_gap=0 ../../../swift_cuda {runtime args here}



HIP
~~~~~~~

TODO.


If you want to pass on flags to the compilers, you can do so by setting variables:

- ``$HIP_CFLAGS``: Cuda-related flags for host (C) code, passed on to the C
  compiler. Use to e.g. manually provide include path
  (``HIP_CFLAGS=-I/path/to/hip/include``)
- ``$HIPCC_FLAGS``: Flags to be passed on to the ``hipcc`` compiler.



