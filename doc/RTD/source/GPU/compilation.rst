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
- Similarly, ``$CUDA_LDFLAGS`` and ``$CUDA_LIBS`` can be used to pass library
  flags related to the linking with cuda to the host compiler and linker.
- ``$NVCC_FLAGS``: Flags to be passed on to the ``nvcc`` compiler for compiling
  device code.
- Similarly, ``$NVCC_LDFLAGS`` and ``$NVCC_LIBS`` can be used to pass library
  flags to the device linker.

Example usage (note the line break escape characters ``\\``):

.. code-block:: bash

   CUDA_CFLAGS="-I/path/to/cuda/include" \ 
   NVCC_FLAGS=" -allow-unsupported-compiler -diag-suppress 177 -diag-suppress 550" \
   ./configure --with-cuda=/path/to/your/cuda/installation

Similar to how ``--enable-debug`` enables debug symbols in the executable for
the CPU code, the ``--enable-gpu-debug`` flag will enable debug symbols for cuda
(host and device) code.

To disable optimization (on both host and device code, simultaneously), use the
``--disable-optimization`` flag.

If you want to compile device code for a specific GPU hardware architecture, use
the ``--with-cuda-arch=XX`` flag. The default is ``native``, which you can use
if you compile on a system where the GPU you want to use for your runs is
available. Check the `CUDA Documentation <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list>`_ 
for options, e.g. ``sm_90``.



Adding new files
----------------

The build system doesn't play perfectly nice with ``nvcc`` in combination with
other compilers. We found the following solution:

- Background: In a CPU only build, the build system creates two main convenience
  libraries, ``libswiftsim.la`` (without MPI) and ``libswisim_mpi.la`` (with
  MPI), which are ultimately linked against the 'main' object to create the
  execuables. (There are also 2 more convenience libraries for gravity, but
  let's ignore that here.) 
- We build two additional convenience libraries, ``libswiftsim_cuda.la`` and
  ``libswiftsim_mpicuda.la`` which contain **host** code necessary to run with
  GPUs.
- We build a third convenience library, ``src/libswift_cuda_device.a``, which 
  contains all **device** code.

The third library was necessary because ``nvcc`` doesn't play too nice with
libtool and needs an extra linking step to create device code linkable with the
host compiler.

So if you intend on adding new files, please follow this convention:

- Files containing **host code** are treated the same as C code:
  - Base files should be ``.c`` files. Add them to the ``GPU_CUDA_SOURCES``
    variable in ``src/Makefile.am``.
  - Corresponding header files should be ``.h`` files. Add them to the
    ``include_HEADERS`` variable.
- Files containing **device code**:
  - should be ``.cu`` files. Add them to the ``AM_CUDA_DEVICE_SOURCES``,
    ``AM_CUDA_DEVICE_OBJECTS``, and ``AM_CUDA_DEVICE_DLINK_OBJECTS`` variables
    in ``src/Makefile.am``.
  - Corresponding header files should be ``.h`` files. Add them to the
    ``include_HEADERS`` variable.
- Headers without a corresponding base file (whether ``.c`` or ``.cu``):
  - Add them to the ``nobase_noinst_HEADERS`` variable.
  - Make sure their inclusion into code is guarded by appropriate macros.
    Otherwise, you will destroy the build system.


A note on macros
^^^^^^^^^^^^^^^^

For cuda, we mainly use two vaguely related macros:

- ``HAVE_CUDA``:
   This is set by the ``autoconf`` configuration and signifies whether cuda was
   found on your system. If available, it will be defined in ``config.h``.

- ``WITH_CUDA``:
  This is used to include or exclude code when compiling with or without cuda.
  Internally, it is passed as a flag to the compiler. Remember that we still
  want to be able to compile SWIFT without GPU support, regardless of whether
  CUDA is available or not. So hide code behind this macro which should only be
  compiled if we're compiling to create the cuda convenience libraries and
  executables.









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



