#!/usr/bin/env bash
# Source this helper; configure/make remain the normal build entry points.
case "${1:-}" in
  serial|cuda-a100) ;;
  *) echo 'Usage: source tools/kokkos/cosma-env.sh {serial|cuda-a100}' >&2; return 2 ;;
esac
module purge || return
module load gnu_comp/14.1.0 openmpi/5.0.3 || return
module load hdf5/1.14.4 fftw/3.3.10 gsl/2.8 parmetis/4.0.3 autoconf/2.71 || return
export CC="$(command -v mpicc)"
export CXX="$(command -v g++)"
export KOKKOS_WORK_ROOT="${KOKKOS_WORK_ROOT:-$HOME/swift-kokkos-work}"
export KOKKOS_SOURCE="$KOKKOS_WORK_ROOT/src/kokkos"
export KOKKOS_PREFIX="$KOKKOS_WORK_ROOT/install/$1/kokkos"
export KOKKOS_BACKEND=serial
if [[ "$1" == cuda-a100 ]]; then
  export CUDA_ROOT=/cosma/local/Nvidia-hpc-sdk/2025_253/Linux_x86_64/25.3/cuda/12.8
  export PATH="$CUDA_ROOT/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export NVCC_WRAPPER_DEFAULT_COMPILER="$CXX"
  export CXX="$KOKKOS_SOURCE/bin/nvcc_wrapper"
  # Link-time driver stubs allow configuration without a visible GPU.
  # Never add this directory to LD_LIBRARY_PATH or a runtime rpath.
  export KOKKOS_LDFLAGS="-L$CUDA_ROOT/lib64/stubs ${KOKKOS_LDFLAGS:-}"
  export KOKKOS_BACKEND=cuda
fi
