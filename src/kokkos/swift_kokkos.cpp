/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "swift_kokkos.h"

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>

namespace {
bool owns_runtime = false;
}

extern "C" int swift_kokkos_initialize(void) {
  try {
    if (Kokkos::is_finalized()) {
      std::fprintf(stderr,
                   "Kokkos cannot be initialized after finalization.\n");
      return 1;
    }
    if (!Kokkos::is_initialized()) {
      Kokkos::initialize();
      owns_runtime = true;
    }
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Kokkos initialization failed: %s\n", error.what());
    return 1;
  } catch (...) {
    std::fprintf(stderr, "Kokkos initialization failed.\n");
    return 1;
  }
}

extern "C" void swift_kokkos_finalize(void) {
  try {
    if (owns_runtime && Kokkos::is_initialized() && !Kokkos::is_finalized()) {
      Kokkos::finalize();
      owns_runtime = false;
    }
  } catch (...) {
    /* Never propagate a C++ exception through the C interface/atexit. */
    std::fprintf(stderr, "Kokkos finalization failed.\n");
    std::abort();
  }
}

extern "C" const char *swift_kokkos_backend(void) {
  return Kokkos::DefaultExecutionSpace::name();
}

extern "C" int swift_kokkos_self_test(void) {
  if (!Kokkos::is_initialized() || Kokkos::is_finalized()) {
    std::fprintf(stderr, "Initialize Kokkos before running the check.\n");
    return 1;
  }
  try {
    constexpr int count = 4096;
    Kokkos::View<double *> values("swift_build_check", count);
    Kokkos::parallel_for(
        "swift_build_check_fill", count,
        KOKKOS_LAMBDA(const int i) { values(i) = 2.0 * i + 1.0; });
    double sum = 0.0;
    Kokkos::parallel_reduce(
        "swift_build_check_reduce", count,
        KOKKOS_LAMBDA(const int i, double &partial) { partial += values(i); },
        sum);
    Kokkos::fence();
    auto host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
    for (int i = 0; i < count; ++i) {
      if (host(i) != 2.0 * i + 1.0) {
        std::fprintf(stderr, "Kokkos check failed at element %d.\n", i);
        return 1;
      }
    }
    const double expected = static_cast<double>(count) * count;
    std::printf("Execution space: %s\n", swift_kokkos_backend());
    Kokkos::print_configuration(std::cout, true);
    std::printf("Checksum: %.0f; expected: %.0f\n", sum, expected);
    if (sum != expected) return 1;
    std::puts(
        "PASS: SWIFT C entry point -> Kokkos kernel -> host verification");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Kokkos check failed: %s\n", error.what());
    return 1;
  } catch (...) {
    std::fprintf(stderr, "Kokkos check failed.\n");
    return 1;
  }
}
