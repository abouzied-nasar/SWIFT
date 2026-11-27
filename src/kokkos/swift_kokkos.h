/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef SWIFT_KOKKOS_H
#define SWIFT_KOKKOS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Single-process prototype. Call from the main thread only. */
int swift_kokkos_initialize(void);
void swift_kokkos_finalize(void);
const char *swift_kokkos_backend(void);
/* Requires initialization; launches and verifies a kernel on the backend. */
int swift_kokkos_self_test(void);

#ifdef __cplusplus
}
#endif

#endif
