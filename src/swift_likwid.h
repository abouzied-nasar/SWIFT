#ifndef SWIFT_LIKWID_H
#define SWIFT_LIKWID_H


#include "inline.h"

#define LIKWID_PERFMON

#ifdef LIKWID_PERFMON
#include "likwid-marker.h"
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif


#ifdef LIKWID_PERFMON

/* Sum up all self/pair measurements, or do it separately? */

/* #define SWIFT_LIKWID_SUM_MEASUREMENT */

#define SWIFT_LIKWID_SELF_PAIR_MEASUREMENT

#endif

__attribute__((always_inline)) INLINE void swift_init_likwid(void){
  /* Needs to be in serial region! */
  LIKWID_MARKER_INIT;
}


static __attribute__((always_inline)) INLINE void swift_init_likwid_markers(void){
  /* Do this in parallel region! */

#ifdef SWIFT_LIKWID_SUM_MEASUREMENT
  LIKWID_MARKER_REGISTER("pack_density");
  LIKWID_MARKER_REGISTER("pack_grad");
  LIKWID_MARKER_REGISTER("pack_force");
#elif defined SWIFT_LIKWID_SELF_PAIR_MEASUREMENT
  LIKWID_MARKER_REGISTER("pack_density_self");
  LIKWID_MARKER_REGISTER("pack_density_pair");
  LIKWID_MARKER_REGISTER("pack_grad_self");
  LIKWID_MARKER_REGISTER("pack_grad_pair");
  LIKWID_MARKER_REGISTER("pack_force_self");
  LIKWID_MARKER_REGISTER("pack_force_pair");
#elif defined LIKWID_PERFMON
#pragma error "Invalid macro defined"
#endif
}


static __attribute__((always_inline)) INLINE void swift_close_likwid(void){
  LIKWID_MARKER_CLOSE;
}



#endif
