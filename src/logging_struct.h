#pragma once

/* Struct to hold data for binary output dump. */

#include <stdio.h>

#include "part.h"

struct logging_entry {

  /*! subtype: d (density), g (gradient), or f (force) */
  char subtype;

  /*! pack_or_unpack: p for packing operation, u for unpacking */
  char pack_or_unpack;

  /*! offset of c->hydro.parts array in global parts: space->parts  */
  long offset;

  /*! nr of particles in the cell */
  int count;

  /*! timing of operation */
  double time;
};


struct logging_data {

  volatile int count;

  struct logging_entry* entries;

  struct part* all_parts;
};

