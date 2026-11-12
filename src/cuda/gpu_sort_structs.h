#ifndef CUDA_GPU_SORT_STRUCTS_H
#define CUDA_GPU_SORT_STRUCTS_H

/*! Sort entry used by GPU pair interactions */
struct gpu_sort_entry {
  /* Particle position projected along the sort direction */
  float d;
  /* Absolute index of the particle in the packed GPU particle buffer
   * In CPU sorted code this represents local index within a cell (I think!)*/
  int i;
};

/* Location of the sorted particle lists for a leaf computation */
struct gpu_leaf_sort_metadata {
  /* Index where ci's sorted particle list starts in sort_entries */
  int ci_sort_start;
  /* Number of particles in ci's sorted particle list */
  int ci_sort_count;
  /* Index where cj's sorted particle list starts in sort_entries */
  int cj_sort_start;
  /* Number of particles in cj's sorted particle list */
  int cj_sort_count;
  /* First sorted particle in ci which can potentially interact with cj */
  int ci_first;
  /* Last sorted particle in cj which can potentially interact with ci */
  int cj_last;
  /* Maximum combined displacement since the two cells were sorted */
  float dx_max;
  /* Periodic shift projected onto the pair sort direction */
  float rshift;
};

#endif
