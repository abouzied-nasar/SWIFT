#ifndef CUDA_GPU_SORT_STRUCTS_H
#define CUDA_GPU_SORT_STRUCTS_H

/*! Sort entry used by GPU pair interactions */
struct gpu_sort_entry {
  /* Particle position projected along the sort direction */
  float d;
  /* Absolute index of the particle in the packed GPU particle buffer */
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
  /* Conservative interaction reach */
  float max_reach;
};

/* Candidate density target ranges for one pair leaf */
struct gpu_sorted_pair_range {
  /* First candidate position in ci's sorted list */
  int ci_first;

  /* Last candidate position in cj's sorted list */
  int cj_last;

  /* Number of candidate target particles in ci */
  int ci_target_count;

  /* Number of candidate target particles in cj */
  int cj_target_count;

  /* Maximum interaction reach of particles in cell i */
  float ci_reach;
  /* Maximum interaction reach of particles in cell j */
  float cj_reach;
};

#endif
