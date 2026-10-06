/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2025 Abouzied M. A. Nasar (abouzied.nasar@manchester.ac.uk)
 *                    Mladen Ivkovic (mladen.ivkovic@durham.ac.uk)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#ifndef RUNNER_GPU_PACK_UNIQUE_SORT_FUNCTIONS_H
#define RUNNER_GPU_PACK_UNIQUE_SORT_FUNCTIONS_H


/* Simple hash function for pointers */
__attribute__((always_inline)) INLINE static uintptr_t hash_func(const struct cell *ptr, const uintptr_t hash_size) {
    return ((uintptr_t)ptr) % hash_size;
}

/* Insert into hash table. No need for probing as we will
 * only store one cell in each index of hash table*/
__attribute__((always_inline)) INLINE static void hash_insert(const struct cell *restrict c, const int unique_count, const uintptr_t h_id, struct hash_entry * ht) {
    ht[h_id].c = (struct cell*)c;
    /*This is where the cell will be located in the unique_cells array*/
    ht[h_id].index = unique_count;
    ht[h_id].occupied = 1;
}

/* Lookup in hash table */
__attribute__((always_inline)) INLINE static void hash_lookup_and_pack(const struct cell *restrict c, const int hash_size,
        struct hash_entry *restrict ht, struct gpu_offload_data *restrict buf, const int ij,
        const enum task_subtypes task_subtype) {

  /*Get the hash using the cell's pointer address*/
  struct gpu_pack_metadata *md = &buf->md;
  uintptr_t h_id = hash_func(c, hash_size);
#ifdef SWIFT_DEBUG_CHECKS
  uintptr_t start = h_id;
#endif
  const int n_leaves_packed = md->n_leaves_packed;
  int unique_count = md->n_unique;
  /*Do a linear probe of hash table
   * TODO: If this becomes a large overhead look into
   * optimising the hashing*/
  while(ht[h_id].occupied){
    /*If we already have a cell hashed to h_id.
     * Return it's index in the array of
     * unique cells*/
    if (ht[h_id].c == c){
      /*We found this cell's hash value exists -> Not unique.
       * The hash_lookup returns it's position in the sorted list
       * (not in the hash table)*/
      /*Check if this is ci*/
      if(ij == 0){
        md->my_index[n_leaves_packed].x = ht[h_id].index;
      }
      /*cell is cj*/
      else{
        md->my_index[n_leaves_packed].y = ht[h_id].index;
      }
      return;
    }
    /*Add one to the hash table index and continue linear probing */
    h_id = (h_id + 1) % hash_size;
#ifdef SWIFT_DEBUG_CHECKS
    if(h_id == start)
        error("hash table full");
#endif
  }
#ifdef SWIFT_DEBUG_CHECKS
  if(h_id >= hash_size)
      error("Ran over hash table");
#endif

  /*unique_cells is different from hash table.
   * This is just an array to keep track of
   * unique cells. Used for debugging but no longer necessary
   * TODO: Remove unique_cells if no longer needed*/
  md->unique_cells[unique_count] = (struct cell *)c;
  md->hash_table.count++;
  int c_count = c->hydro.count;
  /*Store where ci starts*/
  md->unique_start_end[unique_count].x = md->count_parts_unique;
  /*Store where ci ends*/
  md->unique_start_end[unique_count].y = md->count_parts_unique + c_count + 1;

  if(ij == 0){ /*This is ci and it is unique*/
    /*This cell has not been found yet.
     * Add to unique_cells and store it's index ascending
     * from index where we last inserted a unique cell*/
    md->my_index[n_leaves_packed].x = unique_count;
    /*Now pack the particles since this cell is unique*/
    if(task_subtype == task_subtype_gpu_density)
      gpu_pack_part_density(c, buf->parts_send_d, md->count_parts_unique);
    else if(task_subtype == task_subtype_gpu_gradient)
      gpu_pack_part_gradient(c, buf->parts_send_g, md->count_parts_unique);
    else if(task_subtype == task_subtype_gpu_force)
      gpu_pack_part_force(c, buf->parts_send_f, md->count_parts_unique);
    /*Add one as we have packed the cells position in index count_parts_unique + cii_count*/
    md->count_parts_unique += c_count + 1;
  }
  else{ /*This is cj and it is unique*/
    /*This cell has not been found yet.
     * Add to unique_cells and store it's index ascending
     * from index where we last inserted a unique cell*/
    md->my_index[n_leaves_packed].y = unique_count;
    /*Now pack the particles since this cell is unique*/
    if(task_subtype == task_subtype_gpu_density)
      gpu_pack_part_density(c, buf->parts_send_d, md->count_parts_unique);
    else if(task_subtype == task_subtype_gpu_gradient)
      gpu_pack_part_gradient(c, buf->parts_send_g, md->count_parts_unique);
    else if(task_subtype == task_subtype_gpu_force)
      gpu_pack_part_force(c, buf->parts_send_f, md->count_parts_unique);
    /*Add one as we have packed the cells position in index count_parts_unique + cii_count*/
    md->count_parts_unique += c_count + 1;
  }

  /*Store pointers for this unique cell, update it's unique index in array of unique cells*/
  hash_insert(c, unique_count, h_id, ht);
  md->n_unique++;

}

__attribute__((always_inline)) INLINE static void pack_cell_particles_in_unique_list(const struct runner *r,
                                      const struct scheduler *s,
                                      struct gpu_offload_data *restrict buf,
                                      const char timer, const struct task * t, const struct cell *restrict cii,
                                      const struct cell *restrict cjj, const enum task_subtypes task_subtype) {

#ifdef SWIFT_DEBUG_CHECKS
  if (cii == NULL) error("Got NULL cell ci?");
  if (cjj == NULL) error("Got NULL cell cj?");
#endif
  /* Grab some handles. */
  /* packing data and metadata */
  struct gpu_pack_metadata *md = &buf->md;
  const struct gpu_md *gpu_md = &buf->gpu_md;
  const int cii_count = cii->hydro.count;
  const int cjj_count = cjj->hydro.count;

#ifdef SWIFT_DEBUG_CHECKS
  /* Anything to do here? */
  if (cii_count == 0 || cjj_count == 0)
    error("Empty cells should've been excluded during the recursion.");
#endif

  /* Get how many particles we've packed until now */
  int pack_ind = md->count_parts_unique;

  int last_ind = pack_ind + cii_count;
  if (cii != cjj) last_ind += cjj_count; /* packing pair interaction */
  if (last_ind >= md->params.part_buffer_size) {
    error(
        "Exceeded particle buffer size. Increase "
        "Scheduler:gpu_part_buffer_size."
        "ind=%d, counts=%d %d, buffer_size=%d, task_subtype=%s, is self "
        "task?=%d",
        pack_ind, cii_count, cjj_count, md->params.part_buffer_size,
        subtaskID_names[task_subtype], cii == cjj);
  }
  /*Figure out where cells start for controlling GPU computations*/
  /*How many blocks have we packed so far?
   * Each cell is split into count/BS chunks so that
   * multiple cuda blocks work on particles in each cell if cell is big enough*/
  const int n_blocks_packed = md->n_blocks_packed;
  /*How many blocks will the current cell be split into*/
  int n_blocks_current;
  if(cii == cjj){
	  n_blocks_current = (cii_count + GPU_THREAD_BLOCK_SIZE - 1)/GPU_THREAD_BLOCK_SIZE;
  }else{/*This is a pair task need to take the max count of ci and cj*/
	  n_blocks_current = (max(cii_count, cjj_count) + GPU_THREAD_BLOCK_SIZE - 1)/GPU_THREAD_BLOCK_SIZE;
  }

  /*Let the CUDA blocks know which parts of the data we send they need to work on*/
  for(int b = 0; b < n_blocks_current; b++){
	  /*Which leaf computation will this block (n_blocks_packed + b) work on?*/
	  gpu_md->block_leaf_id[n_blocks_packed + b].x = md->n_leaves_packed;
	  /*Save the id of the first block acting on this leaf comp.
	   * Needed for indexing in kernel*/
	  gpu_md->block_leaf_id[n_blocks_packed + b].y = n_blocks_packed;
  }

  /*Check to see we've not somehow gone over the number of blocks we allocated*/
  /*TODO: Put in debug checks ifdef. Leave for now while dev'ing*/
  ///////////////////////////////////////////////////////////////////////
  int n_blocks_max = (md->params.part_buffer_size + GPU_THREAD_BLOCK_SIZE - 1)/GPU_THREAD_BLOCK_SIZE;
  md->n_blocks_packed += n_blocks_current;
  if(md->n_blocks_packed > n_blocks_max)
	  error("exceeded n_block_max due to insufficient gpu_part_buffer_size. Increase gpu_part_buffer_size in your *.yml file");
  ///////////////////////////////////////////////////////////////////////

  /*Get a pointer to the full hash table and it's size
   * TODO: Make this a dynamically sized hash table
   * to use load factor to resize so that it is only ever 50% full*/
  struct hash_entry * ht = md->hash_table.entry;
  const int hash_size = md->hash_size;

  /*Check if ci has already been found.
   * If so, return where it's unique copy
   * is found in the hash table
   * Otherwise, add cell to hash table*/
  /*Flag that we're testing ci*/
  int ij = 0;
  hash_lookup_and_pack(cii, hash_size, ht, buf, ij, task_subtype);
  /*Same for cj. For self tasks this will point to ci's location*/
  /*Flag that we're testing cj*/
  ij = 1;
  hash_lookup_and_pack(cjj, hash_size, ht, buf, ij, task_subtype);

  /* Now finish up bookkeeping*/
  /* Update incremented pack length accordingly */
  if (cii == cjj) {
	  /* We packed a self interaction */
	  md->count_parts += cii_count;
  } else {
	  /* We packed a pair interaction */
	  md->count_parts += cii_count + cjj_count;
  }
  /* Record that we have now packed a new leaf cell (pair) & increment number
   * of leaf cells to offload */
  md->n_leaves_packed++;

}

#endif /* RUNNER_GPU_PACK_UNIQUE_SORT_FUNCTIONS_H */
