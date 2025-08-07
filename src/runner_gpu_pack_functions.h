#include "../config.h"
#include "runner.h"
#include "timeline.h"

/* Temporary warning during dev works. */
#if !(defined(HAVE_CUDA) || defined(HAVE_HIP))
#pragma warning "Don't have CUDA nor HIP"
#endif

#ifdef WITH_CUDA
#include "cuda/GPU_offload_data.h"
#include "cuda/GPU_part_structs.h"
#endif

#ifdef WITH_HIP
#pragma message "YES"
#include "hip/GPU_part_structs.h"
#endif

void gpu_pack_self_density_cell(
    const struct cell* restrict c,
    struct gpu_offload_data *buf);
void gpu_pack_self_gradient_cell(
    const struct cell* restrict c,
    struct gpu_offload_data *buf);
void gpu_pack_self_force_cell(
    const struct cell* restrict c,
    struct gpu_offload_data *buf);
void gpu_unpack_self_density_cell(struct cell* restrict c,
    const struct part_aos_f4_recv_d* restrict parts_aos_buffer,
    const int tid, const size_t pack_position,
    const size_t count, const struct engine *e);
void gpu_unpack_self_gradient_cell(struct cell* restrict c,
    const struct part_aos_f4_recv_g* restrict parts_aos_buffer,
    const int tid, const size_t pack_position,
    const size_t count, const struct engine *e);
void gpu_unpack_self_force_cell(struct cell* restrict c,
    const struct part_aos_f4_recv_f* restrict parts_aos_buffer,
    const int tid, const size_t pack_position,
    const size_t count, const struct engine *e);

void gpu_pack_pair_density_cell(
    const struct cell *c, struct part_aos_f4_send_d *parts_aos_buffer,
    const int tid, const int local_pack_position,
    const int count, const double3 shift, const int2 cstarts);
void gpu_pack_pair_density(struct gpu_offload_data* buf,
    const struct runner *r, const struct cell *ci, const struct cell *cj,
    const double3 shift_tmp, const int tid);



/* ------------------------------------------ */
void runner_doself1_gpu_unpack_neat_aos_f4(
    const struct runner *r, struct cell *c,
    struct part_aos_f4_recv_d *parts_aos_buffer, int timer, size_t *pack_length, int tid,
    int count_max_parts_tmp,
    const struct engine *e);

void runner_doself1_gpu_unpack_neat_aos_f4_g(
    const struct runner *r, struct cell *c,
    struct part_aos_f4_recv_g *parts_aos_buffer, int timer, size_t *pack_length,
    int tid, int count_max_parts_tmp, const struct engine *e);

void runner_doself1_gpu_unpack_neat_aos_f4_f(
    const struct runner *r, struct cell *restrict c,
    struct part_aos_f4_recv_f *restrict parts_aos_buffer, int timer,
    size_t *pack_length, int tid, int count_max_parts_tmp, const struct engine *e);

void runner_do_ci_cj_gpu_unpack_neat_aos_f4(
    struct runner *r, struct cell *ci, struct cell *cj,
    struct part_aos_f4_recv_d *parts_aos_buffer, int timer, size_t *pack_length,
    int tid, int count_max_parts_tmp, const struct engine *e);

void runner_do_ci_cj_gpu_unpack_neat_aos_f4_g(
    struct runner *r, struct cell *ci, struct cell *cj,
    struct part_aos_f4_recv_g *parts_aos_buffer, int timer, size_t *pack_length,
    int tid, int count_max_parts_tmp, const struct engine *e);

void runner_do_ci_cj_gpu_unpack_neat_aos_f4_f(
    struct runner *r, struct cell *ci, struct cell *cj,
    struct part_aos_f4_recv_f *parts_aos_buffer, int timer, size_t *pack_length,
    int tid, int count_max_parts_tmp, const struct engine *e);

void runner_do_ci_cj_gpu_pack_neat_aos_f4(
    struct runner *r, struct cell *restrict ci, struct cell *restrict cj,
    struct part_aos_f4_send_d *restrict parts_aos_buffer, int timer,
    size_t *pack_length, int tid, int count_max_parts_tmp, const int count_ci,
    const int count_cj, double3 shift_tmp);

void runner_do_ci_cj_gpu_pack_neat_aos_f4_g(
    struct runner *r, struct cell *restrict ci, struct cell *restrict cj,
    struct part_aos_f4_send_g *restrict parts_aos_buffer, int timer,
    size_t *pack_length, int tid, int count_max_parts_tmp, const int count_ci,
    const int count_cj, double3 shift_tmp);

void runner_do_ci_cj_gpu_pack_neat_aos_f4_f(
    struct runner *r, struct cell *restrict ci, struct cell *restrict cj,
    struct part_aos_f4_send_f *restrict parts_aos_buffer, int timer,
    size_t *pack_length, int tid, int count_max_parts_tmp, const int count_ci,
    const int count_cj, double3 shift_tmp);




/* TODO MLADEN: CAN THIS BE DELETED? */
/* void runner_doself1_gpu_pack_forc_aos(struct runner *r, struct cell *c, */
/*                                       struct part_aos *parts_aos, int timer, */
/*                                       int *pack_length, int tid, */
/*                                       int count_max_parts_tmp); */
/* void runner_doself1_gpu_pack_grad_aos(struct runner *r, struct cell *c, */
/*                                       struct part_aos *parts_aos, int timer, */
/*                                       int *pack_length, int tid, */
/*                                       int count_max_parts_tmp); */
/* void runner_doself1_gpu_unpack_neat(struct runner *r, struct cell *c, */
/*                                     struct part_soa parts_soa, int timer, */
/*                                     int *pack_length, int tid, */
/*                                     int count_max_parts_tmp, struct engine *e); */
/* void runner_doself1_gpu_unpack_neat_aos(struct runner *r, struct cell *c, */
/*                                         struct part_aos *parts_aos_buffer, */
/*                                         int timer, int *pack_length, int tid, */
/*                                         int count_max_parts_tmp, */
/*                                         struct engine *e); */
/* void runner_doself1_gpu_unpack_neat_aos_g(struct runner *r, struct cell *c, */
/*                                           struct part_aos_g *parts_aos_buffer, */
/*                                           int timer, int *pack_length, int tid, */
/*                                           int count_max_parts_tmp, */
/*                                           struct engine *e); */
/* void runner_doself1_gpu_unpack_neat_aos_f(struct runner *r, struct cell *c, */
/*                                           struct part_aos_f *parts_aos_buffer, */
/*                                           int timer, int *pack_length, int tid, */
/*                                           int count_max_parts_tmp, */
/*                                           struct engine *e); */
/* void pack(struct cell *c, double *x_p, double *y_p, double *z_p, int tid, */
/*           int *tid_p, long long *id, float *ux, float *uy, float *uz, */
/*           float *a_hydrox, float *a_hydroy, float *a_hydroz, float *mass, */
/*           float *h, float *u, float *u_dt, float *rho, float *SPH_sum, */
/*           float *locx, float *locy, float *locz, float *widthx, float *widthy, */
/*           float *widthz, float *h_max, int *count_p, float *wcount, */
/*           float *wcount_dh, float *rho_dh, float *rot_u, float *rot_v, */
/*           float *rot_w, float *div_v, float *div_v_previous_step, */
/*           float *alpha_visc, float *v_sig, float *laplace_u, float *alpha_diff, */
/*           float *f, float *soundspeed, float *h_dt, float *balsara, */
/*           float *pressure, float *alpha_visc_max_ngb, timebin_t *time_bin, */
/*           timebin_t *wakeup, timebin_t *min_ngb_time_bin, */
/*           char *to_be_synchronized, int local_pack_position, int count); */
/* void pack_neat(struct cell *c, struct part_soa parts_soa, int tid, */
/*                int local_pack_position, int count); */
/* void pack_neat_aos(struct cell *c, struct part_aos *parts_aos_buffer, int tid, */
/*                    int local_pack_position, int count); */
/* void pack_neat_aos_g(struct cell *c, struct part_aos_g *parts_aos_buffer, */
/*                      int tid, int local_pack_position, int count); */
/* void pack_neat_aos_f(struct cell *c, struct part_aos_f *parts_aos, int tid, */
/*                      int local_pack_position, int count); */
/* void pack_neat_aos_f4(struct cell *c, struct part_aos_f4_send *parts_aos_buffer, */
/*                       int tid, int local_pack_position, int count, */
/*                       int2 frst_lst_prts); */
/* void pack_neat_aos_f4_g(struct cell *c, */
/*                         struct part_aos_f4_g_send *parts_aos_buffer, int tid, */
/*                         int local_pack_position, int count); */
/* void pack_neat_aos_f4_f(const struct cell *restrict c, */
/*                         struct part_aos_f4_f_send *restrict parts_aos, int tid, */
/*                         int local_pack_position, int count); */
/* void unpack_neat(struct cell *c, struct part_soa parts_soa_buffer, int tid, */
/*                  int local_pack_position, int count, struct engine *e); */
/* void unpack_neat_aos(struct cell *c, struct part_aos *parts_aos_buffer, int tid, */
/*                      int local_pack_position, int count, struct engine *e); */
/* void unpack_neat_aos_f4(struct cell *c, */
/*                         struct part_aos_f4_recv *parts_aos_buffer, int tid, */
/*                         int local_pack_position, int count, struct engine *e); */
/* void unpack_neat_aos_g(struct cell *c, struct part_aos_g *parts_aos_buffer, */
/*                        int tid, int local_pack_position, int count, */
/*                        struct engine *e); */
/* void unpack_neat_aos_f4_g(struct cell *c, */
/*                           struct part_aos_f4_g_recv *parts_aos_buffer, int tid, */
/*                           int local_pack_position, int count, struct engine *e); */
/* void unpack_neat_aos_f(struct cell *c, struct part_aos_f *parts_aos_buffer, */
/*                        int tid, int local_pack_position, int count, */
/*                        struct engine *e); */
/* void unpack_neat_aos_f4_f(struct cell *restrict c, */
/*                           struct part_aos_f4_f_recv *restrict parts_aos_buffer, */
/*                           int tid, int local_pack_position, int count, */
/*                           struct engine *e); */
/* void unpack(struct cell *c, double *x_p, double *y_p, double *z_p, int tid, */
/*             int *tid_p, long long *id, float *ux, float *uy, float *uz, */
/*             float *a_hydrox, float *a_hydroy, float *a_hydroz, float *mass, */
/*             float *h, float *u, float *u_dt, float *rho, float *SPH_sum, */
/*             float *locx, float *locy, float *locz, float *widthx, float *widthy, */
/*             float *widthz, float *h_max, int *count_p, float *wcount, */
/*             float *wcount_dh, float *rho_dh, float *rot_u, float *rot_v, */
/*             float *rot_w, float *div_v, float *div_v_previous_step, */
/*             float *alpha_visc, float *v_sig, float *laplace_u, */
/*             float *alpha_diff, float *f, float *soundspeed, float *h_dt, */
/*             float *balsara, float *pressure, float *alpha_visc_max_ngb, */
/*             timebin_t *time_bin, timebin_t *wakeup, timebin_t *min_ngb_time_bin, */
/*             char *to_be_synchronized, int local_pack_position, int count, */
/*             struct engine *e); */
/* void runner_doself1_gpu_unpack( */
/*     struct runner *r, struct cell *c, int timer, int *pack_length, double *x_p, */
/*     double *y_p, double *z_p, int tid, int *tid_p, long long *id, float *ux, */
/*     float *uy, float *uz, float *a_hydrox, float *a_hydroy, float *a_hydroz, */
/*     float *mass, float *h, float *u, float *u_dt, float *rho, float *SPH_sum, */
/*     float *locx, float *locy, float *locz, float *widthx, float *widthy, */
/*     float *widthz, float *h_max, int *count_p, float *wcount, float *wcount_dh, */
/*     float *rho_dh, float *rot_u, float *rot_v, float *rot_w, float *div_v, */
/*     float *div_v_previous_step, float *alpha_visc, float *v_sig, */
/*     float *laplace_u, float *alpha_diff, float *f, float *soundspeed, */
/*     float *h_dt, float *balsara, float *pressure, float *alpha_visc_max_ngb, */
/*     timebin_t *time_bin, timebin_t *wakeup, timebin_t *min_ngb_time_bin, */
/*     char *to_be_synchronized, int count_max_parts_tmp, struct engine *e); */
/*  */
/* void runner_do_ci_cj_gpu_pack_neat(struct runner *r, struct cell *ci, */
/*                                    struct cell *cj, */
/*                                    struct part_soa parts_soa_buffer, int timer, */
/*                                    int *pack_length, int tid, */
/*                                    int count_max_parts_tmp, int count_ci, */
/*                                    int count_cj); */
/*  */
/* void runner_do_ci_cj_gpu_pack_neat_aos(struct runner *r, struct cell *ci, */
/*                                        struct cell *cj, */
/*                                        struct part_aos *parts_aos_buffer, */
/*                                        int timer, int *pack_length, int tid, */
/*                                        int count_max_parts_tmp, int count_ci, */
/*                                        int count_cj, float3 shift_tmp); */
/*  */
/*  */
/* void runner_do_ci_cj_gpu_pack_neat_aos_g(struct runner *r, struct cell *ci, */
/*                                          struct cell *cj, */
/*                                          struct part_aos_g *parts_aos_buffer, */
/*                                          int timer, int *pack_length, int tid, */
/*                                          int count_max_parts_tmp, int count_ci, */
/*                                          int count_cj); */
/*  */
/*  */
/* void runner_do_ci_cj_gpu_pack_neat_aos_f(struct runner *r, struct cell *ci, */
/*                                          struct cell *cj, */
/*                                          struct part_aos_f *parts_aos_buffer, */
/*                                          int timer, int *pack_length, int tid, */
/*                                          int count_max_parts_tmp, int count_ci, */
/*                                          int count_cj); */
/*  */
/*  */
/* void runner_do_ci_cj_gpu_unpack_neat(struct runner *r, struct cell *ci, */
/*                                      struct cell *cj, */
/*                                      struct part_soa parts_soa_buffer, */
/*                                      int timer, int *pack_length, int tid, */
/*                                      int count_max_parts_tmp, struct engine *e); */
/*  */
/* void runner_do_ci_cj_gpu_unpack_neat_aos(struct runner *r, struct cell *ci, */
/*                                          struct cell *cj, */
/*                                          struct part_aos *parts_aos_buffer, */
/*                                          int timer, int *pack_length, int tid, */
/*                                          int count_max_parts_tmp, */
/*                                          struct engine *e); */
/*  */
/*  */
/*  */
/* void runner_do_ci_cj_gpu_unpack_neat_aos_g(struct runner *r, struct cell *ci, */
/*                                            struct cell *cj, */
/*                                            struct part_aos_g *parts_aos_buffer, */
/*                                            int timer, int *pack_length, int tid, */
/*                                            int count_max_parts_tmp, */
/*                                            struct engine *e); */
/*  */
/* void runner_do_ci_cj_gpu_unpack_neat_aos_f(struct runner *r, struct cell *ci, */
/*                                            struct cell *cj, */
/*                                            struct part_aos_f *parts_aos_buffer, */
/*                                            int timer, int *pack_length, int tid, */
/*                                            int count_max_parts_tmp, */
/*                                            struct engine *e); */
/*  */
