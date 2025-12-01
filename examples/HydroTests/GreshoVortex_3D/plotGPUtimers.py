#!/usr/bin/env python3

import matplotlib

matplotlib.use("Agg")

import argparse
import os
import numpy as np
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser(
    prog="plotGPUtimers.py",
    description="plot the outputs of the timers.",
    epilog="To get timers output with swift, first configure with `./configure --enable-timers` and then run swift with `./swift --timers`",
)

parser.add_argument(
    "timer_file",
    nargs="?",
    action="store",
    default="timers_0.txt",
    help="file to read in. Default: 'timers_0.txt'",
    type=str,
)

parser.add_argument(
    "-n",
    "--nthreads",
    action="store",
    default=1,
    type=int,
    help="normalise timers assuming N threads",
)
parser.add_argument(
    "-s",
    "--seconds",
    action="store_true",
    help="use seconds as units, not milliseconds",
)

parser.add_argument(
    "-z",
    "--include-step-zero",
    action="store_true",
    help="Include the zeroth step in the timing averages",
)

args = parser.parse_args()
nthreads = args.nthreads
units = "ms"
if args.seconds:
    units = "s"

if not os.path.exists(args.timer_file):
    print(f"Couldn't find timer file {args.timer_file}.")
    exit(1)

# Which data do we include in the averages?
first_index = 1
if args.include_step_zero:
    first_index = 0

timer_names = [
    # 0: step |
    "none",
    "prepare",
    "init",
    "init_grav",
    "drift_part",
    "drift_gpart",
    "drift_spart",
    "drift_bpart",
    "kick1",
    "kick2",
    "timestep",
    "end_hydro_force",
    "end_grav_force",
    "dosort",
    "doself_density",
    "doself_gradient",
    "doself_force",
    "doself_limiter",
    "doself_stars_density",
    "doself_stars_feedback",
    "doself_bh_density",
    "doself_bh_swallow",
    "doself_bh_feedback",
    "doself_grav_pp",
    "doself_sinks_density",
    "doself_sinks_swallow",
    "dopair_density",
    "dopair_gradient",
    "dopair_force",
    "dopair_limiter",
    "dopair_stars_density",
    "dopair_stars_feedback",
    "dopair_bh_density",
    "dopair_bh_swallow",
    "dopair_bh_feedback",
    "dopair_grav_mm",
    "dopair_grav_pp",
    "dopair_sinks_density",
    "dopair_sinks_swallow",
    "dograv_external",
    "dograv_down",
    "dograv_mesh",
    "dograv_top_level",
    "dograv_long_range",
    "dosub_self_density",
    "dosub_self_gradient",
    "dosub_self_force",
    "dosub_self_limiter",
    "dosub_self_stars_density",
    "dosub_self_stars_feedback",
    "dosub_self_bh_density",
    "dosub_self_bh_swallow",
    "dosub_self_bh_feedback",
    "dosub_self_grav",
    "dosub_self_sinks_density",
    "dosub_self_sinks_swallow",
    "dosub_pair_density",
    "dosub_pair_gradient",
    "dosub_pair_force",
    "dosub_pair_limiter",
    "dosub_pair_stars_density",
    "dosub_pair_stars_feedback",
    "dosub_pair_bh_density",
    "dosub_pair_bh_swallow",
    "dosub_pair_bh_feedback",
    "dosub_pair_grav",
    "dosub_pair_sinks_density",
    "dosub_pair_sinks_swallow",
    "doself_subset",
    "dopair_subset",
    "dopair_subset_naive",
    "dosub_subset",
    "do_ghost",
    "do_extra_ghost",
    "do_stars_ghost",
    "do_black_holes_ghost",
    "do_sinks_ghost",
    "dorecv_part",
    "dorecv_gpart",
    "dorecv_spart",
    "dorecv_bpart",
    "do_limiter",
    "do_cooling",
    "do_star_formation",
    "do_star_evol",
    "gettask",
    "qget",
    "qsteal",
    "locktree",
    "runners",
    "step",
    "csds",
    "do_stars_sort",
    "do_stars_resort",
    "fof_self",
    "fof_pair",
    "drift_sink",
    "rt_ghost1",
    "rt_ghost2",
    "doself_rt_gradient",
    "dopair_rt_gradient",
    "dosub_self_rt_gradient",
    "dosub_pair_rt_gradient",
    "doself_rt_transport",
    "dopair_rt_transport",
    "dosub_self_rt_transport",
    "dosub_pair_rt_transport",
    "rt_tchem",
    "rt_advance_cell_time",
    "rt_collect_times",
    "do_sync",
    "neutrino_weighting",
    "gpu_self_pack_density",
    "gpu_self_pack_gradient",
    "gpu_self_pack_force",
    "gpu_self_unpack_density",
    "gpu_self_unpack_gradient",
    "gpu_self_unpack_force",
    "gpu_self_launch_density",
    "gpu_self_launch_gradient",
    "gpu_self_launch_force",
    "gpu_pair_pack_density",
    "gpu_pair_pack_gradient",
    "gpu_pair_pack_force",
    "gpu_pair_unpack_density",
    "gpu_pair_unpack_gradient",
    "gpu_pair_unpack_force",
    "gpu_pair_launch_density",
    "gpu_pair_launch_gradient",
    "gpu_pair_launch_force",
    "gpu_pair_recurse",
]


cols_to_use = []
# add +1: 0th column in `timer_names` is "step"
cols_to_use.append(timer_names.index("gpu_self_pack_density") + 1)
cols_to_use.append(timer_names.index("gpu_self_pack_gradient") + 1)
cols_to_use.append(timer_names.index("gpu_self_pack_force") + 1)
cols_to_use.append(timer_names.index("gpu_self_unpack_density") + 1)
cols_to_use.append(timer_names.index("gpu_self_unpack_gradient") + 1)
cols_to_use.append(timer_names.index("gpu_self_unpack_force") + 1)
cols_to_use.append(timer_names.index("gpu_self_launch_density") + 1)
cols_to_use.append(timer_names.index("gpu_self_launch_gradient") + 1)
cols_to_use.append(timer_names.index("gpu_self_launch_force") + 1)
cols_to_use.append(timer_names.index("gpu_pair_pack_density") + 1)
cols_to_use.append(timer_names.index("gpu_pair_pack_gradient") + 1)
cols_to_use.append(timer_names.index("gpu_pair_pack_force") + 1)
cols_to_use.append(timer_names.index("gpu_pair_unpack_density") + 1)
cols_to_use.append(timer_names.index("gpu_pair_unpack_gradient") + 1)
cols_to_use.append(timer_names.index("gpu_pair_unpack_force") + 1)
cols_to_use.append(timer_names.index("gpu_pair_launch_density") + 1)
cols_to_use.append(timer_names.index("gpu_pair_launch_gradient") + 1)
cols_to_use.append(timer_names.index("gpu_pair_launch_force") + 1)
cols_to_use.append(timer_names.index("gpu_pair_recurse") + 1)


data = np.loadtxt(args.timer_file, usecols=cols_to_use)
if nthreads > 1:
    print(f"Normalising times assuming {nthreads} threads")
    data /= nthreads
if args.seconds:
    data *= 1e-3

# Print values to screen
timesum_avg = 0.0
timesum_total = 0.0

print()
print("Times averaged over all available measured steps:")

if nthreads <= 1:
    print("{0:25} {1:>18s}".format("Task Type", f"Total time {units}"))

    for i, col in enumerate(cols_to_use):
        avg = data[first_index:, i].mean()
        name = timer_names[col - 1]  # subtract 1 to get the index in my hand-made list

        print("{0:25} {1:18.3e}".format(name, avg))
        timesum_avg += avg

    print()
    print(f"Total: {timesum_avg:18.3e} {units}")

else:
    print(
        "{0:25} {1:>18} {2:>18}".format(
            "Task Type", f"Avg. Time {nthreads} thr", f"Total time {units}"
        )
    )

    for i, col in enumerate(cols_to_use):
        avg = data[first_index:, i].mean()
        name = timer_names[col - 1]  # subtract 1 to get the index in my hand-made list

        print("{0:25} {1:18.3e} {2:18.3e}".format(name, avg, avg * nthreads))
        timesum_avg += avg
        timesum_total += avg * nthreads

    print()
    print(f"Total: All threads:     {timesum_total:18.3e} {units}")
    print(f"       Avg. per thread: {timesum_avg:18.3e} {units}")
print()

fig = plt.figure(figsize=(8, 8), dpi=200)

ax = fig.add_subplot(111)
#with open("average_timings.txt", "w", encoding="utf-8") as f:
#    f.write("gpu_self_pack_density, gpu_self_pack_gradient, gpu_self_pack_force, " \
#    "gpu_self_unpack_density, gpu_self_unpack_gradient, gpu_self_unpack_force, "\
#    "gpu_self_launch_density, gpu_self_launch_gradient, gpu_self_launch_force, "\
#    "gpu_pair_pack_density, gpu_pair_pack_gradient, gpu_pair_pack_force, "\
#    "gpu_pair_unpack_density, gpu_pair_unpack_gradient, gpu_pair_unpack_force, "\
#    "gpu_pair_launch_density, gpu_pair_launch_gradient, gpu_pair_launch_force, "\
#    "gpu_pair_recurse,\n")
with open("average_timings.txt", "w", encoding="utf-8") as f:
    f.write("pack_unpack, launch \n")
pack_time_self = 0.0
launch_time_self = 0.0
pack_time_pair = 0.0
launch_time_pair = 0.0

for i in range(0, 6):
    pack_time_self = pack_time_self + data[:, i].mean()

for i in range(9, 15):
    pack_time_pair = pack_time_pair + data[:, i].mean()
pack_time_pair = pack_time_pair + data[:, 18].mean()

for i in range(6,9):
    launch_time_self = launch_time_self + data[:, i].mean()

for i in range(15,18):
    launch_time_pair = launch_time_pair + data[:, i].mean()
#launch_time = launch_time * nthreads

with open("average_timings.txt", "a", encoding="utf-8") as f:
    f.write(str(pack_time_self))
    f.write(", ")
    f.write(str(launch_time_self))    
    f.write(", ")
    f.write(str(pack_time_pair))
    f.write(", ")
    f.write(str(launch_time_pair))



for i, col in enumerate(cols_to_use):
    avg = data[:, i].mean()
    minval = data[:, i].min()
    maxval = data[:, i].max()

    name = timer_names[col - 1]  # subtract 1 to get the index in my hand-made list
    color = "C0"
    if "_pack_" in name:
        color = "C0"
    if "_unpack_" in name:
        color = "C1"
    if "launch" in name:
        color = "C2"
    if "recurse" in name:
        color = "C3"
    ax.errorbar(
        name,
        avg,
        yerr=[[avg - minval], [maxval - avg]],
        c=color,
        capsize=4,
        fmt="o",
        markersize=4,
    )

if nthreads <= 1:
    ax.set_ylabel(f"Task times [{units}] summed over all threads")
else:
    ax.set_ylabel(f"Task times [{units}] averaged per thread ({nthreads} total)")

locs = ax.get_xticks()
labels = ax.get_xticklabels()
ax.set_xticks(locs, labels, rotation="vertical")
ax.set_yscale("log")
ax.grid()

plt.tight_layout(rect=(0.05, 0.05, 0.95, 0.95))

plt.savefig("gpu_timers.png")
print("saved gpu_timers.png")
