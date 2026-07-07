#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt

GPU_TIMERS = [
    "gpu_pack_density",
    "gpu_pack_gradient",
    "gpu_pack_force",
    "gpu_unpack_density",
    "gpu_unpack_gradient",
    "gpu_unpack_force",
    "gpu_launch_density",
    "gpu_launch_gradient",
    "gpu_launch_force",
    "gpu_self_recurse",
    "gpu_pair_recurse",
]


def find_header_and_columns(filename):

    header_line = None

    with open(filename, "r") as f:

        for line in f:

            if "gpu_pair_recurse" in line:

                header_line = line
                break

    if header_line is None:
        raise RuntimeError(
            "Could not find timer header containing gpu_pair_recurse"
        )

    header_line = header_line.replace("##", "")
    header_line = header_line.replace("|", " ")

    headers = header_line.split()

    #
    # Remove "step" label
    #
    if headers[0] == "step":
        headers = headers[1:]

    #
    # Count columns in first data row
    #
    data_cols = None

    with open(filename, "r") as f:

        for line in f:

            line = line.strip()

            if line == "":
                continue

            if line.startswith("#"):
                continue

            data_cols = len(line.split())
            break

    if data_cols is None:
        raise RuntimeError(
            "Could not find data rows"
        )

    print()
    print("Header timer count =", len(headers))
    print("Data column count  =", data_cols)

    #
    # Build mapping.
    #
    # SWIFT files sometimes have one more header token
    # than numerical columns, so shift if necessary.
    #
    offset = 0

    if len(headers) == data_cols:
        offset = 1

    print("Column offset =", offset)

    column_map = {}

    for i, name in enumerate(headers):

        column_map[name] = i + offset

    cols = []
    names = []

    for timer in GPU_TIMERS:

        if timer not in column_map:
            continue

        cols.append(column_map[timer])
        names.append(timer)

    print()
    print("Found timers:\n")

    for name, col in zip(names, cols):
        print(f"{name:30s} -> {col}")

    return cols, names


parser = argparse.ArgumentParser(
    description="Analyse SWIFT GPU timers"
)

parser.add_argument(
    "timer_file",
    nargs="?",
    default="timers_0.txt",
)

args = parser.parse_args()

cols_to_use, timer_names = find_header_and_columns(
    args.timer_file
)

print()
print("Columns used:")
print(cols_to_use)

cols_to_use = [c - 1 for c in cols_to_use]

print(cols_to_use)

data = np.loadtxt(
    args.timer_file,
    comments="#",
    usecols=cols_to_use,
)

if data.ndim == 1:
    data = data.reshape(1, -1)

#
# Sanity check
#
print("\nSanity check using first timestep:\n")

for i, timer in enumerate(timer_names):

    first = data[0, i]

    print(
        f"{timer:30s} {first:12.3f}"
    )

#
# Basic consistency checks
#
launch_cols = [
    i for i, t in enumerate(timer_names)
    if t.startswith("gpu_launch_")
]

recurse_cols = [
    i for i, t in enumerate(timer_names)
    if t.endswith("_recurse")
]

if launch_cols and recurse_cols:

    launch_avg = np.mean(data[:, launch_cols])
    recurse_avg = np.mean(data[:, recurse_cols])

    print("\nConsistency checks:")
    print(
        f"Average launch time  = {launch_avg:.3f}"
    )
    print(
        f"Average recurse time = {recurse_avg:.3f}"
    )

    if launch_avg < recurse_avg:

        print(
            "\nWARNING: Launch times are smaller "
            "than recurse times. "
            "Column mapping may be wrong."
        )
    else:

        print(
            "\nColumn mapping looks plausible."
        )

averages = np.mean(data, axis=0)

print()
print("Average times (ms):\n")

for timer, avg in zip(timer_names, averages):

    print(
        f"{timer:30s} {avg:12.3f}"
    )

pack_total = 0.0
unpack_total = 0.0
launch_total = 0.0
recurse_total = 0.0

for timer, avg in zip(timer_names, averages):

    if timer.startswith("gpu_pack_"):
        pack_total += avg

    elif timer.startswith("gpu_unpack_"):
        unpack_total += avg

    elif timer.startswith("gpu_launch_"):
        launch_total += avg

    elif timer.endswith("_recurse"):
        recurse_total += avg

print()
print("Summary:\n")

print(f"Pack    : {pack_total:.3f} ms")
print(f"Unpack  : {unpack_total:.3f} ms")
print(f"Launch  : {launch_total:.3f} ms")
print(f"Recurse : {recurse_total:.3f} ms")

with open(
    "average_timings.csv",
    "w",
    encoding="utf-8",
) as f:

    f.write("timer,avg_ms\n")

    for timer, avg in zip(timer_names, averages):

        f.write(
            f"{timer},{avg:.6f}\n"
        )

print()
print("Wrote average_timings.csv")

fig, ax = plt.subplots(
    figsize=(10, 5),
    dpi=200,
)

x = np.arange(len(timer_names))

ax.bar(
    x,
    averages,
)

ax.set_xticks(x)
ax.set_xticklabels(
    timer_names,
    rotation=90,
)

ax.set_ylabel("Average time [ms]")
ax.set_yscale("log")
ax.grid(True)

plt.tight_layout()

plt.savefig(
    "gpu_timers.png",
    bbox_inches="tight",
)

print("Wrote gpu_timers.png")

fig, ax = plt.subplots(
    figsize=(5, 4),
    dpi=200,
)

labels = [
    "Pack",
    "Unpack",
    "Launch",
    "Recurse",
]

values = [
    pack_total,
    unpack_total,
    launch_total,
    recurse_total,
]

ax.bar(
    labels,
    values,
)

ax.set_ylabel("Average time [ms]")
ax.set_yscale("log")
ax.grid(True)

plt.tight_layout()

plt.savefig(
    "gpu_timer_summary.png",
    bbox_inches="tight",
)

print("Wrote gpu_timer_summary.png")
