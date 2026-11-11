# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""NumPy second checks for two hand-computed expectations of the trajectory specs.

Writes fixtures/trajectories-lines-numpy-checks.json next to this script. Run from anywhere with
    python modules/analysis/test/oracles/trajectories-lines-numpy-checks.py
(NumPy only; recorded with the pinned NumPy 2.5.4 on CPython 3.14. The first run, with NumPy 2.0.2
on the system Python 3.9.6, gave the same values to the last digit.)

1. Arc-length resampling of a long track (review finding trajectories-lines-03), consumed by
   test/gpu-spatial-analysis/trajectory-interpolation/gpu-trajectory-resample.spec.ts. The track
   is 86,400 steps back and forth between (0, 0) and (float32(13.9), 0), one row per time unit.
   The step lengths are taken from the float32 positions and summed in binary64 with
   numpy.cumsum; sample k of 8 is placed at k / 7 of the total with numpy.interp. The spec's own
   expectation is the closed form row = 86400 k / 7, which uses no sum at all.
2. Volume isopleths of six cells (trajectories-lines-04), consumed by
   test/gpu-spatial-analysis/trajectory-home-range/home-range-recursion-mob07.spec.ts: a stable
   descending argsort and a cumulative sum of the float32 densities.
"""

import json
import os
import platform

import numpy as np


def get_arc_length_samples():
    step_count = 86_400
    sample_count = 8
    step_length = np.float32(13.9)
    x = np.zeros(step_count + 1, dtype=np.float32)
    x[1::2] = step_length
    y = np.zeros(step_count + 1, dtype=np.float32)
    times = np.arange(step_count + 1, dtype=np.float64)
    # Float32 coordinate differences, lengths and their running sum in binary64.
    steps = np.hypot(np.diff(x.astype(np.float64)), np.diff(y.astype(np.float64)))
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    targets = cumulative[-1] * np.arange(sample_count) / (sample_count - 1)
    return {
        "stepCount": step_count,
        "sampleCount": sample_count,
        "stepLength": float(step_length),
        "totalLength": float(cumulative[-1]),
        "sampleTimes": np.interp(targets, cumulative, times).tolist(),
        "sampleX": np.interp(targets, cumulative, x.astype(np.float64)).tolist(),
    }


def get_volume_isopleths():
    level = 0.9
    densities = np.array([0.5, 0.3, 0.15, 0.05, 0.0, 0.0], dtype=np.float32).astype(np.float64)
    order = np.argsort(-densities, kind="stable")
    total = densities.sum()
    cumulative = np.cumsum(densities[order])
    before = cumulative - densities[order]
    volume = np.ones_like(densities)
    inside = np.zeros(len(densities), dtype=int)
    positive = densities[order] > 0
    volume[order[positive]] = cumulative[positive] / total
    inside[order[positive & (before / total < level)]] = 1
    return {
        "level": level,
        "densities": densities.tolist(),
        "volume": volume.tolist(),
        "inside": inside.tolist(),
    }


def main():
    fixture = {
        "generator": {
            "script": "trajectories-lines-numpy-checks.py",
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "arcLengthResample": get_arc_length_samples(),
        "volumeIsopleths": get_volume_isopleths(),
    }
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fixtures", "trajectories-lines-numpy-checks.json"
    )
    with open(path, "w", encoding="utf-8") as output:
        json.dump(fixture, output, indent=1)
        output.write("\n")


if __name__ == "__main__":
    main()
