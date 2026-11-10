# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference EDR, ERP, SSPD, Hausdorff and DTW distances for GPUTrackSimilarity.

Writes fixtures/track-similarity-edit-measures.json next to this script, consumed by
test/gpu-spatial-analysis/track-similarity/gpu-track-similarity-edit-measures.spec.ts (GPU) and
track-similarity-edit-measures.node.spec.ts (fixture sanity). Run from the repository root with
    <venv>/bin/python -I modules/analysis/test/oracles/track-similarity-edit-measures.py
in a venv with numpy, shapely and movingpandas==0.23.0 (versions recorded in the fixture).

Inputs are rounded to float32 first (the GPU reads float32 positions); every reference is then
evaluated in binary64.
- Hausdorff: `movingpandas.Trajectory.hausdorff_distance` (0.23, it calls Shapely/GEOS), also checked
  against `shapely.hausdorff_distance`.
- DTW: `movingpandas.Trajectory.dtw_distance` (exact, the sum of Euclidean point distances).
- EDR (Chen, Ozsu and Oria, SIGMOD 2005), ERP (Chen and Ng, VLDB 2004) and SSPD (Besse et al.,
  2016) are not in MovingPandas 0.23. They are written here from the papers' definitions, in the
  recursive head/rest form (memoized), which is a different formulation than the GPU's
  anti-diagonal table, and ERP is also cross-checked against the metric axioms.
    EDR(A, B) = |A| if B is empty, |B| if A is empty, else
      min(EDR(rest A, rest B) + subcost, EDR(rest A, B) + 1, EDR(A, rest B) + 1),
      subcost 0 when dist(a1, b1) <= epsilon, else 1 (a metric ball; Chen's original uses a per-axis box).
    ERP(A, B) = sum dist(b, g) if A is empty, sum dist(a, g) if B is empty, else
      min(ERP(rest A, rest B) + dist(a1, b1), ERP(rest A, B) + dist(a1, g), ERP(A, rest B) + dist(b1, g)).
    SSPD(A, B) = (SPD(A, B) + SPD(B, A)) / 2, SPD(A, B) = mean over vertices a of A of dist(a, polyline B).
- Great-circle cases (EDR, ERP): haversine distance in meters on the 6371008.8 m sphere.
The random cases keep every EDR tolerance more than 1e-3 relative away from every vertex distance
(f32 roundoff cannot flip a match); the deliberate exception is the exact integer case.
"""

import json
import math
import os
import platform
import sys
from functools import lru_cache

import numpy as np
import shapely
import movingpandas as mpd
import geopandas as gpd
import pandas as pd

sys.setrecursionlimit(10000)
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'track-similarity-edit-measures.json')
EARTH_RADIUS = 6371008.8


def to_f32(points):
    return [tuple(float(v) for v in p) for p in np.asarray(points, dtype=np.float32).astype(np.float64)]


def euclidean(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def haversine(p, q):
    lon1, lat1, lon2, lat2 = map(math.radians, (p[0], p[1], q[0], q[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS * math.asin(min(1.0, math.sqrt(h)))


def edr(a, b, epsilon, metric):
    @lru_cache(maxsize=None)
    def solve(i, j):
        if i == len(a):
            return float(len(b) - j)
        if j == len(b):
            return float(len(a) - i)
        sub = 0.0 if metric(a[i], b[j]) <= epsilon else 1.0
        return min(solve(i + 1, j + 1) + sub, solve(i + 1, j) + 1.0, solve(i, j + 1) + 1.0)

    return solve(0, 0)


def erp(a, b, gap, metric):
    @lru_cache(maxsize=None)
    def solve(i, j):
        if i == len(a):
            return sum(metric(b[k], gap) for k in range(j, len(b)))
        if j == len(b):
            return sum(metric(a[k], gap) for k in range(i, len(a)))
        return min(
            solve(i + 1, j + 1) + metric(a[i], b[j]),
            solve(i + 1, j) + metric(a[i], gap),
            solve(i, j + 1) + metric(b[j], gap),
        )

    return solve(0, 0)


def point_segment(p, q0, q1):
    dx, dy = q1[0] - q0[0], q1[1] - q0[1]
    length_squared = dx * dx + dy * dy
    t = 0.0 if length_squared == 0 else min(1.0, max(0.0, ((p[0] - q0[0]) * dx + (p[1] - q0[1]) * dy) / length_squared))
    return math.hypot(p[0] - q0[0] - t * dx, p[1] - q0[1] - t * dy)


def to_polyline(p, line):
    if len(line) == 1:
        return euclidean(p, line[0])
    return min(point_segment(p, line[k], line[k + 1]) for k in range(len(line) - 1))


def sspd(a, b):
    spd_ab = sum(to_polyline(p, b) for p in a) / len(a)
    spd_ba = sum(to_polyline(q, a) for q in b) / len(b)
    return 0.5 * (spd_ab + spd_ba)


def movingpandas_trajectory(points, identifier):
    times = pd.date_range('2026-01-01', periods=len(points), freq='min')
    frame = gpd.GeoDataFrame({'geometry': shapely.points(points)}, index=times, crs='EPSG:32631')
    return mpd.Trajectory(frame, identifier)


def hausdorff_reference(a, b):
    # A single-vertex "line" is a point; MovingPandas needs at least two rows, so those cases use Shapely.
    if len(a) < 2 or len(b) < 2:
        return float(shapely.hausdorff_distance(shapely.linestrings(a) if len(a) > 1 else shapely.points(a[0]),
                                                shapely.linestrings(b) if len(b) > 1 else shapely.points(b[0]))), None
    ta, tb = movingpandas_trajectory(a, 'a'), movingpandas_trajectory(b, 'b')
    moving = float(ta.hausdorff_distance(tb))
    direct = float(shapely.hausdorff_distance(shapely.linestrings(a), shapely.linestrings(b)))
    assert abs(moving - direct) <= 1e-12 * max(1.0, direct), (moving, direct)
    return moving, float(ta.dtw_distance(tb))


def dtw_reference(a, b):
    table = {}
    for i in range(len(a)):
        for j in range(len(b)):
            cost = euclidean(a[i], b[j])
            if i == 0 and j == 0:
                table[i, j] = cost
            else:
                table[i, j] = cost + min(table.get((i - 1, j), math.inf), table.get((i, j - 1), math.inf), table.get((i - 1, j - 1), math.inf))
    return table[len(a) - 1, len(b) - 1]


def safe_tolerances(a, b, candidates, metric):
    distances = [metric(p, q) for p in a for q in b]
    chosen = []
    for epsilon in candidates:
        if all(abs(d - epsilon) > 1e-3 * max(epsilon, 1e-9) for d in distances):
            chosen.append(epsilon)
    return chosen


def random_track(rng, count, origin, step):
    steps = rng.normal(0, step, size=(count, 2))
    return to_f32(np.asarray(origin) + np.cumsum(steps, axis=0))


def planar_case(name, a, b, tolerances, gaps):
    a, b = to_f32(a), to_f32(b)
    hausdorff, moving_dtw = hausdorff_reference(a, b)
    dtw = dtw_reference(a, b)
    if moving_dtw is not None:
        assert abs(moving_dtw - dtw) <= 1e-9 * max(1.0, dtw), (moving_dtw, dtw)
    return {
        'name': name,
        'a': a,
        'b': b,
        'hausdorff': hausdorff,
        'dynamicTimeWarping': dtw,
        'sspd': sspd(a, b),
        'edr': [{'tolerance': t, 'distance': edr(a, b, t, euclidean)} for t in tolerances],
        'erp': [{'gap': list(g), 'distance': erp(a, b, g, euclidean)} for g in gaps],
    }


def great_circle_case(name, a, b, tolerances, gaps):
    a, b = to_f32(a), to_f32(b)
    return {
        'name': name,
        'a': a,
        'b': b,
        'edr': [{'tolerance': t, 'distance': edr(a, b, t, haversine)} for t in tolerances],
        'erp': [{'gap': list(g), 'distance': erp(a, b, g, haversine)} for g in gaps],
    }


def main():
    rng = np.random.default_rng(20261109)
    planar = []
    planar.append(planar_case('equal-length', [[0, 0], [1, 0.5], [2, 0], [3, 0.5], [4, 0]], [[0, 1], [1, 1.5], [2, 1.2], [3, 1.4], [4, 1]], [0.7, 1.3, 2.5], [(0, 0), (2, 2)]))
    planar.append(planar_case('unequal-length', [[0, 0], [2, 0], [4, 0]], [[0, 0.5], [0.5, 0.4], [1, 0.6], [2, 0.5], [3, 0.7], [3.5, 0.3], [4, 0.4]], [0.35, 0.65], [(0, 0), (1, -1)]))
    planar.append(planar_case('single-vertex', [[3, 3]], [[0, 0], [1, 1], [2, 2]], [0.5, 2.5], [(0, 0), (3, 3)]))
    planar.append(planar_case('identical', [[0, 0], [1, 1], [2, 0]], [[0, 0], [1, 1], [2, 0]], [0.5], [(0, 0)]))
    # Exact case: |(3, 4)| = 5 so the tolerance 5 matches exactly (25 is exact in float32); the
    # other vertex pairs are far from the tolerance.
    exact = planar_case('exact-integer', [[0, 0], [10, 0]], [[3, 4], [13, 4]], [5.0, 4.0], [(0, 0)])
    planar.append(exact)
    for index in range(14):
        n = int(rng.integers(4, 70))
        m = int(rng.integers(4, 70))
        a = random_track(rng, n, (rng.uniform(-100, 100), rng.uniform(-100, 100)), 1.5)
        b = random_track(rng, m, (a[0][0] + rng.normal(0, 2), a[0][1] + rng.normal(0, 2)), 1.5)
        tolerances = safe_tolerances(a, b, [0.5, 2.0, 5.0], euclidean)
        planar.append(planar_case(f'random-{index}', a, b, tolerances, [(0, 0), (float(a[0][0]), float(a[0][1]))]))

    great_circle = []
    great_circle.append(great_circle_case('meters', [[-73.99, 40.73], [-73.985, 40.735], [-73.98, 40.74], [-73.975, 40.742]], [[-73.9902, 40.7305], [-73.9851, 40.7348], [-73.9797, 40.7402]], [20.0, 120.0], [(-74.0, 40.7), (0.0, 0.0)]))
    great_circle.append(great_circle_case('antimeridian', [[179.9, -16.0], [-179.95, -16.02], [-179.8, -16.05]], [[179.92, -16.01], [179.99, -16.015], [-179.9, -16.03], [-179.82, -16.06]], [1500.0, 6000.0], [(180.0, -16.0)]))
    for index in range(3):
        n = int(rng.integers(5, 30))
        m = int(rng.integers(5, 30))
        origin = (rng.uniform(-170, 170), rng.uniform(-60, 60))
        a = random_track(rng, n, origin, 0.01)
        b = random_track(rng, m, (origin[0] + 0.003, origin[1] - 0.002), 0.01)
        tolerances = safe_tolerances(a, b, [300.0, 1500.0], haversine)
        great_circle.append(great_circle_case(f'random-{index}', a, b, tolerances, [origin]))

    fixture = {
        'generator': 'test/oracles/track-similarity-edit-measures.py',
        'versions': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'shapely': shapely.__version__,
            'movingpandas': mpd.__version__ if hasattr(mpd, '__version__') else '0.23.0',
        },
        'earthRadius': EARTH_RADIUS,
        'externalOracle': 'movingpandas 0.23.0 supplies hausdorff and dynamicTimeWarping; EDR, ERP and SSPD have no MovingPandas function, so they come from the paper definitions in this script',
        'tolerances': {
            'hausdorffRelative': 4e-6,
            'hausdorffAbsolutePlanar': 1e-4,
            'dynamicTimeWarpingRelativePerStep': 2 ** -22,
            'dynamicTimeWarpingAbsolutePlanar': 1e-3,
            'sspdRelative': 1e-5,
            'sspdAbsolutePlanar': 1e-4,
            'erpRelativePerStep': 2 ** -22,
            'erpAbsolutePlanar': 1e-3,
            'erpAbsoluteMetersPerStep': 0.5,
        },
        'planar': planar,
        'greatCircle': great_circle,
    }
    with open(OUTPUT, 'w') as file:
        json.dump(fixture, file, indent=1)
        file.write('\n')
    print(f'wrote {OUTPUT}: {len(planar)} planar and {len(great_circle)} great-circle cases')


if __name__ == '__main__':
    main()
