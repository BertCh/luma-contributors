# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Recreate the PTS-141 scenes with NumPy 2.0.2 / scikit-learn 1.6.1.

Run: /usr/bin/python3 clustering-fixture-oracles.py --output <scratch/result.json>
No production clustering code is imported. Float32 scenes are promoted to binary64 before
fitting. Relocation uses the exact historical input dumps (their original RNG seed is unknown),
explicit first-k centers, n_init=1, Lloyd, tol=0; labels retain initialization order.
The separated scene also reproduces the historical n_init=10, random_state=0 runs
with both k-means++ and random initialization (Lloyd, max_iter=300, tol=1e-4).
DBSCAN uses self-inclusive min_samples=5, haversine radians in [latitude, longitude] order;
cluster memberships are canonicalized by smallest row, noise is listed separately.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import sklearn
from sklearn.cluster import DBSCAN, KMeans


def separated_scene():
    state = 12345

    def random():
        nonlocal state
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        return state / 2**32

    points = []
    for blob in range(4):
        angle = 2 * math.pi * blob / 4
        for _ in range(60):
            points.append([400 * math.cos(angle) + (random() + random() + random() - 1.5) * 6,
                           400 * math.sin(angle) + (random() + random() + random() - 1.5) * 6])
    return np.asarray(points, dtype=np.float32).astype(np.float64)


def geographic_scene():
    state = 77

    def random():
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        value = ((state ^ (state >> 15)) * (state | 1)) & 0xFFFFFFFF
        value = (value ^ (value + (((value ^ (value >> 7)) * (value | 61)) & 0xFFFFFFFF))) & 0xFFFFFFFF
        return ((value ^ (value >> 14)) & 0xFFFFFFFF) / 2**32

    points = []
    radius = 6371008.8
    for longitude, latitude in [(10, 0), (10, 45), (-20, 70)]:
        for _ in range(60):
            east = (random() + random() + random() - 1.5) * 600
            north = (random() + random() + random() - 1.5) * 600
            points.append([longitude + east / (radius * math.pi * math.cos(latitude * math.pi / 180) / 180),
                           latitude + north / (radius * math.pi / 180)])
    for _ in range(20):
        points.append([-50 + random() * 20, -30 + random() * 20])
    return np.asarray(points, dtype=np.float32).astype(np.float64)


def generate():
    if np.__version__ != '2.0.2' or sklearn.__version__ != '1.6.1':
        raise RuntimeError('Requires NumPy 2.0.2 and scikit-learn 1.6.1')
    points = separated_scene()
    separated = KMeans(n_clusters=4, init=points[[0, 60, 120, 180]], n_init=1,
                       algorithm='lloyd', max_iter=20, tol=0, random_state=0).fit(points)
    separated_configurations = [dict(initialization='explicit', nInit=1, randomState=0,
                                     maximumIterations=20, tolerance=0, inertia=separated.inertia_)]
    for initialization in ['k-means++', 'random']:
        historical = KMeans(n_clusters=4, init=initialization, n_init=10, random_state=0,
                            algorithm='lloyd', max_iter=300, tol=1e-4).fit(points)
        assert abs(historical.inertia_ - 3707.8660708989237) < 1e-8
        separated_configurations.append(dict(initialization=initialization, nInit=10, randomState=0,
                                              maximumIterations=300, tolerance=1e-4, inertia=historical.inertia_))
    # Canonical input dump after the duplicated first centers and the center at (40,30).
    tail = [7, 2, 0, -7, -1, -1, 0, -3, 0, -2, -5, 4, 4, 7, 0, -2, -2, -6,
            4, -4, -5, -1, 6, 1, 36, 27, 43, 29, 37, 29, 43, 38, 35, 27, 37, 20,
            36, 26, 44, 29, 34, 33, 39, 23, 38, 28, 37, 30, -37, 29, 55, -48]
    relocation = []
    for name, clusters, duplicates in [('one-empty', 3, 2), ('two-empty', 4, 3)]:
        positions = np.asarray([0, 0] * duplicates + [40, 30] + tail, dtype=np.float64).reshape(-1, 2)
        steps = {}
        for budget in [1, 2, 3, 50]:
            model = KMeans(n_clusters=clusters, init=positions[:clusters].copy(), n_init=1,
                           algorithm='lloyd', max_iter=budget, tol=0, random_state=0).fit(positions)
            steps[str(budget)] = dict(centers=model.cluster_centers_.ravel().tolist(),
                                      labels=model.labels_.tolist(), inertia=model.inertia_)
        relocation.append(dict(name=name, k=clusters, positions=positions.ravel().tolist(), steps=steps))
    geographic = geographic_scene()
    model = DBSCAN(eps=150 / 6371008.8, min_samples=5, metric='haversine').fit(np.radians(geographic[:, ::-1]))
    memberships = sorted([np.flatnonzero(model.labels_ == label).tolist()
                          for label in set(model.labels_) if label != -1], key=lambda rows: rows[0])
    # External values agree with the historical pins; this re-generation has an honest version.
    assert abs(separated.inertia_ - 3707.8660708989237) < 1e-8
    assert sorted(map(len, memberships)) == [4, 7, 23, 32, 41]
    return dict(generator='clustering-fixture-oracles.py', numpy=np.__version__, sklearn=sklearn.__version__,
                separatedInertia=separated.inertia_, separatedConfigurations=separated_configurations, relocation=relocation,
                geographic=dict(clusters=memberships, noise=np.flatnonzero(model.labels_ == -1).tolist(),
                                coreRows=sorted(model.core_sample_indices_.tolist())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.write_text(json.dumps(generate(), indent=2) + '\n')
