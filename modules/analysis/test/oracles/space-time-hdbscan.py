# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent STX-03 oracle: sklearn HDBSCAN on scaled space-time coordinates.

Run from the repository root with /usr/bin/python3. Inputs are rounded to float32
before the float64 oracle measures distances. Cluster IDs are compared up to relabeling.
"""
import json
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.cluster import HDBSCAN
from scipy.spatial.distance import cdist

random = np.random.default_rng(3041)
scenes = []


def add_scene(name, positions, times, scale, minimum_samples, minimum_cluster_size):
    positions = np.asarray(positions, dtype=np.float32).astype(np.float64)
    times = np.asarray(times, dtype=np.float32).astype(np.float64)
    scale = float(np.float32(scale))
    coordinates = np.column_stack([positions, times * scale])
    distances = cdist(coordinates, coordinates)
    core_distances = np.sort(distances, axis=1)[:, minimum_samples - 1]
    labels = HDBSCAN(min_samples=minimum_samples, min_cluster_size=minimum_cluster_size,
                     algorithm='brute', copy=True).fit_predict(coordinates)
    scenes.append(dict(name=name, positions=positions.tolist(), times=times.tolist(),
                       timeScale=scale, minimumSamples=minimum_samples,
                       minimumClusterSize=minimum_cluster_size,
                       labels=labels.tolist(), coreDistances=core_distances.tolist(),
                       clusterCount=len(set(labels) - {-1})))


# Three events at the same sites: spatial nearest neighbors give the wrong time cores.
sites = random.normal(0, 0.2, (24, 2))
positions = np.tile(sites, (3, 1))
times = np.repeat([0, 40, 100], len(sites)) + random.normal(0, 0.15, 72)
add_scene('overlapping-events', positions, times, 1, 4, 10)
add_scene('overlapping-events-zero-scale', positions, times, 0, 4, 10)
add_scene('overlapping-events-small-scale', positions, times, 0.002, 4, 10)
# Non-power-of-two leaf count and maximum private top-k capacity.
positions = np.vstack([random.normal(center, 0.3, (count, 2))
                       for center, count in [([0, 0], 86), ([4, 1], 86), ([1, 4], 85)]])
times = np.concatenate([random.normal(center, 0.5, count)
                        for center, count in [(0, 86), (20, 86), (40, 85)]])
add_scene('padding-257-maximum-samples', positions, times, 0.5, 32, 16)
add_scene('one-sample-cores', positions[:172], times[:172], 0.5, 1, 16)
add_scene('tied-duplicate-events', [[0, 0]] * 17, [0] * 8 + [10] * 9, 2, 2, 4)

fixture = dict(oracle='sklearn.cluster.HDBSCAN, Euclidean [x, y, timeScale*t], defaults (EOM)',
               versions=dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
               scenes=scenes)
Path('modules/analysis/test/oracles/fixtures/space-time-hdbscan.json').write_text(
    json.dumps(fixture, indent=2) + '\n')
