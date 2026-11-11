# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Generate independent CPU twin fixtures with scikit-learn OPTICS.

Run /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-clustering/optics-operation-fixtures.py
without -I. Float32 coordinates/times are promoted before fitting. Temporal scenes use a
precomputed matrix whose out-of-window distances exceed max_eps. Infinity is serialized as
"Infinity"; noise/predecessor -1 is serialized as the contributor's uint32 missing sentinel.
The sklearn hierarchy consists of inclusive ordering ranks, and labels are input-row indexed.
"""

import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.cluster import OPTICS, cluster_optics_dbscan, cluster_optics_xi
from scipy.spatial.distance import cdist


def serializeValues(values):
    if isinstance(values, np.ndarray):
        return serializeValues(values.tolist())
    if isinstance(values, list):
        return [serializeValues(value) for value in values]
    if isinstance(values, float) and np.isinf(values):
        return "Infinity"
    return values


def createScene(name, positions, minimumSamples, maximumDistance, times=None):
    positions = np.asarray(positions, dtype=np.float32).astype(np.float64)
    distances = cdist(positions, positions)
    options = {"minimumSamples": minimumSamples, "maximumDistance": maximumDistance}
    result = {"name": name, "positions": positions.flatten().tolist(), "options": options}
    if times is not None:
        times = np.asarray(times, dtype=np.float32).astype(np.float64)
        distances[np.abs(times[:, None] - times[None, :]) > 2] = maximumDistance * 2 + 1
        result["times"] = times.tolist()
        options["maximumTimeDistance"] = 2
    fitted = OPTICS(min_samples=minimumSamples, max_eps=maximumDistance, metric="precomputed").fit(distances)
    result["ordering"] = fitted.ordering_.tolist()
    result["coreDistances"] = serializeValues(fitted.core_distances_)
    result["reachability"] = serializeValues(fitted.reachability_)
    result["predecessors"] = [int(value) if value >= 0 else 0xffffffff for value in fitted.predecessor_]
    extractionDistance = maximumDistance / 2
    dbscan = cluster_optics_dbscan(reachability=fitted.reachability_, core_distances=fitted.core_distances_, ordering=fitted.ordering_, eps=extractionDistance)
    result["dbscan"] = {"extractionDistance": extractionDistance, "labels": [int(value) if value >= 0 else 0xffffffff for value in dbscan]}
    result["xi"] = []
    for xi in [0.05, 0.2]:
        for correction in [True, False]:
            labels, hierarchy = cluster_optics_xi(reachability=fitted.reachability_, predecessor=fitted.predecessor_, ordering=fitted.ordering_, min_samples=minimumSamples, min_cluster_size=minimumSamples, xi=xi, predecessor_correction=correction)
            result["xi"].append({"xi": xi, "predecessorCorrection": correction, "labels": [int(value) if value >= 0 else 0xffffffff for value in labels], "intervals": hierarchy.tolist()})
    return result


random = np.random.default_rng(20261111)
points = np.concatenate([random.normal([0, 0], 0.1, (8, 2)), random.normal([2, 0], 0.3, (9, 2)), random.normal([8, 4], 0.2, (7, 2)), [[20, 20]]])
points = points[random.permutation(len(points))]
fixture = {
    "generator": "optics-operation-fixtures.py",
    "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__},
    "scenes": [
        createScene("integer-ties-and-noise", [[value, 0] for value in [0, 10, 1, 11, 3, 13, 2, 12, 40]], 3, 5),
        createScene("nested-hierarchy", points, 3, 10),
        createScene("temporal-window", [[value, 0] for value in [0, 1, 2, 0, 1, 2, 20]], 2, 4, [0, 1, 2, 10, 11, 12, 0]),
    ],
}
Path(__file__).with_suffix('.json').write_text(json.dumps(fixture, indent=2, allow_nan=False) + '\n')
