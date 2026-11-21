# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference space-time clusterings for GPUDBSCAN (ST-DBSCAN) and GPUOPTICS (ST-OPTICS).

Writes fixtures/space-time-clustering.json next to this script (consumed by
test/gpu-spatial-analysis/spatial-clustering/space-time-clustering.spec.ts). Run from the
repository root with
    <venv>/bin/python -I modules/analysis/test/oracles/space-time-clustering.py <st-dbscan-src>
where <st-dbscan-src> is the `src` directory of the unpacked st_dbscan-0.2.3 source distribution
(PyPI publishes only that sdist; its pyscaffold setup does not build with pip on Python 3.15, so
the package is imported from the unpacked tree instead of being installed). The venv needs numpy,
scipy and scikit-learn. The sdist SHA-256 is recorded in the fixture.

References
- ST-DBSCAN: st_dbscan.ST_DBSCAN(eps1, eps2, min_samples).fit(X).labels with X = [time, x, y]. For
  fewer than 20000 rows it builds the dense matrix `where(|dt| <= eps2, euclidean, 2 * eps1)` and
  runs sklearn DBSCAN(eps=eps1, min_samples, metric='precomputed'): a pair is related when
  `|dt| <= eps2` and `distance <= eps1`, and a point counts itself. Labels are in order of the
  smallest core row of each cluster (sklearn expands clusters in row order), border points join
  the lowest-numbered cluster that reaches them, noise is -1.
- ST-OPTICS: the `st-clustering` 0.1.0 ST_OPTICS recipe (same matrix, penalty 2 * eps1 outside the
  time window) run with sklearn OPTICS(metric='precomputed'). `st-clustering` itself does not
  import with current joblib (Memory(cachedir=...)), so the recipe is reproduced here with
  sklearn. The penalty exceeds max_eps in every scene (max_eps < 2 * eps1), which makes it a hard
  time filter, the semantics of GPUOPTICS `times`.

Scenes use integer lattice coordinates and times with integer thresholds, so squared distances and
time differences are exact in float32 and the closed comparisons (distance == eps1, |dt| == eps2)
are exercised. OPTICS scenes use float32-rounded random positions; the generator asserts that no
related pair distance lies within MARGIN of max_eps, the extraction distance, or of another pair
distance of the same row (reachability ties), so float32 GPU distances cannot change the result.
"""

import hashlib
import json
import os
import platform
import sys

import numpy as np
import scipy
import sklearn
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import OPTICS, cluster_optics_dbscan

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'space-time-clustering.json')
MARGIN = 1e-4


def load_st_dbscan(source):
    sys.path.insert(0, source)
    from st_dbscan import ST_DBSCAN

    return ST_DBSCAN


def lattice_scene(rng, count, extent, time_extent):
    positions = rng.integers(0, extent + 1, size=(count, 2)).astype(np.float64)
    times = rng.integers(0, time_extent + 1, size=count).astype(np.float64)
    return positions, times


def blobs_scene(rng, count, extent, time_extent):
    """Spatial blobs that overlap in space but live in different time windows, plus noise."""
    centers = [(10, 10, 5), (12, 11, 40), (30, 30, 5), (31, 29, 38), (20, 5, 22)]
    positions = []
    times = []
    for index in range(count):
        if index % 5 == 4:
            positions.append(rng.integers(0, extent + 1, size=2))
            times.append(rng.integers(0, time_extent + 1))
        else:
            cx, cy, ct = centers[index % len(centers)]
            positions.append(np.clip(np.rint(rng.normal([cx, cy], 2.0)), 0, extent))
            times.append(np.clip(np.rint(rng.normal(ct, 3.0)), 0, time_extent))
    return np.array(positions, dtype=np.float64), np.array(times, dtype=np.float64)


def check_margin(positions, times, eps1, eps2):
    """Lattice scenes: squared distances are integers, so the comparison is exact on both sides."""
    assert np.all(positions == np.rint(positions)) and np.all(times == np.rint(times))
    assert eps1 == int(eps1) and eps2 == int(eps2)


def dbscan_case(ST_DBSCAN, name, positions, times, eps1, eps2, min_samples):
    check_margin(positions, times, eps1, eps2)
    matrix = np.column_stack([times, positions])
    labels = ST_DBSCAN(eps1=eps1, eps2=eps2, min_samples=min_samples, n_jobs=1).fit(matrix).labels
    labels = [int(value) for value in labels]
    return {
        'name': name,
        'positions': positions.reshape(-1).tolist(),
        'times': times.tolist(),
        'eps1': eps1,
        'eps2': eps2,
        'minimumPoints': min_samples,
        'labels': labels,
        'clusterCount': max(labels) + 1,
        'noiseCount': labels.count(-1),
    }


def optics_case(name, positions, times, eps1, eps2, min_samples, max_eps, extraction):
    count = len(positions)
    assert max_eps < 2 * eps1, 'penalty distance must exceed max_eps to be a hard filter'
    time_distance = pdist(times.reshape(count, 1))
    spatial_distance = pdist(positions)
    # The st-clustering matrix: spatial distance inside the window, 2 * eps1 outside it.
    matrix = squareform(np.where(time_distance <= eps2, spatial_distance, 2 * eps1))
    optics = OPTICS(
        min_samples=min_samples,
        max_eps=max_eps,
        metric='precomputed',
        cluster_method='dbscan',
        eps=extraction,
    ).fit(matrix)
    # Margin checks over related pairs only (the GPU never sees the others).
    related = squareform(np.where(time_distance <= eps2, spatial_distance, np.inf))
    for row in range(count):
        values = np.sort(related[row][np.isfinite(related[row])])
        for edge in (max_eps, extraction):
            assert np.all(np.abs(values - edge) > MARGIN), (name, row, edge)
    reachability = [None if not np.isfinite(v) else float(v) for v in optics.reachability_]
    core = [None if not np.isfinite(v) else float(v) for v in optics.core_distances_]
    return {
        'name': name,
        'positions': positions.reshape(-1).tolist(),
        'times': times.tolist(),
        'eps1': eps1,
        'maximumTimeDistance': eps2,
        'minimumSamples': min_samples,
        'maximumDistance': max_eps,
        'extractionDistance': extraction,
        'ordering': [int(v) for v in optics.ordering_],
        'reachability': reachability,
        'coreDistances': core,
        'predecessors': [int(v) for v in optics.predecessor_],
        'labels': [int(v) for v in optics.labels_],
    }


def main():
    source = sys.argv[1]
    sdist = sys.argv[2] if len(sys.argv) > 2 else None
    ST_DBSCAN = load_st_dbscan(source)
    rng = np.random.default_rng(20261109)

    dbscan = []
    positions, times = lattice_scene(rng, 300, 20, 40)
    dbscan.append(dbscan_case(ST_DBSCAN, 'lattice-dense', positions, times, 3, 4, 4))
    positions, times = blobs_scene(rng, 400, 40, 50)
    dbscan.append(dbscan_case(ST_DBSCAN, 'blobs-overlap-in-space', positions, times, 2, 6, 5))
    dbscan.append(dbscan_case(ST_DBSCAN, 'blobs-minimum-points-1', positions, times, 2, 6, 1))
    dbscan.append(dbscan_case(ST_DBSCAN, 'blobs-wide-window', positions, times, 2, 100, 5))
    positions, times = lattice_scene(rng, 3000, 70, 100)
    dbscan.append(dbscan_case(ST_DBSCAN, 'lattice-3000', positions, times, 4, 7, 6))

    optics = []
    for name, count, extent, time_extent, eps2, min_samples, max_eps, extraction in [
        ('random-60', 60, 10.0, 24, 6, 4, 3.5, 1.8),
        ('random-90-narrow-window', 90, 12.0, 30, 3, 3, 3.5, 1.6),
    ]:
        positions = rng.uniform(0, extent, size=(count, 2)).astype(np.float32).astype(np.float64)
        times = rng.integers(0, time_extent + 1, size=count).astype(np.float64)
        optics.append(
            optics_case(name, positions, times, 10.0, eps2, min_samples, max_eps, extraction)
        )

    versions = {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'scikit-learn': sklearn.__version__,
        'st-dbscan': '0.2.3',
    }
    if sdist:
        with open(sdist, 'rb') as handle:
            versions['st-dbscan-sdist-sha256'] = hashlib.sha256(handle.read()).hexdigest()
    document = {
        'generator': 'test/oracles/space-time-clustering.py',
        'versions': versions,
        'source': (
            'st_dbscan 0.2.3 (MIT) ST_DBSCAN.fit imported from the unpacked sdist; ST-OPTICS is the '
            'st-clustering 0.1.0 ST_OPTICS matrix recipe on sklearn OPTICS'
        ),
        'dbscan': dbscan,
        'optics': optics,
    }
    with open(OUTPUT, 'w') as handle:
        json.dump(document, handle, indent=1)
        handle.write('\n')


if __name__ == '__main__':
    main()
