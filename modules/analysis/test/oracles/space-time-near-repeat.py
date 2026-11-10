# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""pointpats 2.6.0 (BSD-3) reference for the near-repeat Knox table, local Knox and Jacquez test.

Library: pointpats.spacetime.Knox, KnoxLocal and jacquez (PySAL, BSD-3). The NearRepeat R package
(GPL-3) is NOT used: only its definition (counts per spatial band x temporal band, a Monte Carlo
p-value per cell) is followed.

Near-repeat table. pointpats has no band table, so each cell is built from cumulative
`Knox(delta=edge_s, tau=edge_t)` runs by inclusion-exclusion. Every run is preceded by the same
`numpy.random.seed`, and `_knox` draws its permutations from `numpy.random.permutation` without
consuming randomness elsewhere, so all runs see identical permutations and the differences are valid
per-cell permutation counts. Bands are (low, high], the first one [0, high], because the pointpats
neighbour tests are `<= delta` and `<= tau`.

Local Knox. `KnoxLocal(...).nsti` are the observed per-event counts and `p_sims` the conditional
permutation p-values (the focal event keeps its time).

Jacquez. `jacquez(space, time, k, permutations=0)` is the observed statistic; the p-value comes from
`permutations` draws. Times are continuous, so no k-nearest-neighbour ties exist (asserted).

The GPU operators draw their own swap-or-not permutations, so the permutation draws here are not
the GPU draws. The spec compares observed counts exactly and the permutation inference
statistically.
"""
import argparse
import json
import platform
import warnings
from pathlib import Path

import libpysal
import numpy
import pointpats
import scipy
from pointpats.spacetime import Knox, KnoxLocal, jacquez
from scipy.spatial.distance import pdist

# libpysal warns that the clustered k-nearest-neighbour graph is not connected; that is expected.
warnings.filterwarnings('ignore', message='The weights matrix is not fully connected')

PERMUTATIONS = 999
SEED = 31015
SPATIAL_EDGES = [0.0, 1.0, 2.5, 4.0]
TEMPORAL_EDGES = [0.0, 2.0, 5.0, 9.0]
LOCAL_DELTA = 2.5
LOCAL_TAU = 5.0
JACQUEZ_K = 4


def makeLatticeEvents(count, seed):
    """Space-time clustered events on a lattice: positions 1/8, times 1/2, so f32 holds them exactly."""
    random = numpy.random.default_rng(seed)
    positions = numpy.zeros((count, 2))
    times = numpy.zeros(count)
    for index in range(count):
        if index % 3 == 0:
            center = random.integers(0, 4)
            positions[index] = [center * 3 + random.random() * 2, center * 2.5 + random.random() * 2]
            times[index] = center * 11 + random.random() * 3
        else:
            positions[index] = random.random(2) * 12
            times[index] = random.random() * 44
    return numpy.rint(positions * 8) / 8, numpy.rint(times * 2) / 2


def assertEdgesSafe(positions, times):
    """A distance is either exactly on an edge or farther than 1e-4 from every edge."""
    distances = pdist(positions)
    for edge in SPATIAL_EDGES + [LOCAL_DELTA]:
        gap = numpy.abs(distances - edge)
        assert numpy.all((gap == 0) | (gap > 1e-4)), edge
    float32 = distances.astype(numpy.float32).astype(float)
    for edge in SPATIAL_EDGES + [LOCAL_DELTA]:
        assert numpy.all((distances <= edge) == (float32 <= edge)), edge


def cumulativeKnox(positions, times, delta, tau):
    numpy.random.seed(SEED)
    knox = Knox(positions, times.reshape(-1, 1), delta, tau, permutations=PERMUTATIONS, keep=True)
    return knox.nst, numpy.asarray(knox.sim, dtype=float)


def makeTable(positions, times):
    spatialBands = len(SPATIAL_EDGES) - 1
    temporalBands = len(TEMPORAL_EDGES) - 1
    observed = numpy.zeros((spatialBands + 1, temporalBands + 1))
    permuted = numpy.zeros((spatialBands + 1, temporalBands + 1, PERMUTATIONS))
    for spatial in range(1, spatialBands + 1):
        for temporal in range(1, temporalBands + 1):
            observed[spatial, temporal], permuted[spatial, temporal] = cumulativeKnox(
                positions, times, SPATIAL_EDGES[spatial], TEMPORAL_EDGES[temporal]
            )
    cellObserved = numpy.zeros((spatialBands, temporalBands), dtype=int)
    cellPermuted = numpy.zeros((spatialBands, temporalBands, PERMUTATIONS))
    for spatial in range(1, spatialBands + 1):
        for temporal in range(1, temporalBands + 1):
            cellObserved[spatial - 1, temporal - 1] = (
                observed[spatial, temporal]
                - observed[spatial - 1, temporal]
                - observed[spatial, temporal - 1]
                + observed[spatial - 1, temporal - 1]
            )
            cellPermuted[spatial - 1, temporal - 1] = (
                permuted[spatial, temporal]
                - permuted[spatial - 1, temporal]
                - permuted[spatial, temporal - 1]
                + permuted[spatial - 1, temporal - 1]
            )
    exceedances = (cellPermuted >= cellObserved[:, :, None]).sum(axis=2)
    return {
        'spatialEdges': SPATIAL_EDGES,
        'temporalEdges': TEMPORAL_EDGES,
        'permutations': PERMUTATIONS,
        'observed': cellObserved.ravel().tolist(),
        'permutationMean': cellPermuted.mean(axis=2).ravel().tolist(),
        'permutationVariance': cellPermuted.var(axis=2, ddof=1).ravel().tolist(),
        'pseudoPGreater': ((exceedances + 1) / (PERMUTATIONS + 1)).ravel().tolist(),
    }


def makeLocal(positions, times):
    numpy.random.seed(SEED)
    local = KnoxLocal(positions, times.reshape(-1, 1), LOCAL_DELTA, LOCAL_TAU, permutations=PERMUTATIONS, keep=True)
    return {
        'delta': LOCAL_DELTA,
        'tau': LOCAL_TAU,
        'permutations': PERMUTATIONS,
        'observed': numpy.asarray(local.nsti).astype(int).tolist(),
        'permutationMean': numpy.asarray(local.sims).mean(axis=1).tolist(),
        'pseudoPGreater': numpy.asarray(local.p_sims).tolist(),
        'global': int(local.nst),
    }


def makeJacquezAttempt(count, seed):
    random = numpy.random.default_rng(seed)
    positions = numpy.zeros((count, 2))
    times = numpy.zeros(count)
    for index in range(count):
        if index % 3 == 0:
            center = random.integers(0, 4)
            positions[index] = [center * 25 + random.random() * 6, center * 20 + random.random() * 6]
            times[index] = center * 30 + random.random() * 4
        else:
            positions[index] = random.random(2) * 100
            times[index] = random.random() * 120
    positions = positions.astype(numpy.float32).astype(float)
    times = times.astype(numpy.float32).astype(float)
    # The k-th and (k + 1)-th nearest distances differ clearly for every event (the k-nearest set is unambiguous).
    timeDistances = numpy.sort(numpy.abs(times[:, None] - times[None]), axis=1)[:, 1:]
    spatialDistances = numpy.sort(numpy.linalg.norm(positions[:, None] - positions[None], axis=2), axis=1)[:, 1:]
    if numpy.min(timeDistances[:, JACQUEZ_K] - timeDistances[:, JACQUEZ_K - 1]) <= 1e-3:
        return None
    if numpy.min(spatialDistances[:, JACQUEZ_K] - spatialDistances[:, JACQUEZ_K - 1]) <= 1e-3:
        return None
    observed = jacquez(positions, times.reshape(-1, 1), JACQUEZ_K, permutations=0)
    numpy.random.seed(SEED)
    inference = jacquez(positions, times.reshape(-1, 1), JACQUEZ_K, permutations=PERMUTATIONS)
    assert inference['stat'] == observed
    return {
        'k': JACQUEZ_K,
        'permutations': PERMUTATIONS,
        'positions': positions.ravel().tolist(),
        'times': times.tolist(),
        'observed': int(observed),
        'pseudoPGreater': float(inference['pvalue']),
        'expectedUnderNull': count * JACQUEZ_K * JACQUEZ_K / (count - 1),
    }


def makeJacquez(count):
    for attempt in range(1000):
        result = makeJacquezAttempt(count, SEED + attempt)
        if result is not None:
            result['dataSeed'] = SEED + attempt
            return result
    raise AssertionError('no tie-free Jacquez data set')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'fixtures')
    arguments = parser.parse_args()
    positions, times = makeLatticeEvents(72, 11)
    assertEdgesSafe(positions, times)
    fixture = {
        'generator': Path(__file__).name,
        'reference': 'pointpats.spacetime Knox, KnoxLocal, jacquez (BSD-3); NearRepeat (GPL-3) used as a definition only',
        'versions': {
            'python': platform.python_version(),
            'numpy': numpy.__version__,
            'scipy': scipy.__version__,
            'pointpats': pointpats.__version__,
            'libpysal': libpysal.__version__,
        },
        'seed': SEED,
        'lattice': {
            'positions': positions.ravel().tolist(),
            'times': times.tolist(),
            'table': makeTable(positions, times),
            'local': makeLocal(positions, times),
        },
        'jacquez': makeJacquez(150),
    }
    output = arguments.output_dir / 'space-time-near-repeat.json'
    serialized = json.dumps(fixture, separators=(',', ':'), allow_nan=False) + '\n'
    output.write_text(serialized)
    print(output, len(serialized), fixture['lattice']['table']['observed'], fixture['jacquez']['observed'])


if __name__ == '__main__':
    main()
