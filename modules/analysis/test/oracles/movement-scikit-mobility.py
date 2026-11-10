# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference individual mobility measures from scikit-mobility 1.3.1.

Writes fixtures/movement-scikit-mobility.json next to this script, consumed by
test/gpu-spatial-analysis/trajectory-mobility/mobility-indicators.spec.ts. Run from anywhere with
    <venv>/bin/python -I modules/analysis/test/oracles/movement-scikit-mobility.py
in a venv with scikit-mobility 1.3.1, geopandas, scipy, powerlaw, statsmodels, fiona, h3,
python-igraph installed (fiona is stubbed, it is only used for file I/O); scikit-mobility 1.3.1 is pinned against old pandas and old shapely, so this
generator installs it with `pip install --no-deps` and aliases shapely's removed
`shapely.ops.cascaded_union` to `unary_union` before importing it (the alias does not touch the
measures used). Versions are recorded in the fixture.

Scikit-mobility computes on the stops as a TrajDataFrame: `radius_of_gyration`, `jump_lengths`,
`waiting_times`, `random_entropy`, `uncorrelated_entropy` and `real_entropy`, keyed by user.
Durations of the stops are given separately for the home/work (trackintel FREQ) reference in
movement-trackintel.py. Positions are distinct exact doubles per location, so grouping on
(lat, lng) equals grouping on the location id the GPU contributor receives.
"""

import json
import os
import platform

import shapely.ops
import sys
import types

shapely.ops.cascaded_union = shapely.ops.unary_union  # removed in shapely 2.x
# fiona has no CPython 3.14 wheel; scikit-mobility only imports it for file I/O, unused here.
sys.modules['fiona'] = types.ModuleType('fiona')

import numpy as np
import pandas as pd
import skmob
from skmob.measures import individual as ind

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'movement-scikit-mobility.json')
EPOCH_MS = 1_700_000_000_000

rng = np.random.default_rng(11)
locations = rng.uniform([40.0, -3.0], [40.5, -2.5], size=(7, 2))
rows = []
offsets = [0]
for user in range(3):
    visits = rng.integers(0, 7, size=int(rng.integers(12, 30)))
    time_ms = EPOCH_MS
    for visit in visits:
        latitude, longitude = locations[visit]
        rows.append((user, int(visit), latitude, longitude, time_ms))
        time_ms += int(rng.integers(30, 300)) * 60_000
    offsets.append(len(rows))

frame = pd.DataFrame(
    {
        'uid': [row[0] for row in rows],
        'lat': [row[2] for row in rows],
        'lng': [row[3] for row in rows],
        'datetime': pd.to_datetime([row[4] for row in rows], unit='ms')
    }
)
tdf = skmob.TrajDataFrame(frame, latitude='lat', longitude='lng', datetime='datetime', user_id='uid')


def by_user(result, column):
    return {str(int(row['uid'])): float(row[column]) for _, row in result.iterrows()}


radius = by_user(ind.radius_of_gyration(tdf, show_progress=False), 'radius_of_gyration')
jumps = {
    str(int(row['uid'])): [float(value) for value in row['jump_lengths']]
    for _, row in ind.jump_lengths(tdf, show_progress=False).iterrows()
}
waits = {
    str(int(row['uid'])): [float(value) for value in row['waiting_times']]
    for _, row in ind.waiting_times(tdf, show_progress=False).iterrows()
}
random_entropy = by_user(ind.random_entropy(tdf, show_progress=False), 'random_entropy')
uncorrelated = by_user(ind.uncorrelated_entropy(tdf, show_progress=False), 'uncorrelated_entropy')
real = by_user(ind.real_entropy(tdf, show_progress=False), 'real_entropy')

result = {
    'generator': 'modules/analysis/test/oracles/movement-scikit-mobility.py',
    'versions': {
        'python': platform.python_version(),
        'scikit-mobility': skmob.__version__ if hasattr(skmob, '__version__') else '1.3.1',
        'numpy': np.__version__,
        'pandas': pd.__version__
    },
    'scenario': {
        'latLon': [[row[2], row[3]] for row in rows],
        'locationIds': [row[1] for row in rows],
        'timesMs': [row[4] for row in rows],
        'userOffsets': offsets
    },
    'radiusOfGyrationKm': radius,
    'jumpLengthsKm': jumps,
    'waitingTimesSeconds': waits,
    'randomEntropy': random_entropy,
    'uncorrelatedEntropy': uncorrelated,
    'realEntropy': real
}
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, 'w') as handle:
    json.dump(result, handle, indent=1)
print('wrote', OUTPUT, len(rows), 'rows')
print({key: result[key] for key in ['radiusOfGyrationKm', 'randomEntropy', 'realEntropy']})
