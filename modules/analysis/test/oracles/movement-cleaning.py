# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference trajectory cleaning from MovingPandas 0.23.0, PTRAIL's Hampel filter and TopDownTimeRatio.

Writes fixtures/movement-cleaning.json next to this script, consumed by
test/gpu-spatial-analysis/trajectory-cleaning/trajectory-cleaning.spec.ts. Run from anywhere with
    <venv>/bin/python -I modules/analysis/test/oracles/movement-cleaning.py
in a venv with movingpandas==0.23.0 and hampel (the package PTRAIL 1.0 uses), geopandas.

Positions are EPSG:3857 metres, so MovingPandas measures distance in CRS units. Cases: observation
gap split, value-change split, minimum-distance and minimum-time thinning, and TopDownTimeRatio on
a zigzag track with non-uniform times. The Hampel series is a noisy sine with two spikes (window
5, 3 sigma); MovingPandas OutlierCleaner is recorded on isolated spikes, where the point-centred
rule agrees (see the spec).
"""

import datetime
import json
import os
import platform
import warnings

import geopandas as gpd
import movingpandas as mpd
import numpy as np
import pandas as pd
from hampel import hampel
from shapely.geometry import Point

warnings.simplefilter('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'movement-cleaning.json')
EPOCH_MS = 1_700_000_000_000


def make_trajectory(rows, identifier):
    frame = gpd.GeoDataFrame(
        {
            't': pd.to_datetime([EPOCH_MS + row[2] * 1000 for row in rows], unit='ms'),
            'value': [row[3] for row in rows],
            'trip_id': [identifier for _ in rows]
        },
        geometry=[Point(row[0], row[1]) for row in rows],
        crs='EPSG:3857'
    ).set_index('t')
    return mpd.Trajectory(frame, 'trip_id')


def kept_rows(generalized, source_rows):
    times = list(generalized.df.index)
    source_times = [pd.to_datetime(EPOCH_MS + row[2] * 1000, unit='ms') for row in source_rows]
    return sorted({source_times.index(time) for time in times})


rng = np.random.default_rng(3)
rows_a = []
time_s = 0
x = 0.0
for index in range(60):
    x += 30.0 + rng.uniform(-5.0, 5.0)
    y = 200.0 * np.sin(index / 6.0) + rng.uniform(-3.0, 3.0)
    time_s += int(rng.integers(5, 45)) + (3600 if index in (20, 41) else 0)
    value = 1 if index < 15 else 2 if index < 40 else 7
    rows_a.append((round(x, 3), round(y, 3), time_s, value))

gap_seconds = 1000
gap_traj = make_trajectory(rows_a, 'a')
gap_split = mpd.ObservationGapSplitter(gap_traj).split(gap=datetime.timedelta(seconds=gap_seconds))
gap_segments = [kept_rows(part, rows_a) for part in gap_split.trajectories]

value_split = mpd.ValueChangeSplitter(gap_traj).split(col_name='value')
value_segments = [kept_rows(part, rows_a) for part in value_split.trajectories]

minimum_distance = mpd.MinDistanceGeneralizer(gap_traj).generalize(tolerance=250.0)
minimum_time = mpd.MinTimeDeltaGeneralizer(gap_traj).generalize(tolerance=datetime.timedelta(seconds=300))
minimum_distance_rows = kept_rows(minimum_distance, rows_a)
minimum_time_rows = kept_rows(minimum_time, rows_a)

times_ratio = mpd.TopDownTimeRatioGeneralizer(gap_traj).generalize(tolerance=15.0)
tdtr_rows = kept_rows(times_ratio, rows_a)

# Spiky track for outlier flags, sparse spikes only.
rows_b = []
for index in range(40):
    time_b = index * 10
    xb = index * 12.0
    yb = 5.0 * np.cos(index / 3.0)
    if index in (9, 25):
        yb += 800.0
    rows_b.append((xb, yb, time_b, 0))
spike_traj = make_trajectory(rows_b, 'b')
# metres per second in CRS units: 12 m / 10 s normal, spikes tower over the 30 m/s limit.
cleaned = mpd.OutlierCleaner(spike_traj).clean(v_max=30, units=('m', 's'))
outlier_kept = kept_rows(cleaned, rows_b)
outlier_removed = [index for index in range(len(rows_b)) if index not in outlier_kept]

series = np.array([np.sin(index / 5.0) + 0.05 * np.sin(index * 1.7) for index in range(50)], dtype=np.float32)
series[12] += 4.0
series[33] -= 3.0
hampel_result = hampel(series, window_size=5, n_sigma=3.0)

result = {
    'generator': 'modules/analysis/test/oracles/movement-cleaning.py',
    'versions': {
        'python': platform.python_version(),
        'movingpandas': mpd.__version__,
        'pandas': pd.__version__,
        'geopandas': gpd.__version__,
        'numpy': np.__version__
    },
    'trajectory': {'rows': [list(row) for row in rows_a], 'gapSeconds': gap_seconds},
    'gapSegments': gap_segments,
    'valueSegments': value_segments,
    'minimumDistance': {'tolerance': 250.0, 'keptRows': minimum_distance_rows},
    'minimumTime': {'toleranceSeconds': 300.0, 'keptRows': minimum_time_rows},
    'timeRatio': {'tolerance': 15.0, 'keptRows': tdtr_rows},
    'spiky': {'rows': [list(row) for row in rows_b], 'removedRows': outlier_removed, 'vMax': 30.0},
    'hampel': {
        'series': [float(value) for value in series],
        'windowSize': 5,
        'nSigma': 3.0,
        'filtered': [float(value) for value in hampel_result.filtered_data],
        'outlierIndices': [int(value) for value in hampel_result.outlier_indices]
    }
}
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, 'w') as handle:
    json.dump(result, handle, indent=1)
print('wrote', OUTPUT)
print(gap_segments, value_segments)
print(minimum_distance_rows, minimum_time_rows, tdtr_rows, outlier_removed)
