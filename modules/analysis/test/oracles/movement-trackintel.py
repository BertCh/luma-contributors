# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference staypoints, triplegs, locations, trips, tracking quality and modal split from trackintel.

Writes fixtures/movement-trackintel.json next to this script, consumed by
test/gpu-spatial-analysis/trajectory-staypoints/trackintel-parity.spec.ts and
test/gpu-spatial-analysis/trajectory-mobility/mobility-indicators.spec.ts. Run from anywhere with
    <venv>/bin/python -I modules/analysis/test/oracles/movement-trackintel.py
in a venv with trackintel (versions are recorded in the fixture).

Two users in EPSG:4326 near Zurich: stays with jitter below 40 m and some exact duplicate
coordinates, walks in 25 m steps every 20 s, a 30-minute gap inside a walk, a stay that ends
the track (for include_last), and a user whose first activity is ignored.
Deterministic: rejection-sampled numpy seed so no tested pair is within 1.0 m of a threshold. Haversine radius 6371000 (trackintel's), so the GPU spec sets
`sphereRadius: 6371000`. Row positions are written both as written and rounded to float32 so the
spec can check the rounding did not move any distance across a threshold (see `margins`).
"""

import datetime
import json
import os
import platform

import geopandas as gpd
import numpy as np
import pandas as pd
import trackintel as ti
from shapely.geometry import Point

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'movement-trackintel.json')
METERS_PER_DEGREE_LATITUDE = 6371000 * np.pi / 180.0
ORIGIN = (8.5417, 47.3769)
EPOCH_MS = 1_700_000_000_000


def to_lon_lat(east, north):
    lat = ORIGIN[1] + north / METERS_PER_DEGREE_LATITUDE
    lon = ORIGIN[0] + east / (METERS_PER_DEGREE_LATITUDE * np.cos(np.deg2rad(ORIGIN[1])))
    return lon, lat


def make_user(rng, user, segments):
    """segments: ('stay', east, north, rows, jitter) or ('walk', east, north, rows) or ('gap', seconds)."""
    rows, time_ms, east, north = [], 0, 0.0, 0.0
    for segment in segments:
        kind = segment[0]
        if kind == 'gap':
            time_ms += int(segment[1] * 1000)
            continue
        if kind == 'walk':
            target_east, target_north = segment[1], segment[2]
            # Fixed 36 m steps: consecutive multiples of 36 m stay far from the 60 m and 100 m thresholds.
            count = max(1, round(np.hypot(target_east - east, target_north - north) / 36.0))
            for step in range(1, count + 1):
                fraction = step / count
                noise_east, noise_north = rng.uniform(-0.3, 0.3, 2)
                rows.append((user, time_ms, east + (target_east - east) * fraction + noise_east,
                             north + (target_north - north) * fraction + noise_north))
                time_ms += 20_000
            east, north = target_east, target_north
        else:
            stay_east, stay_north, count, jitter = segment[1], segment[2], segment[3], segment[4]
            for index in range(count):
                offset_east, offset_north = rng.uniform(-jitter, jitter, 2)
                if index % 7 == 3:
                    # Exact duplicate coordinates, which trackintel's union_all collapses.
                    offset_east, offset_north = 2.0, -1.5
                rows.append((user, time_ms, stay_east + offset_east, stay_north + offset_north))
                time_ms += 20_000
            east, north = stay_east, stay_north
    return rows


USERS = {
    0: [
        ('stay', 0.0, 0.0, 50, 12.0),
        ('walk', 800.0, 0.0, 32),
        ('stay', 800.0, 0.0, 40, 18.0),
        ('walk', 800.0, 600.0, 24),
        ('gap', 1800.0),
        ('walk', 1300.0, 600.0, 20),
        ('stay', 1300.0, 600.0, 45, 15.0),
        ('walk', 0.0, 0.0, 60),
        ('stay', 0.0, 20.0, 30, 12.0)
    ],
    1: [
        ('walk', 500.0, 500.0, 30),
        ('stay', 500.0, 500.0, 60, 20.0),
        ('walk', -700.0, 300.0, 50),
        ('stay', -700.0, 300.0, 18, 10.0),
        ('walk', 500.0, 500.0, 44),
        ('stay', 500.0, 500.0, 60, 20.0),
        ('walk', -700.0, 300.0, 50)
    ],
    # Gaps directly around activities: a trip origin after a gap and a destination before one.
    2: [
        ('stay', 100.0, 100.0, 60, 10.0),
        ('gap', 1800.0),
        ('walk', 700.0, 100.0, 20),
        ('stay', 700.0, 100.0, 50, 15.0),
        ('walk', 1000.0, 400.0, 16),
        ('gap', 1800.0),
        ('stay', 1000.0, 400.0, 50, 10.0)
    ]
}
GAP_RUNS = [15.0, 24 * 60.0]
DISTANCE_THRESHOLDS = [100, 60]


def haversine(a, b):
    """trackintel's point_haversine_dist in degrees, radius 6371000."""
    radius = 6371000
    longitude_a, latitude_a = np.deg2rad(a[0]), np.deg2rad(a[1])
    longitude_b, latitude_b = np.deg2rad(b[0]), np.deg2rad(b[1])
    d_latitude = latitude_b - latitude_a
    d_longitude = longitude_b - longitude_a
    value = np.sin(d_latitude / 2) ** 2 + np.cos(latitude_a) * np.cos(latitude_b) * np.sin(d_longitude / 2) ** 2
    return radius * 2 * np.arcsin(np.sqrt(value))


def tested_pair_margin(points, track_times, offsets, threshold, gap):
    """Smallest |distance - threshold| over every (anchor, row) pair the sliding rule tests."""
    margin = float('inf')
    for begin, end in zip(offsets[:-1], offsets[1:]):
        start = begin
        for current in range(begin + 1, end):
            if track_times[current] - track_times[current - 1] > gap * 60_000:
                start = current
                continue
            distance = haversine(points[start], points[current])
            margin = min(margin, abs(distance - threshold))
            if distance >= threshold:
                start = current
    return margin


# Rejection-sample the seed so no tested distance sits within the float32 rounding of a
# threshold: float32 degrees move a point by up to about 0.5 m at 47 degrees latitude.
MARGIN_METERS = 1.0
for seed in range(7, 4000):
    rng = np.random.default_rng(seed)
    rows = []
    track_offsets = [0]
    for user, segments in USERS.items():
        user_rows = make_user(rng, user, segments)
        rows.extend(user_rows)
        track_offsets.append(len(rows))
    lon_lat = [to_lon_lat(east, north) for _, _, east, north in rows]
    candidate_times = [row[1] for row in rows]
    margins = [
        tested_pair_margin(lon_lat, candidate_times, track_offsets, threshold, gap)
        for threshold in DISTANCE_THRESHOLDS
        for gap in GAP_RUNS
    ]
    if min(margins) > MARGIN_METERS:
        print('seed', seed, 'margin', min(margins))
        break
else:
    raise RuntimeError('no seed found')

frame = gpd.GeoDataFrame(
    {
        'user_id': [row[0] for row in rows],
        'tracked_at': pd.to_datetime([EPOCH_MS + row[1] for row in rows], unit='ms', utc=True)
    },
    geometry=[Point(lon, lat) for lon, lat in lon_lat],
    crs='EPSG:4326'
)
frame = frame.set_index(pd.Index(range(len(frame)), name='id'))
positionfixes = ti.Positionfixes(frame)
times_ms = [EPOCH_MS + row[1] for row in rows]


def find_row(track, milliseconds):
    return times_ms.index(milliseconds, track_offsets[track], track_offsets[track + 1])


def sp_rows(pfs_with_sp):
    return [
        -1 if pd.isna(value) else int(value) for value in pfs_with_sp['staypoint_id'].to_numpy()
    ]


CASES = [
    {'name': 'include-last', 'dist': 100, 'time': 5.0, 'gap': 15.0, 'include_last': True},
    {'name': 'default', 'dist': 100, 'time': 5.0, 'gap': 15.0, 'include_last': False},
    {'name': 'tight', 'dist': 60, 'time': 3.0, 'gap': 15.0, 'include_last': False},
    {'name': 'no-gap-limit', 'dist': 100, 'time': 5.0, 'gap': 24 * 60.0, 'include_last': True}
]
output_cases = []
for case in CASES:
    pfs, sp = positionfixes.generate_staypoints(
        method='sliding',
        distance_metric='haversine',
        dist_threshold=case['dist'],
        time_threshold=case['time'],
        gap_threshold=case['gap'],
        include_last=case['include_last']
    )
    row_staypoints = sp_rows(pfs)
    table = []
    for staypoint_id in sp.index.tolist():
        member_rows = [i for i, value in enumerate(row_staypoints) if value == staypoint_id]
        row = sp.loc[staypoint_id]
        finished_ms = int(row['finished_at'].timestamp() * 1000)
        finish_row = find_row(int(row['user_id']), finished_ms)
        table.append(
            {
                'track': int(row['user_id']),
                'startRow': member_rows[0],
                'endRow': member_rows[-1] + 1,
                'finishRow': finish_row,
                'centroid': [row.geometry.x, row.geometry.y],
                'durationMs': int(
                    (row['finished_at'] - row['started_at']).total_seconds() * 1000
                )
            }
        )

    pfs_tpl, tpls = pfs.generate_triplegs(sp, method='between_staypoints', gap_threshold=case['gap'])
    row_triplegs = [
        -1 if pd.isna(value) else int(value) for value in pfs_tpl['tripleg_id'].to_numpy()
    ]
    tripleg_table = []
    for tripleg_id in tpls.index.tolist():
        member_rows = [i for i, value in enumerate(row_triplegs) if value == tripleg_id]
        tripleg_table.append(
            {
                'track': int(tpls.loc[tripleg_id, 'user_id']),
                'startRow': member_rows[0],
                'endRow': member_rows[-1] + 1
            }
        )

    sp_loc, locations = sp.generate_locations(
        method='dbscan', epsilon=150, num_samples=1, distance_metric='haversine', agg_level='user'
    )
    location_ids = [
        -1 if pd.isna(value) else int(value) for value in sp_loc['location_id'].to_numpy()
    ]
    location_table = []
    for location_id in locations.index.tolist():
        center = locations.loc[location_id, 'center']
        location_table.append(
            {'track': int(locations.loc[location_id, 'user_id']), 'center': [center.x, center.y]}
        )

    from trackintel.analysis.location_identification import freq_method

    purposes = freq_method(sp_loc.copy(), 'home', 'work')['purpose']
    home_locations = {str(user): -1 for user in USERS}
    work_locations = {str(user): -1 for user in USERS}
    for (index, purpose), location, user in zip(
        purposes.items(), sp_loc['location_id'].to_numpy(), sp_loc['user_id'].to_numpy()
    ):
        if purpose == 'home':
            home_locations[str(int(user))] = int(location)
        if purpose == 'work':
            work_locations[str(int(user))] = int(location)

    sp_activity = ti.analysis.create_activity_flag(sp.copy(), time_threshold=15.0)
    sp_trip, tpls_trip, trips = ti.preprocessing.generate_trips(
        sp_activity, tpls, gap_threshold=case['gap'], add_geometry=False
    )
    trip_table = []
    for trip_id in trips.index.tolist():
        row = trips.loc[trip_id]
        started_ms = int(row['started_at'].timestamp() * 1000)
        finished_ms = int(row['finished_at'].timestamp() * 1000)
        origin = row['origin_staypoint_id']
        destination = row['destination_staypoint_id']
        trip_table.append(
            {
                'track': int(row['user_id']),
                'startRow': find_row(int(row['user_id']), started_ms),
                'finishRow': find_row(int(row['user_id']), finished_ms),
                'origin': -1 if pd.isna(origin) else int(origin),
                'destination': -1 if pd.isna(destination) else int(destination)
            }
        )
    tripleg_trip_ids = [
        -1 if pd.isna(value) else int(value) for value in tpls_trip.sort_index()['trip_id'].to_numpy()
    ]
    staypoint_activity = [bool(value) for value in sp_activity['is_activity'].to_numpy()]

    quality = ti.analysis.temporal_tracking_quality(
        pd.concat(
            [
                pd.DataFrame(sp[['user_id', 'started_at', 'finished_at']]),
                pd.DataFrame(tpls[['user_id', 'started_at', 'finished_at']])
            ],
            ignore_index=True
        ),
        granularity='all'
    )
    tracking_quality = {int(row.user_id): float(row.quality) for row in quality.itertuples()}

    tpls_mode = tpls.copy()
    tpls_mode['mode'] = ['slow_mobility' if i % 2 == 0 else 'motorized_mobility' for i in range(len(tpls_mode))]
    modal_split_count = {}
    for metric in ['count', 'duration']:
        split = ti.analysis.calculate_modal_split(tpls_mode, metric=metric, per_user=True)
        modal_split_count[metric] = {
            str(user): {column: float(split.loc[user, column]) for column in split.columns}
            for user in split.index
        }

    # Smallest absolute distance between any tested pair and the threshold, so the float32 spec
    # can bound the risk of a test flipping under rounding.
    output_cases.append(
        {
            'name': case['name'],
            'distanceThreshold': case['dist'],
            'timeThresholdMinutes': case['time'],
            'gapThresholdMinutes': case['gap'],
            'includeLast': case['include_last'],
            'staypoints': table,
            'rowStaypointIds': row_staypoints,
            'triplegs': tripleg_table,
            'rowTriplegIds': row_triplegs,
            'locationIds': location_ids,
            'locations': location_table,
            'staypointIsActivity': staypoint_activity,
            'trips': trip_table,
            'triplegTripIds': tripleg_trip_ids,
            'homeLocations': home_locations,
            'workLocations': work_locations,
            'trackingQuality': {str(key): value for key, value in tracking_quality.items()},
            'modalSplit': modal_split_count
        }
    )

result = {
    'generator': 'modules/analysis/test/oracles/movement-trackintel.py',
    'versions': {
        'python': platform.python_version(),
        'trackintel': ti.__version__,
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'geopandas': gpd.__version__
    },
    'margin': {'meters': MARGIN_METERS, 'seed': seed},
    'scenario': {
        'positions': [list(pair) for pair in lon_lat],
        'timesMs': times_ms,
        'trackOffsets': track_offsets,
        'sphereRadius': 6371000
    },
    'cases': output_cases
}
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, 'w') as handle:
    json.dump(result, handle, indent=1)
print('wrote', OUTPUT)
print(len(rows), 'rows', [len(case['staypoints']) for case in output_cases])
