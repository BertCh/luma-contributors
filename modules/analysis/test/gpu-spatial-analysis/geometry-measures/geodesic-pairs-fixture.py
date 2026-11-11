# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Generate independent binary64 references: PROJ geodesics and SciPy rhumb quadrature.

Run with /usr/bin/python3 (NumPy, pyproj and SciPy are required).
"""

import json
import math
from pathlib import Path

import numpy
from pyproj import Geod
from scipy.integrate import quad

RADIUS = 6371008.8
sphere = Geod(a=RADIUS, f=0)
ellipsoid = Geod(ellps='WGS84')
scenes = [
    [-73.75, 40.5, 2.25, 48.75],
    [179.998, 12.5, -179.996, 12.501],
    [-4.0, -30.0, 100.0, 60.0],
    [0.0, 0.0, 179.9, 0.1],
    [20.0, 89.0, 40.0, 89.5],
    [15.0, 0.0, 15.001, 0.0],
]


def normalize_bearing(bearing):
    return (bearing + 180) % 360 - 180 if bearing % 360 != 180 else 180


fixtures = []
for scene in scenes:
    longitude, latitude, target_longitude, target_latitude = map(float, numpy.array(scene, dtype=numpy.float32))
    origin = [longitude, latitude]
    target = [target_longitude, target_latitude]
    sphere_initial, _, sphere_distance = sphere.inv(*origin, *target)
    midpoint_longitude, midpoint_latitude, _ = sphere.fwd(*origin, sphere_initial, sphere_distance / 2)
    midpoint_longitude = longitude + (midpoint_longitude - longitude + 180) % 360 - 180
    for model, geodesic in [('great-circle', sphere), ('ellipsoidal', ellipsoid)]:
        initial, reverse, distance = geodesic.inv(*origin, *target)
        fixtures.append(dict(model=model, origin=origin, target=target, distance=distance, initial=normalize_bearing(initial), final=normalize_bearing(reverse + 180), midpoint=[midpoint_longitude, midpoint_latitude]))
    # Rhumb oracle: integrate d(psi) = sec(phi) d(phi), rather than the production
    # half-angle atanh/log evaluation. Integrate the path length in its lat/lon chart.
    first = math.radians(latitude)
    second = math.radians(target_latitude)
    longitude_delta = math.radians((target_longitude - longitude + 180) % 360 - 180)
    latitude_delta = second - first
    stretched_delta = quad(lambda angle: 1 / math.cos(angle), first, second, epsabs=1e-12)[0]
    if latitude_delta == 0:
        distance = RADIUS * abs(math.cos(first) * longitude_delta)
        fraction = 0.5
    else:
        def speed(fraction):
            latitude_at_fraction = first + fraction * latitude_delta
            longitude_speed = longitude_delta * latitude_delta / (stretched_delta * math.cos(latitude_at_fraction))
            return RADIUS * math.hypot(latitude_delta, math.cos(latitude_at_fraction) * longitude_speed)
        distance = quad(speed, 0, 1, epsabs=1e-7)[0]
        fraction = quad(lambda angle: 1 / math.cos(angle), first, (first + second) / 2, epsabs=1e-12)[0] / stretched_delta
    bearing = math.degrees(math.atan2(longitude_delta, stretched_delta))
    fixtures.append(dict(model='rhumb', origin=origin, target=target, distance=distance, initial=bearing, final=bearing, midpoint=[longitude + math.degrees(longitude_delta) * fraction, (latitude + target_latitude) / 2]))

Path(__file__).with_suffix('.json').write_text(json.dumps(fixtures, indent=2) + '\n')
