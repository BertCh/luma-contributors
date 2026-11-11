# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Generate independent GEOS bounds/circle/rectangle and SciPy all-point diameter references."""
import json
from pathlib import Path

import numpy
from scipy.spatial.distance import pdist
from shapely import minimum_bounding_circle, minimum_bounding_radius
from shapely.geometry import MultiPoint

random = numpy.random.default_rng(20261111)
fixtures = []
for scene in range(24):
    count = 8 + scene * 3
    angles = numpy.linspace(0, 2 * numpy.pi, count, endpoint=False)
    if scene % 3 == 0:
        points = numpy.column_stack([numpy.cos(angles) * 5, numpy.sin(angles) * 3])
    else:
        points = random.normal(size=(count, 2)) * [3, 5]
    points = (points + [scene * 10, -scene * 7]).astype(numpy.float32).astype(numpy.float64)
    geometry = MultiPoint(points)
    circle = minimum_bounding_circle(geometry)
    bounds = circle.bounds
    hull = geometry.convex_hull
    fixtures.append(dict(
        positions=points.flatten().tolist(),
        hullSize=len(hull.exterior.coords) - 1,
        rectangleArea=geometry.minimum_rotated_rectangle.area,
        circle=[(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2, minimum_bounding_radius(geometry)],
        diameter=float(pdist(points).max()),
        bounds=list(geometry.bounds),
    ))

Path(__file__).with_suffix('.json').write_text(json.dumps(fixtures, indent=2) + '\n')
