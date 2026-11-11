# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Independent all-pairs shortest paths, generated with /usr/bin/python3 (without -I)."""

import json
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import floyd_warshall
from pyproj import Geod, Transformer

width, height = 5, 4
elevation = np.array([0, 2, 1, -3, 0, 1, 4, -2, 0, 2, 3, 1, 5, -1, 0, 4, 2, -3, 1, 3], dtype=np.float32)
friction = np.array([1, 3, 2, 4, 1, 2, 0, 3, 1, 4, 3, 2, 1, 4, 2, 1, 4, 2, 3, 1], dtype=np.float32) / 8
sources = [0, 19]
cases = []
for kind in ["grass", "tobler"]:
    # Exactly float32-representable parameters, with sheared anisotropic grid axes.
    column_x, row_x, column_y, row_y = 10., 2., 1., -8.
    graph = np.full((width * height, width * height), np.inf)
    np.fill_diagonal(graph, 0)
    for source in range(width * height):
        source_row, source_column = divmod(source, width)
        for destination in range(width * height):
            destination_row, destination_column = divmod(destination, width)
            column_step = source_column - destination_column
            row_step = source_row - destination_row
            if max(abs(column_step), abs(row_step)) != 1:
                continue
            distance = np.hypot(column_x * column_step + row_x * row_step, column_y * column_step + row_y * row_step)
            rise = float(elevation[destination]) - float(elevation[source])
            slope = rise / distance
            if kind == "grass":
                coefficient = 3. if rise > 0 else (-2. if slope < -0.25 else -0.5)
                energy = 2. * distance + coefficient * rise
            else:
                energy = 0.6 * distance * np.exp(3.5 * abs(slope + 0.05))
            graph[source, destination] = energy + 0.75 * 0.5 * float(friction[source] + friction[destination]) * distance
    paths = floyd_warshall(graph, directed=True)
    expected = np.min(paths[sources], axis=0).astype(np.float32)
    cases.append({"kind": kind, "expected": expected.tolist()})

geodesic = Geod(ellps="WGS84")
projection = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
mercator_center = 6378137. * np.arcsinh(np.tan(np.pi / 3))
ground_cases = []
for mode, affine in [
    ("geographic", [0.001, 0, 0, 0, -0.002, 60.001]),
    ("web-mercator", [10., 0, 0, 0, -1000., mercator_center + 500.])
]:
    north_edge = float(np.float32(affine[5]))
    south_edge = float(np.float32(affine[5] + affine[4]))
    center_edge = 0.5 * (north_edge + south_edge)
    step = float(np.float32(affine[0]))
    if mode == "web-mercator":
        longitude_start, latitude = projection.transform(0, center_edge)
        longitude_end, _ = projection.transform(step, center_edge)
    else:
        longitude_start, longitude_end, latitude = 0., step, center_edge
    # Tiny finite geodesic arc approximates the local parallel length to far below float32 error.
    _, _, distance = geodesic.inv(longitude_start, latitude, longitude_end, latitude)
    ground_cases.append({"mode": mode, "affine": affine, "expected": np.float32([0, 2 * distance, 4 * distance]).tolist()})

Path(__file__).with_suffix(".json").write_text(json.dumps({
    "oracle": "scipy.sparse.csgraph.floyd_warshall on the explicit directed graph",
    "width": width, "height": height, "affine": [10, 2, 0, 1, -8, 32],
    "elevation": elevation.tolist(), "friction": friction.tolist(), "sources": sources, "cases": cases,
    "groundOracle": "pyproj.Geod WGS84 inverse over a short parallel arc near 60 degrees north",
    "groundCases": ground_cases
}, indent=2) + "\n")
