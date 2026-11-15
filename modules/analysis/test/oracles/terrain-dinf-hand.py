# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Regenerate with /usr/bin/python3 modules/analysis/test/oracles/terrain-dinf-hand.py.

Independent NumPy oracle: fit a plane to each Tarboton triangle, constrain its
steepest direction to the triangle, then solve the absorbing-flow linear system.
HAND uses TauDEM DinfDistDown -m ave v with edge contamination checking enabled:
https://hydrology.usu.edu/taudem/taudem5/help53/DInfinityDistanceDown.html
The GPU uses dependency sweeps; this oracle uses dense linear algebra instead.
"""
import json
from pathlib import Path
import numpy as np

OFFSETS = np.array([(1, 0), (1, 1), (0, 1), (-1, 1),
                    (-1, 0), (-1, -1), (0, -1), (1, -1)])
FACETS = [(0, 1), (0, 7), (2, 1), (2, 3), (4, 3), (4, 5), (6, 5), (6, 7)]


def make_case(name, elevations, spacing):
    elevations = np.asarray(elevations, dtype=np.float64)
    height, width = elevations.shape
    count = elevations.size
    heights = elevations.ravel()
    fractions = np.zeros((count, 8), dtype=np.float64)
    directions = np.zeros(count, dtype=np.uint32)
    neighbors = np.full((count, 8), -1, dtype=int)
    for cell in range(count):
        row, column = divmod(cell, width)
        for direction, (column_offset, row_offset) in enumerate(OFFSETS):
            other_column, other_row = column + column_offset, row + row_offset
            if 0 <= other_column < width and 0 <= other_row < height:
                neighbors[cell, direction] = other_row * width + other_column
        best_slope = 0
        for cardinal, diagonal in FACETS:
            first, second = neighbors[cell, cardinal], neighbors[cell, diagonal]
            if first < 0 or second < 0:
                # Boundary cells are streams in these fixtures; their routing is unused.
                continue
            vectors = OFFSETS[[cardinal, diagonal]] * spacing
            gradient = np.linalg.solve(vectors, heights[cell] - heights[[first, second]])
            cardinal_unit = vectors[0] / np.linalg.norm(vectors[0])
            across = vectors[1] - np.dot(vectors[1], cardinal_unit) * cardinal_unit
            across /= np.linalg.norm(across)
            maximum_angle = np.arctan2(np.dot(vectors[1], across), np.dot(vectors[1], cardinal_unit))
            angle = np.arctan2(np.dot(gradient, across), np.dot(gradient, cardinal_unit))
            candidates = [0.0, maximum_angle]
            if 0 < angle < maximum_angle:
                candidates.append(angle)
            slopes = [np.dot(gradient, np.cos(candidate) * cardinal_unit + np.sin(candidate) * across)
                      for candidate in candidates]
            selected = int(np.argmax(slopes))
            if slopes[selected] > best_slope:
                best_slope = slopes[selected]
                fractions[cell] = 0
                fractions[cell, diagonal] = candidates[selected] / maximum_angle
                fractions[cell, cardinal] = 1 - fractions[cell, diagonal]
        for direction in range(8):
            neighbor = neighbors[cell, direction]
            if neighbor >= 0 and heights[cell] > heights[neighbor]:
                if directions[cell] == 0:
                    directions[cell] = 1 << direction
    # f32 input weights, f64 oracle calculation and one final output rounding.
    fractions = fractions.astype(np.float32).astype(np.float64)
    streams = np.zeros((height, width), dtype=np.uint32)
    streams[[0, -1], :] = 1
    streams[:, [0, -1]] = 1
    streams = streams.ravel()
    matrix = np.eye(count)
    source = np.zeros(count)
    absorption_source = np.zeros((count, count))
    for cell in range(count):
        if streams[cell]:
            absorption_source[cell, cell] = 1
            continue
        weights = fractions[cell] / fractions[cell].sum()
        for direction, weight in enumerate(weights):
            if weight == 0:
                continue
            neighbor = neighbors[cell, direction]
            matrix[cell, neighbor] -= weight
            source[cell] += weight * (heights[cell] - heights[neighbor])
    hand = np.linalg.solve(matrix, source).astype(np.float32)
    absorption = np.linalg.solve(matrix, absorption_source)
    drainage = []
    for probabilities in absorption:
        reached = np.flatnonzero(probabilities > 1e-10)
        drainage.append(int(reached[0]) if len(reached) == 1 else 0xfffffffd)
    return dict(name=name, width=width, height=height, spacing=spacing,
                elevation=heights.tolist(), fractions=fractions.ravel().tolist(),
                flowDirections=directions.tolist(), streams=streams.tolist(),
                hand=hand.tolist(), drainage=drainage)


cases = [
    make_case('oblique-plane', [[100 - 3 * column - row for column in range(5)] for row in range(5)], [1, 1]),
    make_case('uneven-rectangular', [[42, 39, 36, 34, 30], [41, 38, 34, 31, 27],
                                   [39, 35, 31, 28, 23], [36, 32, 29, 23, 19],
                                   [32, 29, 25, 20, 16]], [2, 3]),
    make_case('cardinal-ties', [[20 - 2 * column for column in range(5)] for row in range(5)], [1, 1])
]
Path(__file__).with_suffix('.json').write_text(json.dumps(cases, indent=2) + '\n')
