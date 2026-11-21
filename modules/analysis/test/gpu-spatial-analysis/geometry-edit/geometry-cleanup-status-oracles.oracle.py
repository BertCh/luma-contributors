# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/geometry-edit/geometry-cleanup-status-oracles.oracle.py
# Independent GEOS/Shapely oracle. No production or test helper is imported.
from shapely import LineString, Polygon, remove_repeated_points, set_precision

line = LineString([(0, 0), (1, 0), (0, 1)])
for gridSize in [0, 10]:
    for tolerance in [0, 0.5]:
        cleaned = remove_repeated_points(
            set_precision(line, gridSize, mode='pointwise'), tolerance=tolerance
        )
        expectedCoordinates = (
            [(0, 0), (1, 0), (0, 1)] if gridSize == 0 else [(0, 0), (0, 0)]
        )
        assert list(cleaned.coords) == expectedCoordinates
        requiredCount = len(expectedCoordinates)
        for capacity in [1, 4]:
            status = [min(requiredCount, capacity), requiredCount, int(requiredCount > capacity)]
            print(f'grid={gridSize}, tolerance={tolerance}, capacity={capacity}: {status}')

for tolerance in [0, 0.5]:
    polygon = remove_repeated_points(Polygon([(0, 0), (1, 0), (0, 1)]), tolerance=tolerance)
    assert list(polygon.exterior.coords) == [(0, 0), (1, 0), (0, 1), (0, 0)]
    print(f'closed polygon, tolerance={tolerance}: [1,4,1,0] / [4,4,0,0] at capacities 1/4')
    # GEOS does not publish a degenerate polygon. The contributor explicitly
    # empties rings with fewer than three surviving rows and counts the collapse.
    collapsedBoundary = remove_repeated_points(
        set_precision(LineString(polygon.exterior.coords), 10, mode='pointwise'),
        tolerance=tolerance
    )
    assert list(collapsedBoundary.coords) == [(0, 0), (0, 0)]
    print(f'snapped polygon, tolerance={tolerance}: [0,0,0,1]')
